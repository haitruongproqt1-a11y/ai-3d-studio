import os
import sys
import time
import base64
import json
import logging
import threading
import zipfile
import shutil
import urllib.request
import urllib.parse
import urllib.error
import socket

# Ensure portable cache paths on SSD H: are ALWAYS set FIRST before importing torch/TSR!
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, ".cache")
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["U2NET_HOME"] = os.path.join(CACHE_DIR, "u2net")



import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import webview

# Add TripoSR path
sys.path.append(os.path.join(os.path.dirname(__file__), "TripoSR"))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
import trimesh
import cv2
import rembg

# Monkeypatch huggingface_hub SpaceRuntime to safely handle ZeroGPU spaces
try:
    import huggingface_hub._space_api as _space_api
    def _safe_space_init(self, data):
        self.stage = data.get("stage")
        hw = data.get("hardware") or {}
        self.hardware = hw.get("current") if isinstance(hw, dict) else None
        self.requested_hardware = hw.get("requested") if isinstance(hw, dict) else None
        self.sleep_time = data.get("gcTimeout")
        self.storage = data.get("storage")
        self.raw = data
    _space_api.SpaceRuntime.__init__ = _safe_space_init
except Exception:
    pass

logging.basicConfig(level=logging.INFO)

APP_VERSION = "v1.9.0"
DEFAULT_GITHUB_REPO = "haitruongproqt1-a11y/ai-3d-studio"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "output_app")
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hardware Detection ──────────────────────────────────────────────────────
def detect_hardware():
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            return {
                "has_gpu": True,
                "device": "cuda:0",
                "name": gpu_name,
                "vram_gb": total_vram_gb,
                "preset": "gpu_high" if total_vram_gb >= 5.5 else "gpu_low",
                "status_text": f"NVIDIA {gpu_name} ({total_vram_gb} GB)",
            }
        except Exception:
            pass
    import multiprocessing
    n = multiprocessing.cpu_count()
    return {
        "has_gpu": False, "device": "cpu", "name": f"CPU ({n} luồng)",
        "vram_gb": 0, "preset": "cpu_fallback",
        "status_text": f"CPU ({n} luồng) – Không có GPU",
    }

HARDWARE_INFO = detect_hardware()
current_device = HARDWARE_INFO["device"]
model = None
_model_ready = threading.Event()

def load_ai_model():
    global model
    if model is not None and _model_ready.is_set():
        return True
    try:
        logging.info(f"Loading TripoSR on {current_device}…")
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(16384 if HARDWARE_INFO["has_gpu"] else 1024)
        model.to(current_device)
        _model_ready.set()
        logging.info("Model loaded ✓")
        return True
    except Exception as e:
        logging.error(f"Model load failed: {e}")
        return False

def load_config():
    cfg = {
        "github_repo": DEFAULT_GITHUB_REPO,
        "hf_token": "",
        "tripo_api_key": "",
        "meshy_api_key": ""
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                cfg.update(loaded)
    except Exception:
        pass
    return cfg

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"save_config: {e}")

# ── Translation helper ──────────────────────────────────────────────────────
def translate_to_en(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            res = json.loads(r.read().decode("utf-8"))
            translated = res[0][0][0]
            if translated:
                return translated
    except Exception:
        pass
    return text

# ── Texture & PBR helpers ───────────────────────────────────────────────────
def enhance_texture(img: Image.Image) -> Image.Image:
    try:
        img = ImageEnhance.Color(img).enhance(1.30)
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    except Exception:
        pass
    return img

# ── Library & Metadata helpers ──────────────────────────────────────────────
def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def _get_thumbnail_data_url(item_dir: str) -> str:
    thumb_path = os.path.join(item_dir, "thumb.jpg")
    if not os.path.exists(thumb_path):
        for c in ("input.png", "concept.png", "texture.png", "material_0.png"):
            cp = os.path.join(item_dir, c)
            if os.path.exists(cp):
                try:
                    with Image.open(cp) as im:
                        im = im.convert("RGB")
                        im.thumbnail((180, 180))
                        im.save(thumb_path, "JPEG", quality=75)
                    break
                except Exception:
                    pass
    if os.path.exists(thumb_path):
        try:
            with open(thumb_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            pass
    return ""

def _save_model_metadata(item_dir, name="", engine="RTX 3050 Offline", prompt="", source="image", input_img_path=None):
    try:
        os.makedirs(item_dir, exist_ok=True)
        # Generate thumbnail
        thumb_path = os.path.join(item_dir, "thumb.jpg")
        if not os.path.exists(thumb_path):
            img_cand = input_img_path if (input_img_path and os.path.exists(input_img_path)) else None
            if not img_cand:
                for c in ("input.png", "concept.png", "texture.png", "material_0.png"):
                    cp = os.path.join(item_dir, c)
                    if os.path.exists(cp):
                        img_cand = cp
                        break
            if img_cand:
                try:
                    with Image.open(img_cand) as im:
                        im = im.convert("RGB")
                        im.thumbnail((180, 180))
                        im.save(thumb_path, "JPEG", quality=75)
                except Exception as te:
                    logging.warning(f"thumb.jpg error: {te}")

        # Save metadata
        meta_path = os.path.join(item_dir, "meta.json")
        meta = {
            "id": os.path.basename(item_dir),
            "name": name or prompt or f"Mô hình 3D #{os.path.basename(item_dir)[-6:]}",
            "engine": engine,
            "prompt": prompt,
            "source": source,
            "created_at": time.time(),
            "created_str": time.strftime("%d/%m/%Y %H:%M", time.localtime())
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"_save_model_metadata error: {e}")


# ── AppApi ──────────────────────────────────────────────────────────────────
class AppApi:
    def __init__(self):
        self._window = None
        self.last_glb = ""
        self.last_obj = ""
        self.last_folder = OUTPUT_DIR
        self.config = load_config()
        self._tasks = {}

    def set_window(self, w):
        self._window = w

    def _find_latest_model(self):
        try:
            if not os.path.exists(OUTPUT_DIR):
                return None
            dirs = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR)
                    if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
            dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
            for d in dirs:
                glb = os.path.join(d, "model.glb")
                obj = os.path.join(d, "model.obj")
                if os.path.exists(glb) and os.path.getsize(glb) > 1000:
                    self.last_glb = glb
                    self.last_obj = obj if os.path.exists(obj) else ""
                    self.last_folder = d
                    return {
                        "has_model": True,
                        "folder": d,
                        "glb_path": glb,
                        "obj_path": self.last_obj,
                    }
        except Exception as e:
            logging.warning(f"Error finding latest model: {e}")
        return None

    def load_latest_model(self):
        if self.last_glb and os.path.exists(self.last_glb):
            try:
                with open(self.last_glb, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                return {
                    "success": True,
                    "glb_data": f"data:model/gltf-binary;base64,{b64}",
                    "folder": self.last_folder,
                    "glb_path": self.last_glb,
                    "obj_path": self.last_obj,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "Chưa có mô hình nào được tạo."}

    # ── 3D MODEL LIBRARY API ────────────────────────────────────────────────
    def get_library_items(self):
        items = []
        if not os.path.exists(OUTPUT_DIR):
            return {"success": True, "items": [], "count": 0, "total_size_mb": 0}

        dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
        total_bytes = 0

        for d in dirs:
            p = os.path.join(OUTPUT_DIR, d)
            glb = os.path.join(p, "model.glb")
            if not os.path.exists(glb) or os.path.getsize(glb) < 1000:
                continue

            glb_bytes = os.path.getsize(glb)
            obj = os.path.join(p, "model.obj")
            obj_bytes = os.path.getsize(obj) if os.path.exists(obj) else 0
            total_bytes += glb_bytes + obj_bytes

            meta_file = os.path.join(p, "meta.json")
            meta = {}
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            engine = meta.get("engine", "")
            engine_key = "local"
            if not engine:
                if d.startswith("img2three_"):
                    engine = "img2threejs (TRELLIS)"
                    engine_key = "img2threejs"
                elif d.startswith("txt_img2three_"):
                    engine = "img2threejs (Từ chữ)"
                    engine_key = "img2threejs"
                elif d.startswith("txt_"):
                    engine = "RTX 3050 (Từ chữ)"
                    engine_key = "local"
                elif d.startswith("meshy_"):
                    engine = "Meshy AI AAA"
                    engine_key = "meshy"
                elif d.startswith("tripo3d_"):
                    engine = "Tripo3D Studio"
                    engine_key = "tripo"
                elif d.startswith("cloud_"):
                    engine = "Hybrid Cloud"
                    engine_key = "cloud"
                else:
                    engine = "RTX 3050 Offline"
                    engine_key = "local"
            else:
                eng_lower = engine.lower()
                if "img2three" in eng_lower:
                    engine_key = "img2threejs"
                elif "meshy" in eng_lower:
                    engine_key = "meshy"
                elif "tripo" in eng_lower:
                    engine_key = "tripo"
                elif "cloud" in eng_lower:
                    engine_key = "cloud"
                else:
                    engine_key = "local"

            created_at = meta.get("created_at") or os.path.getmtime(glb)
            created_str = meta.get("created_str") or time.strftime("%d/%m/%Y %H:%M", time.localtime(created_at))
            name = meta.get("name") or meta.get("prompt") or f"Mô hình 3D #{d[-6:]}"

            thumb_url = _get_thumbnail_data_url(p)

            items.append({
                "id": d,
                "name": name,
                "engine": engine,
                "engine_key": engine_key,
                "prompt": meta.get("prompt", ""),
                "source": meta.get("source", "image"),
                "created_at": created_at,
                "created_str": created_str,
                "glb_size": format_file_size(glb_bytes),
                "glb_bytes": glb_bytes,
                "obj_size": format_file_size(obj_bytes) if obj_bytes else "",
                "has_obj": obj_bytes > 0,
                "thumb_url": thumb_url,
                "folder": p
            })

        items.sort(key=lambda x: x["created_at"], reverse=True)
        return {
            "success": True,
            "items": items,
            "count": len(items),
            "total_size_mb": round(total_bytes / (1024 * 1024), 1)
        }

    def load_library_item(self, item_id):
        p = os.path.join(OUTPUT_DIR, item_id)
        glb = os.path.join(p, "model.glb")
        if not os.path.exists(glb) or os.path.getsize(glb) < 1000:
            return {"success": False, "error": "Tệp mô hình không tồn tại hoặc đã bị xóa."}

        try:
            with open(glb, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            obj = os.path.join(p, "model.obj")
            self.last_glb = glb
            self.last_obj = obj if os.path.exists(obj) else ""
            self.last_folder = p

            meta_file = os.path.join(p, "meta.json")
            name = f"Mô hình #{item_id[-6:]}"
            engine = "3D Studio"
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        name = meta.get("name") or meta.get("prompt") or name
                        engine = meta.get("engine") or engine
                except Exception:
                    pass

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{b64}",
                "folder": p,
                "glb_path": glb,
                "obj_path": self.last_obj,
                "name": name,
                "engine": engine
            }
        except Exception as e:
            return {"success": False, "error": f"Không thể nạp mô hình: {e}"}

    def delete_library_item(self, item_id):
        p = os.path.join(OUTPUT_DIR, item_id)
        if not os.path.exists(p):
            return {"success": False, "error": "Mục không tồn tại hoặc đã bị xóa."}
        try:
            shutil.rmtree(p)
            if self.last_folder == p:
                self.last_folder = OUTPUT_DIR
                self.last_glb = ""
                self.last_obj = ""
            return {"success": True, "deleted_id": item_id}
        except Exception as e:
            return {"success": False, "error": f"Không thể xóa: {e}"}

    def open_library_item_folder(self, item_id):
        p = os.path.join(OUTPUT_DIR, item_id)
        return self.open_folder(p)

    def export_library_item(self, item_id, file_type="glb"):
        p = os.path.join(OUTPUT_DIR, item_id)
        src = os.path.join(p, f"model.{file_type}")
        if not os.path.exists(src):
            return {"success": False, "error": f"Không tìm thấy file model.{file_type}"}

        default_filename = f"{item_id}.{file_type}"
        file_filter = (f"{file_type.upper()} 3D Model (*.{file_type})", "All Files (*.*)")

        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=default_filename,
            file_types=file_filter,
        )
        if result:
            dst = result[0] if isinstance(result, (list, tuple)) else str(result)
            try:
                shutil.copy2(src, dst)
                if file_type == "obj":
                    for extra in ("texture.png", "model.mtl", "normal.png", "material_0.png", "material.mtl"):
                        s_extra = os.path.join(p, extra)
                        if os.path.exists(s_extra):
                            shutil.copy2(s_extra, os.path.join(os.path.dirname(dst), extra))
                return {"success": True, "saved_path": dst}
            except Exception as e:
                return {"success": False, "error": f"Không thể lưu: {e}"}
        return {"success": False, "canceled": True}

    def get_init_data(self):
        latest = self._find_latest_model()
        return {
            "version": APP_VERSION,
            "hardware": HARDWARE_INFO,
            "model_ready": _model_ready.is_set(),
            "config": self.config,
            "latest_model": latest,
        }

    # ── image picker ────────────────────────────────────────────────────────
    def select_image(self):
        if not self._window:
            return None
        types = ("Image Files (*.png;*.jpg;*.jpeg;*.webp)", "All Files (*.*)")
        result = self._window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=types)
        if result:
            fp = result[0]
            try:
                with open(fp, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                ext = os.path.splitext(fp)[1].lower().lstrip(".")
                if ext == "jpg":
                    ext = "jpeg"
                return {"path": fp, "dataUrl": f"data:image/{ext};base64,{b64}", "name": os.path.basename(fp)}
            except Exception as e:
                return {"error": str(e)}
        return None

    def is_model_ready(self):
        return _model_ready.is_set()

    # ── live progress helper (SAFE JSON ESCAPING) ─────────────────────────────
    def _progress(self, msg: str, pct: int = -1, task_id: str = None):
        if task_id and task_id in self._tasks:
            self._tasks[task_id]["msg"] = msg
            if pct >= 0:
                self._tasks[task_id]["pct"] = pct
        else:
            for tid, t in self._tasks.items():
                if t.get("status") == "running":
                    t["msg"] = msg
                    if pct >= 0:
                        t["pct"] = pct
        if self._window:
            try:
                safe = json.dumps(msg)
                self._window.evaluate_js(f"window._setProgress({safe}, {pct});")
            except Exception:
                pass

    # ── Async non-blocking task runner ───────────────────────────────────────
    def start_generate_3d(self, file_path, engine="local",
                          mc_resolution=None, bake_tex=False,
                          smooth=True, quality="pbr_1024"):
        task_id = str(time.time_ns())
        self._tasks[task_id] = {
            "status": "running", "msg": "Đang khởi động tiến trình…",
            "pct": 5, "result": None, "error": None
        }

        def _worker():
            try:
                res = self.generate_3d(
                    file_path, engine=engine, mc_resolution=mc_resolution,
                    bake_tex=bake_tex, smooth=smooth, quality=quality, task_id=task_id
                )
                if res.get("success"):
                    self._tasks[task_id] = {"status": "done", "msg": "Hoàn tất!", "pct": 100, "result": res}
                else:
                    self._tasks[task_id] = {"status": "error", "error": res.get("error", "Lỗi không xác định")}
            except Exception as e:
                logging.exception(f"task worker error: {e}")
                self._tasks[task_id] = {"status": "error", "error": str(e)}

        threading.Thread(target=_worker, daemon=True).start()
        return {"task_id": task_id}

    def start_generate_from_text(self, prompt, quality="pbr_1024", smooth=True, engine="local"):
        task_id = str(time.time_ns())
        self._tasks[task_id] = {
            "status": "running", "msg": "Đang phân tích câu lệnh…",
            "pct": 5, "result": None, "error": None
        }

        def _worker():
            try:
                res = self.generate_from_text(
                    prompt, quality=quality, smooth=smooth, engine=engine, task_id=task_id
                )
                if res.get("success"):
                    self._tasks[task_id] = {"status": "done", "msg": "Hoàn tất!", "pct": 100, "result": res}
                else:
                    self._tasks[task_id] = {"status": "error", "error": res.get("error", "Lỗi không xác định")}
            except Exception as e:
                logging.exception(f"text task worker error: {e}")
                self._tasks[task_id] = {"status": "error", "error": str(e)}

        threading.Thread(target=_worker, daemon=True).start()
        return {"task_id": task_id}

    def poll_task(self, task_id):
        return self._tasks.get(task_id, {"status": "not_found"})

    # ── image preprocessing ──────────────────────────────────────────────────
    def _preprocess(self, file_path: str) -> Image.Image:
        self._progress("🤖 AI đang phân đoạn & tách nền chuẩn u2net…", 10)
        orig = Image.open(file_path).convert("RGB")
        try:
            clean_rgba = rembg.remove(orig)
        except Exception as e:
            logging.warning(f"rembg remove warning: {e}")
            clean_rgba = remove_background(orig)
        fg = resize_foreground(clean_rgba, 0.85)
        arr = np.array(fg).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        return Image.fromarray((arr * 255).astype(np.uint8))

    # ── TEXT TO 3D PIPELINE ──────────────────────────────────────────────────
    def generate_from_text(self, prompt: str, quality="pbr_1024", smooth=True, engine="local", task_id=None):
        prompt = prompt.strip()
        if not prompt:
            return {"success": False, "error": "Vui lòng nhập mô tả văn bản cần tạo 3D!"}

        if engine == "meshy":
            return self._gen_meshy(prompt=prompt)
        elif engine == "img2threejs":
            try:
                self._progress(f"🌐 Đang dịch & tối ưu câu lệnh: '{prompt}'…", 8, task_id=task_id)
                prompt_en = translate_to_en(prompt)
                self._progress(f"🎨 AI đang phác họa hình ảnh 2D từ ý tưởng ('{prompt_en}')…", 15, task_id=task_id)
                ts = int(time.time())
                item_dir = os.path.join(OUTPUT_DIR, f"txt_img2three_{ts}")
                os.makedirs(item_dir, exist_ok=True)
                concept_file = os.path.join(item_dir, "concept.png")
                clean_prompt = f"{prompt_en}, centered studio 3d object, full view, pure white background, hyperrealistic, sharp focus, 8k"
                encoded_prompt = urllib.parse.quote(clean_prompt)
                url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true&seed={ts % 10000}"
                req = urllib.request.Request(url, headers={"User-Agent": "AI-3D-Studio/1.8.0"})
                with urllib.request.urlopen(req, timeout=25) as r, open(concept_file, "wb") as f_img:
                    shutil.copyfileobj(r, f_img)
                with open(concept_file, "rb") as f_img:
                    b64_img = base64.b64encode(f_img.read()).decode()
                concept_data_url = f"data:image/png;base64,{b64_img}"
                if self._window:
                    safe_url = json.dumps(concept_data_url)
                    self._window.evaluate_js(f"window._showConcept({safe_url});")
                res = self._gen_img2threejs(concept_file, target_dir=item_dir)
                if res.get("success"):
                    res["concept_data"] = concept_data_url
                    res["prompt"] = prompt
                    _save_model_metadata(
                        item_dir,
                        name=prompt,
                        engine="img2threejs (Từ văn bản)",
                        prompt=prompt,
                        source="text",
                        input_img_path=concept_file
                    )
                return res
            except Exception as e:
                logging.exception("generate_from_text img2threejs error")
                return {"success": False, "error": f"Lỗi tạo 3D từ văn bản img2threejs: {e}"}

        # Ensure AI model is ready on GPU
        if model is None or not _model_ready.is_set():
            self._progress("⏳ Đang nạp model AI vào GPU RTX 3050 (~15s lần đầu)…", 5, task_id=task_id)
            ok = load_ai_model()
            if not ok or model is None:
                return {"success": False, "error": "Model AI nạp thất bại. Vui lòng thử lại!"}

        try:
            # 1. Translate prompt to English for highest concept accuracy
            self._progress(f"🌐 Đang dịch & tối ưu câu lệnh: '{prompt}'…", 8, task_id=task_id)
            prompt_en = translate_to_en(prompt)

            # 2. Fetch studio-grade 2D concept image
            self._progress(f"🎨 AI đang phác họa hình ảnh 2D từ ý tưởng ('{prompt_en}')…", 15, task_id=task_id)
            ts = int(time.time())
            item_dir = os.path.join(OUTPUT_DIR, f"txt_{ts}")
            os.makedirs(item_dir, exist_ok=True)
            concept_file = os.path.join(item_dir, "concept.png")

            clean_prompt = f"{prompt_en}, centered studio 3d object, full view, pure white background, hyperrealistic, sharp focus, 8k"
            encoded_prompt = urllib.parse.quote(clean_prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true&seed={ts % 10000}"

            req = urllib.request.Request(url, headers={"User-Agent": "AI-3D-Studio/1.7.0"})
            with urllib.request.urlopen(req, timeout=25) as r, open(concept_file, "wb") as f_img:
                shutil.copyfileobj(r, f_img)

            # Send concept image to UI preview immediately
            with open(concept_file, "rb") as f_img:
                b64_img = base64.b64encode(f_img.read()).decode()
            concept_data_url = f"data:image/png;base64,{b64_img}"

            if self._window:
                safe_url = json.dumps(concept_data_url)
                self._window.evaluate_js(f"window._showConcept({safe_url});")

            # 3. Feed directly into RTX 3050 3D reconstruction!
            self._progress("⚡ Đưa hình phác họa vào GPU RTX 3050 tái tạo 3D…", 28, task_id=task_id)
            res = self._gen_local(
                concept_file,
                quality=quality,
                smooth=smooth,
                target_dir=item_dir,
                task_id=task_id
            )
            if res.get("success"):
                res["concept_data"] = concept_data_url
                res["prompt"] = prompt
                _save_model_metadata(
                    item_dir,
                    name=prompt,
                    engine=res.get("engine_used", "RTX 3050 (Từ chữ)"),
                    prompt=prompt,
                    source="text",
                    input_img_path=concept_file
                )
            return res

        except Exception as e:
            logging.exception("generate_from_text error")
            return {"success": False, "error": f"Lỗi tạo ảnh phác họa: {e}\n(Bạn có thể chuyển sang thẻ 'Từ Hình Ảnh' để nạp ảnh trực tiếp)"}

    # ── main 3D generation router ────────────────────────────────────────────
    def generate_3d(self, file_path, engine="local",
                    mc_resolution=None, bake_tex=False,
                    smooth=True, quality="pbr_1024", task_id=None):
        if engine == "meshy":
            return self._gen_meshy(file_path=file_path)
        elif engine == "tripo":
            return self._gen_tripo3d(file_path)
        elif engine == "img2threejs":
            return self._gen_img2threejs(file_path)
        elif engine == "cloud":
            return self._gen_cloud(file_path, quality)
        return self._gen_local(file_path, quality=quality, smooth=smooth, task_id=task_id)

    # ── ENGINE 1: LOCAL GPU (RTX 3050 TAUBIN SMOOTHING + 1024 PBR ATLAS) ─────
    def _gen_local(self, file_path, quality="pbr_1024", smooth=True, target_dir=None, task_id=None):
        global model, current_device
        if model is None or not _model_ready.is_set():
            self._progress("⏳ Đang nạp model AI vào GPU RTX 3050 (~15s lần đầu)…", 5, task_id=task_id)
            ok = load_ai_model()
            if not ok or model is None:
                return {"success": False, "error": "Model AI nạp thất bại. Vui lòng kiểm tra dung lượng VRAM!"}

        try:
            do_bake = (quality == "pbr_1024" or quality == "bake")
            mc_res = 320 if quality == "ultra_320" else 256

            self._progress("🖼 Khử bóng đổ, tách viền & tiền xử lý ảnh chuẩn…", 15)
            image = self._preprocess(file_path)

            if target_dir:
                item_dir = target_dir
            else:
                ts = int(time.time())
                item_dir = os.path.join(OUTPUT_DIR, str(ts))
                os.makedirs(item_dir, exist_ok=True)

            image.save(os.path.join(item_dir, "input.png"))

            self._progress(f"🧠 RTX 3050 suy luận Tensor không gian ({HARDWARE_INFO['name']})…", 30)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                scene_codes = model([image], device=current_device)

            self._progress(f"⚙️ Tái tạo lưới Marching Cubes ({mc_res}x{mc_res} Voxels)…", 45)
            meshes = model.extract_mesh(
                scene_codes,
                has_vertex_color=(not do_bake),
                resolution=mc_res,
            )

            # Taubin non-shrinking smoothing: smooths out stepped polygons while preserving volume & sharp contours!
            if smooth:
                self._progress("✨ Làm mịn bề mặt Taubin (khử bậc thang, giữ nguyên thể tích)…", 55)
                try:
                    trimesh.smoothing.filter_taubin(meshes[0], lamb=0.5, nu=-0.53, iterations=10)
                except Exception:
                    try:
                        trimesh.smoothing.filter_laplacian(meshes[0], lamb=0.08, iterations=2)
                    except Exception as e:
                        logging.warning(f"Smooth skipped: {e}")

            out_obj = os.path.join(item_dir, "model.obj")
            out_glb = os.path.join(item_dir, "model.glb")
            out_tex = os.path.join(item_dir, "texture.png")

            if do_bake and HARDWARE_INFO["has_gpu"]:
                self._progress("🎨 Đang nướng UV Atlas Texture PBR 1024px…", 65)
                from tsr.bake_texture import bake_texture as _bake
                import xatlas

                # Bake in canonical coordinates to guarantee 100% texture alignment
                bake = _bake(
                    meshes[0], model, scene_codes[0], 1024,
                    progress_cb=lambda m, p: self._progress(m, p, task_id=task_id)
                )

                # Center pivot for Unity & Blender export AFTER baking
                centroid = meshes[0].bounding_box.centroid.copy()
                centered_verts = meshes[0].vertices - centroid

                xatlas.export(
                    out_obj,
                    centered_verts[bake["vmapping"]],
                    bake["indices"],
                    bake["uvs"],
                    meshes[0].vertex_normals[bake["vmapping"]],
                )
                self._progress("🖌 Tối ưu màu sắc rực rỡ PBR…", 88)
                tex = Image.fromarray((bake["colors"] * 255).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)
                tex = enhance_texture(tex)
                tex.save(out_tex)
                loaded = trimesh.load(out_obj)
                mat = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=tex, roughnessFactor=0.30, metallicFactor=0.05
                )
                loaded.visual.material = mat
                loaded.export(out_glb)
                engine_label = "RTX 3050 Offline (Đỉnh cao PBR 1024px + Làm mịn Taubin)"
            else:
                # Fast vertex colors mode:
                try:
                    if hasattr(meshes[0].visual, 'vertex_colors') and meshes[0].visual.vertex_colors is not None:
                        vc = meshes[0].visual.vertex_colors.astype(np.float32)
                        rgb = vc[:, :3] / 255.0
                        mean = np.mean(rgb, axis=-1, keepdims=True)
                        # Saturation boost (1.30) + contrast boost (1.12)
                        rgb = np.clip(mean + 1.30 * (rgb - mean), 0.0, 1.0)
                        rgb = np.clip((rgb - 0.5) * 1.12 + 0.5, 0.0, 1.0)
                        vc[:, :3] = rgb * 255.0
                        meshes[0].visual.vertex_colors = vc.astype(np.uint8)
                except Exception as e:
                    logging.warning(f"Vertex color boost skipped: {e}")

                try:
                    meshes[0].vertices -= meshes[0].bounding_box.centroid
                    meshes[0].fix_normals()
                except Exception:
                    pass

                self._progress("💾 Đang xuất file GLB & OBJ chuẩn định dạng…", 88)
                meshes[0].export(out_glb)
                meshes[0].export(out_obj)
                res_title = "Ultra HD 320" if mc_res == 320 else "Siêu tốc 256"
                engine_label = f"RTX 3050 Offline ({res_title})"

            self.last_glb = out_glb
            self.last_obj = out_obj
            self.last_folder = item_dir

            _save_model_metadata(
                item_dir,
                name=os.path.splitext(os.path.basename(file_path))[0],
                engine=engine_label,
                source="image",
                input_img_path=os.path.join(item_dir, "input.png")
            )

            self._progress("✅ Hoàn tất! Đang nạp mô hình 3D vào Studio…", 98)
            with open(out_glb, "rb") as f:
                glb_b64 = base64.b64encode(f.read()).decode()

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{glb_b64}",
                "glb_path": out_glb,
                "obj_path": out_obj,
                "folder": item_dir,
                "engine_used": engine_label,
            }

        except Exception as e:
            logging.exception("_gen_local error")
            return {"success": False, "error": str(e)}

    # ── ENGINE 2: MESHY AI PRO (200 CREDITS FREE MỖI THÁNG – CHUẨN GAME AAA) ─
    def _gen_meshy(self, prompt=None, file_path=None):
        meshy_key = self.config.get("meshy_api_key", "").strip()
        if not meshy_key:
            return {
                "success": False,
                "error": (
                    "🔑 Chưa có Meshy AI API Key!\n\n"
                    "💡 CÁCH LẤY 200 CREDITS MIỄN PHÍ MỖI THÁNG:\n"
                    "1. Mở trang https://meshy.ai đăng ký tài khoản (miễn phí 100%)\n"
                    "2. Nhấn vào Avatar góc trên -> API Keys -> Copy Key (dạng 'msy_...')\n"
                    "3. Bấm nút '⚙️ Cài đặt' ở góc trên AI 3D Studio để dán key!\n\n"
                    "👉 Hoặc dùng tab '⚡ GPU RTX 3050' để tạo 100% Offline trên máy bạn!"
                )
            }

        try:
            ts = int(time.time())
            item_dir = os.path.join(OUTPUT_DIR, f"meshy_{ts}")
            os.makedirs(item_dir, exist_ok=True)
            local_glb = os.path.join(item_dir, "model.glb")
            local_obj = os.path.join(item_dir, "model.obj")

            headers = {
                "Authorization": f"Bearer {meshy_key}",
                "Content-Type": "application/json",
                "User-Agent": "AI-3D-Studio/1.7.0"
            }

            if prompt:
                # Text to 3D mode
                self._progress(f"🚀 [Meshy AI] Khởi tạo tạo 3D từ văn bản: '{prompt}'…", 10)
                prompt_en = translate_to_en(prompt)
                task_payload = json.dumps({
                    "mode": "preview",
                    "prompt": f"{prompt_en}, 3d game asset, highly detailed, realistic, 8k",
                    "art_style": "realistic"
                }).encode("utf-8")
                api_url = "https://api.meshy.ai/v2/text-to-3d"
            else:
                # Image to 3D mode
                self._progress("🚀 [Meshy AI] Đang tải ảnh lên máy chủ Meshy Studio AAA…", 10)
                with open(file_path, "rb") as f:
                    b64_img = base64.b64encode(f.read()).decode()
                ext = os.path.splitext(file_path)[1].lower().lstrip(".")
                mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
                data_uri = f"data:image/{mime};base64,{b64_img}"

                task_payload = json.dumps({
                    "image_url": data_uri,
                    "enable_pbr": True
                }).encode("utf-8")
                api_url = "https://api.meshy.ai/v2/image-to-3d"

            req = urllib.request.Request(api_url, data=task_payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                res = json.loads(r.read().decode("utf-8"))
                task_id = res.get("result")
                if not task_id:
                    return {"success": False, "error": f"Meshy API không trả về task ID: {res}"}

            poll_url = f"{api_url}/{task_id}"
            poll_req = urllib.request.Request(poll_url, headers=headers)

            glb_url = None
            obj_url = None
            for i in range(120):
                time.sleep(3)
                with urllib.request.urlopen(poll_req, timeout=15) as r_poll:
                    poll_data = json.loads(r_poll.read().decode("utf-8"))
                    status = poll_data.get("status")
                    progress = poll_data.get("progress", 0)

                    if status == "IN_PROGRESS":
                        prog_val = 15 + int(progress * 0.75)
                        self._progress(f"✨ [Meshy AI] Đang tái tạo lưới Quad-mesh & vân PBR ({progress}%)…", prog_val)
                    elif status == "SUCCEEDED":
                        urls = poll_data.get("model_urls", {})
                        glb_url = urls.get("glb")
                        obj_url = urls.get("obj")
                        break
                    elif status == "FAILED":
                        err_msg = poll_data.get("task_error", {}).get("message", "Thất bại")
                        return {"success": False, "error": f"Meshy AI báo lỗi: {err_msg}"}

            if not glb_url:
                return {"success": False, "error": "Quá thời gian phản hồi từ Meshy AI (hơn 5 phút)."}

            self._progress("📥 [Meshy AI] Đang tải mô hình PBR chất lượng cao…", 92)
            req_dl = urllib.request.Request(glb_url, headers={"User-Agent": "AI-3D-Studio"})
            with urllib.request.urlopen(req_dl, timeout=60) as r_dl, open(local_glb, "wb") as f_out:
                shutil.copyfileobj(r_dl, f_out)

            if obj_url:
                try:
                    req_obj = urllib.request.Request(obj_url, headers={"User-Agent": "AI-3D-Studio"})
                    with urllib.request.urlopen(req_obj, timeout=60) as r_obj, open(local_obj, "wb") as f_out_obj:
                        shutil.copyfileobj(r_obj, f_out_obj)
                except Exception:
                    pass

            if not os.path.exists(local_obj):
                try:
                    m = trimesh.load(local_glb)
                    m.export(local_obj)
                except Exception:
                    pass

            self.last_glb = local_glb
            self.last_obj = local_obj if os.path.exists(local_obj) else local_glb
            self.last_folder = item_dir

            _save_model_metadata(
                item_dir,
                name=prompt or (os.path.splitext(os.path.basename(file_path))[0] if file_path else "Mô hình Meshy"),
                engine="Meshy AI Pro (Chuẩn Game AAA)",
                prompt=prompt or "",
                source="text" if prompt else "image",
                input_img_path=file_path if file_path else None
            )

            self._progress("✅ Hoàn tất! Đang nạp mô hình Meshy Studio PBR…", 98)
            with open(local_glb, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{b64}",
                "glb_path": local_glb,
                "obj_path": self.last_obj,
                "folder": item_dir,
                "engine_used": "Meshy AI Pro (Chất lượng Game AAA PBR Chuẩn)"
            }

        except urllib.error.HTTPError as he:
            body = he.read().decode("utf-8", errors="ignore")
            if "invalid" in body.lower() or he.code == 401:
                return {
                    "success": False,
                    "error": "⚠️ Meshy API Key không đúng hoặc chưa kích hoạt.\n👉 Vui lòng vào https://meshy.ai -> API Keys để tạo key mới và dán vào '⚙️ Cài đặt'!"
                }
            return {"success": False, "error": f"Lỗi Meshy AI ({he.code}): {body}"}
        except Exception as e:
            logging.exception("_gen_meshy error")
            return {"success": False, "error": str(e)}

    # ── ENGINE 3: TRIPO3D STUDIO PRO ─────────────────────────────────────────
    def _gen_tripo3d(self, file_path):
        tripo_key = self.config.get("tripo_api_key", "").strip()
        if not tripo_key:
            return {
                "success": False,
                "error": "Chưa có Tripo3D API Key!\n👉 Vui lòng nhấn '⚙️ Cài đặt' để dán API Key.\n(Hoặc dùng chế độ '⚡ GPU RTX 3050' Miễn Phí 100% vĩnh viễn trên máy bạn!)"
            }

        try:
            self._progress("🚀 [Tripo3D] Đang tải ảnh lên máy chủ Studio AI…", 10)
            boundary = f"----WebKitFormBoundary{int(time.time()*1000)}"
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            filename = os.path.basename(file_path)
            body = bytearray()
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
            body.extend(b"Content-Type: image/png\r\n\r\n")
            body.extend(file_bytes)
            body.extend(f"\r\n--{boundary}--\r\n".encode())

            req = urllib.request.Request(
                "https://api.tripo3d.ai/v2/openapi/upload",
                data=body,
                headers={
                    "Authorization": f"Bearer {tripo_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": "AI-3D-Studio"
                }
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                upload_res = json.loads(r.read().decode("utf-8"))
                image_token = upload_res.get("data", {}).get("image_token")
                if not image_token:
                    return {"success": False, "error": "Tripo3D không nhận được token ảnh."}

            self._progress("🧠 [Tripo3D] Khởi tạo tác vụ tạo 3D Studio PBR…", 25)
            task_payload = json.dumps({
                "type": "image_to_model",
                "file": {"type": "png", "file_token": image_token}
            }).encode()
            req2 = urllib.request.Request(
                "https://api.tripo3d.ai/v2/openapi/task",
                data=task_payload,
                headers={
                    "Authorization": f"Bearer {tripo_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "AI-3D-Studio"
                }
            )
            with urllib.request.urlopen(req2, timeout=30) as r2:
                task_res = json.loads(r2.read().decode("utf-8"))
                task_id = task_res.get("data", {}).get("task_id")
                if not task_id:
                    msg = task_res.get("message", "Lỗi tạo tác vụ Tripo3D")
                    return {"success": False, "error": f"Tripo3D API: {msg}"}

            glb_url = None
            poll_req = urllib.request.Request(
                f"https://api.tripo3d.ai/v2/openapi/task/{task_id}",
                headers={"Authorization": f"Bearer {tripo_key}", "User-Agent": "AI-3D-Studio"}
            )
            for _ in range(60):
                time.sleep(2)
                with urllib.request.urlopen(poll_req, timeout=15) as r3:
                    poll_res = json.loads(r3.read().decode("utf-8"))
                    p_data = poll_res.get("data", {})
                    p_status = p_data.get("status")
                    p_prog = p_data.get("progress", 0)
                    if p_status == "running":
                        prog_val = 25 + int(p_prog * 0.65)
                        self._progress(f"✨ [Tripo3D] Đang phủ vân PBR & lưới Quad-mesh ({p_prog}%)…", prog_val)
                    elif p_status == "success":
                        out = p_data.get("output", {})
                        glb_url = out.get("pbr_model") or out.get("model")
                        break
                    elif p_status == "failed":
                        return {"success": False, "error": f"Tripo3D báo lỗi: {p_data.get('message', 'Thất bại')}"}

            if not glb_url:
                return {"success": False, "error": "Quá thời gian phản hồi từ Tripo3D (hơn 2 phút)."}

            self._progress("📥 [Tripo3D] Đang tải mô hình PBR chất lượng cao…", 92)
            ts = int(time.time())
            item_dir = os.path.join(OUTPUT_DIR, f"tripo3d_{ts}")
            os.makedirs(item_dir, exist_ok=True)
            local_glb = os.path.join(item_dir, "model.glb")
            local_obj = os.path.join(item_dir, "model.obj")

            req_dl = urllib.request.Request(glb_url, headers={"User-Agent": "AI-3D-Studio"})
            with urllib.request.urlopen(req_dl, timeout=60) as r_dl, open(local_glb, "wb") as f_out:
                shutil.copyfileobj(r_dl, f_out)

            try:
                m = trimesh.load(local_glb)
                m.export(local_obj)
            except Exception:
                pass

            self.last_glb = local_glb
            self.last_obj = local_obj if os.path.exists(local_obj) else local_glb
            self.last_folder = item_dir

            _save_model_metadata(
                item_dir,
                name=os.path.splitext(os.path.basename(file_path))[0],
                engine="Tripo3D Studio Pro",
                source="image",
                input_img_path=file_path
            )

            self._progress("✅ Hoàn tất! Đang nạp mô hình Studio PBR…", 98)
            with open(local_glb, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{b64}",
                "glb_path": local_glb,
                "obj_path": self.last_obj,
                "folder": item_dir,
                "engine_used": "Tripo3D Studio Pro (Chân thực 100% PBR)"
            }
        except urllib.error.HTTPError as he:
            body = he.read().decode("utf-8", errors="ignore")
            if "2010" in body or "purchase more credit" in body or "credit" in body.lower():
                return {
                    "success": False,
                    "error": (
                        "⚠️ Tripo3D thông báo: Tài khoản của bạn có 0 credit trên cổng Developer API.\n\n"
                        "💡 LÝ DO:\n"
                        "Tripo3D cho tạo miễn phí trên web tripo3d.ai, nhưng cổng kết nối API thì họ bắt buộc phải mua gói trả phí ($10-$30/tháng).\n\n"
                        "👉 3 CÁCH SỬ DỤNG TỐT NHẤT CHO BẠN:\n"
                        "1. Dùng thẻ '💎 Meshy AI': Meshy tặng 200 credits miễn phí mỗi tháng dùng trực tiếp trong ứng dụng!\n"
                        "2. Truy cập web tripo3d.ai dùng 300 credit miễn phí, tải file .glb về và bấm '📂 Nạp 3D ngoài'!\n"
                        "3. Dùng thẻ '⚡ GPU RTX 3050': Chạy 100% trên máy tính của bạn, KHÔNG CẦN MẠNG, MIỄN PHÍ VĨNH VIỄN!"
                    )
                }
            return {"success": False, "error": f"Lỗi Tripo3D ({he.code}): {body}"}
        except Exception as e:
            logging.exception("_gen_tripo3d error")
            return {"success": False, "error": str(e)}

    # ── ENGINE 4: CLOUD MULTI-VIEW (HUGGING FACE ZEROGPU) ────────────────────
    def _gen_cloud(self, file_path, quality="pbr_1024"):
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            return {"success": False, "error": "Thiếu thư viện gradio_client. Vui lòng dùng chế độ GPU Offline!"}

        try:
            hf_token = self.config.get("hf_token", "").strip() or None
            self._progress("☁️ Đang kết nối máy chủ AI Cloud…", 5)
            client = Client("TencentARC/InstantMesh", token=hf_token, httpx_kwargs={"timeout": 300.0})

            self._progress("🖼 Đang tải ảnh lên Cloud và tách nền…", 15)
            prep = client.predict(input_image=handle_file(file_path), api_name="/preprocess")

            steps = 50 if quality in ("pbr_1024", "ultra_320") else 30
            self._progress(f"🔮 Tái tạo đa góc nhìn chi tiết ({steps} bước)…", 35)
            client.predict(input_image=handle_file(prep), sample_steps=steps,
                           sample_seed=42, api_name="/generate_mvs")

            self._progress("⚙️ Cloud đang ghép Mesh 3D và texture…", 75)
            obj_path, glb_path = client.predict(api_name="/make3d")

            ts = int(time.time())
            item_dir = os.path.join(OUTPUT_DIR, f"cloud_{ts}")
            os.makedirs(item_dir, exist_ok=True)
            local_glb = os.path.join(item_dir, "model.glb")
            local_obj = os.path.join(item_dir, "model.obj")
            shutil.copy2(glb_path, local_glb)
            shutil.copy2(obj_path, local_obj)

            self.last_glb = local_glb
            self.last_obj = local_obj
            self.last_folder = item_dir

            _save_model_metadata(
                item_dir,
                name=os.path.splitext(os.path.basename(file_path))[0],
                engine=f"Cloud Multi-View ({steps} Steps)",
                source="image",
                input_img_path=file_path
            )

            self._progress("✅ Hoàn tất! Đang nạp mô hình 3D…", 95)
            with open(local_glb, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{b64}",
                "glb_path": local_glb,
                "obj_path": local_obj,
                "folder": item_dir,
                "engine_used": f"Cloud Multi-View ({steps} Steps)",
            }
        except Exception as e:
            logging.exception("_gen_cloud error")
            msg = str(e)
            if "quota" in msg.lower() or "ZeroGPU" in msg or "429" in msg or "timed out" in msg.lower():
                msg = ("⚠️ Máy chủ Cloud miễn phí quá tải hoặc hết Quota dùng chung.\n\n"
                       "👉 GIẢI PHÁP TỐT NHẤT: Chuyển sang thẻ '⚡ GPU RTX 3050' để tạo ngay trên máy bạn 100% offline, miễn phí vĩnh viễn!")
            return {"success": False, "error": msg}

    # ── ENGINE 5: IMG2THREEJS (TRELLIS 3D RECONSTRUCTION + THREE.JS PIPELINE) ─
    def _gen_img2threejs(self, file_path, target_dir=None):
        try:
            # Safe monkeypatch to prevent KeyError: 'storage' with ZeroGPU SpaceRuntime
            try:
                import huggingface_hub._space_api as space_api
                def _safe_space_init(self, data):
                    self.stage = data.get("stage")
                    hw = data.get("hardware") or {}
                    self.hardware = hw.get("current") if isinstance(hw, dict) else None
                    self.requested_hardware = hw.get("requested") if isinstance(hw, dict) else None
                    self.sleep_time = data.get("gcTimeout")
                    self.storage = data.get("storage")
                    self.raw = data
                space_api.SpaceRuntime.__init__ = _safe_space_init
            except Exception:
                pass

            from gradio_client import Client, handle_file
        except ImportError:
            return {"success": False, "error": "Thiếu thư viện gradio_client. Vui lòng dùng chế độ GPU Offline!"}

        try:
            hf_token = self.config.get("hf_token", "").strip() or None
            self._progress("🎨 [img2threejs] Đang kết nối máy chủ TRELLIS Space…", 10)
            client = Client("trellis-community/TRELLIS", token=hf_token, httpx_kwargs={"timeout": 300.0}, verbose=False)

            self._progress("🔄 [img2threejs] Khởi tạo phiên làm việc scratch session…", 18)
            try:
                client.predict(api_name="/start_session")
            except Exception as se:
                logging.warning(f"img2threejs start_session notice: {se}")

            self._progress("🧠 [img2threejs] Khởi chạy tái tạo không gian 3D trên ZeroGPU…", 25)
            primary = handle_file(file_path)
            job = client.submit(
                image=primary,
                multiimages=[],
                seed=0,
                ss_guidance_strength=7.5,
                ss_sampling_steps=12,
                slat_guidance_strength=3.0,
                slat_sampling_steps=12,
                multiimage_algo="stochastic",
                mesh_simplify=0.95,
                texture_size=1024,
                api_name="/generate_and_extract_glb",
            )

            poll_count = 0
            while not job.done():
                time.sleep(2)
                poll_count += 1
                try:
                    st = job.status()
                    code_str = str(st.code) if hasattr(st, "code") else ""
                    if "STARTING" in code_str:
                        self._progress("⏳ [img2threejs] Đang chờ cấp phát GPU Cloud…", 28)
                    elif "PROCESSING" in code_str:
                        prog_val = min(35 + poll_count * 2, 85)
                        self._progress(f"✨ [img2threejs] ZeroGPU đang tái tạo cấu trúc Mesh 3D ({prog_val}%)…", prog_val)
                    elif "FINISHED" in code_str:
                        self._progress("📥 [img2threejs] Đang nạp tệp mô hình 3D về máy tính…", 90)
                except Exception:
                    pass

            result = job.result()

            glb_source = None
            if isinstance(result, (list, tuple)):
                for item in result:
                    if isinstance(item, str) and item.lower().endswith(".glb") and os.path.exists(item):
                        glb_source = item
                        break
                    elif isinstance(item, dict):
                        cand = item.get("video")
                        if isinstance(cand, str) and cand.lower().endswith(".glb") and os.path.exists(cand):
                            glb_source = cand
                            break
            elif isinstance(result, str) and result.lower().endswith(".glb") and os.path.exists(result):
                glb_source = result

            if not glb_source or not os.path.exists(glb_source):
                return {"success": False, "error": f"Không tìm thấy file GLB từ phản hồi TRELLIS: {result}"}

            self._progress("📦 [img2threejs] Đang chuẩn bị tệp mô hình GLB và OBJ…", 92)
            if target_dir:
                item_dir = target_dir
            else:
                ts = int(time.time())
                item_dir = os.path.join(OUTPUT_DIR, f"img2three_{ts}")
                os.makedirs(item_dir, exist_ok=True)

            local_glb = os.path.join(item_dir, "model.glb")
            local_obj = os.path.join(item_dir, "model.obj")
            shutil.copy2(glb_source, local_glb)

            # Export clean triangulated smooth-normal OBJ via trimesh
            try:
                scene_or_mesh = trimesh.load(local_glb, force="scene")
                mesh = scene_or_mesh.to_mesh() if hasattr(scene_or_mesh, "to_mesh") else scene_or_mesh
                if hasattr(mesh, "merge_vertices"):
                    mesh.merge_vertices()
                if hasattr(mesh, "fix_normals"):
                    mesh.fix_normals()
                mesh.export(local_obj)
            except Exception as oe:
                logging.warning(f"OBJ export notice: {oe}")
                if not os.path.exists(local_obj):
                    try:
                        trimesh.load(local_glb).export(local_obj)
                    except Exception:
                        pass

            self.last_glb = local_glb
            self.last_obj = local_obj if os.path.exists(local_obj) else local_glb
            self.last_folder = item_dir

            _save_model_metadata(
                item_dir,
                name=os.path.splitext(os.path.basename(file_path))[0],
                engine="img2threejs (TRELLIS 3D)",
                source="image",
                input_img_path=file_path
            )

            self._progress("✅ Hoàn tất! Đang nạp mô hình img2threejs vào Studio…", 98)
            with open(local_glb, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{b64}",
                "glb_path": local_glb,
                "obj_path": self.last_obj,
                "folder": item_dir,
                "engine_used": "img2threejs (TRELLIS & Code Reconstruction)",
            }

        except Exception as e:
            logging.exception("_gen_img2threejs error")
            msg = str(e)
            if "storage" in msg.lower():
                msg = "⚠️ Phiên kết nối Hugging Face bị gián đoạn. Vui lòng bấm thử lại!"
            elif "quota" in msg.lower() or "zerogpu" in msg.lower() or "429" in msg or "timed out" in msg.lower() or "timeout" in msg.lower() or "exceeded" in msg.lower():
                msg = (
                    "⚠️ Hugging Face ZeroGPU đã hết hạn mức miễn phí hôm nay (giới hạn 1-2 lần/ngày).\n\n"
                    "👉 ĐỂ TẠO KHÔNG GIỚI HẠN:\n"
                    "Chuyển sang thẻ '⚡ RTX 3050 (Offline)' trên thanh công cụ! Chạy trực tiếp 100% trên card đồ họa máy bạn, KHÔNG CẦN MẠNG, TẠO BAO NHIÊU LẦN TÙY THÍCH!"
                )
            return {"success": False, "error": msg}

    # ── settings ─────────────────────────────────────────────────────────────
    def save_settings(self, hf_token=None, tripo_api_key=None, meshy_api_key=None):
        if hf_token is not None:
            self.config["hf_token"] = hf_token.strip()
        if tripo_api_key is not None:
            self.config["tripo_api_key"] = tripo_api_key.strip()
        if meshy_api_key is not None:
            self.config["meshy_api_key"] = meshy_api_key.strip()
        save_config(self.config)
        return {"success": True}

    # ── file operations (ALWAYS ACCESSIBLE) ───────────────────────────────────
    def open_img2threejs_dir(self):
        p = os.path.join(APP_DIR, "img2threejs")
        return self.open_folder(p)

    def open_folder(self, folder=None):
        target = folder or self.last_folder or OUTPUT_DIR
        if not os.path.exists(target):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            target = OUTPUT_DIR
        try:
            os.startfile(os.path.normpath(target))
            return {"success": True}
        except Exception:
            os.system(f'explorer.exe "{os.path.normpath(target)}"')
            return {"success": True}

    def export_file(self, file_type="glb"):
        src = self.last_glb if file_type == "glb" else self.last_obj
        if not src or not os.path.exists(src):
            cand = os.path.join(self.last_folder, f"model.{file_type}")
            if os.path.exists(cand):
                src = cand
            else:
                return {"success": False, "error": "Chưa có mô hình 3D! Vui lòng nạp ảnh hoặc nhập văn bản để tạo mô hình trước."}

        default_filename = f"model_3d_{int(time.time())}.{file_type}"
        file_filter = (f"{file_type.upper()} 3D Model (*.{file_type})", "All Files (*.*)")

        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=default_filename,
            file_types=file_filter,
        )
        if result:
            dst = result[0] if isinstance(result, (list, tuple)) else str(result)
            try:
                shutil.copy2(src, dst)
                if file_type == "obj":
                    src_dir = os.path.dirname(src)
                    dst_dir = os.path.dirname(dst)
                    for extra in ("texture.png", "model.mtl", "normal.png"):
                        s_extra = os.path.join(src_dir, extra)
                        if os.path.exists(s_extra):
                            shutil.copy2(s_extra, os.path.join(dst_dir, extra))
                return {"success": True, "saved_path": dst}
            except Exception as e:
                return {"success": False, "error": f"Không thể lưu: {e}"}
        return {"success": False, "canceled": True}

    def load_external_model(self):
        if not self._window:
            return {"success": False, "error": "Cửa sổ chưa sẵn sàng"}
        types = ("3D Model Files (*.glb;*.gltf;*.obj)", "All Files (*.*)")
        result = self._window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=types)
        if not result:
            return {"success": False, "canceled": True}
        fp = result[0]
        try:
            ext = os.path.splitext(fp)[1].lower()
            item_dir = os.path.dirname(fp)
            local_obj = ""
            if ext == ".glb":
                local_glb = fp
                obj_cand = os.path.splitext(fp)[0] + ".obj"
                if os.path.exists(obj_cand):
                    local_obj = obj_cand
                with open(fp, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                glb_data = f"data:model/gltf-binary;base64,{b64}"
            elif ext in (".obj", ".gltf"):
                m = trimesh.load(fp)
                temp_glb = os.path.join(OUTPUT_DIR, "imported_temp.glb")
                m.export(temp_glb)
                local_glb = temp_glb
                local_obj = fp
                with open(temp_glb, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                glb_data = f"data:model/gltf-binary;base64,{b64}"
            else:
                return {"success": False, "error": "Định dạng không hỗ trợ. Vui lòng chọn file .glb hoặc .obj"}

            self.last_glb = local_glb
            self.last_obj = local_obj or local_glb
            self.last_folder = item_dir
            return {
                "success": True,
                "glb_data": glb_data,
                "folder": item_dir,
                "filename": os.path.basename(fp)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def open_external_url(self, url):
        try:
            import webbrowser
            webbrowser.open(url)
            return True
        except Exception:
            return False

    # ── OTA update (100% PROTECTED .venv) ─────────────────────────────────────
    def check_updates(self):
        repo = self.config.get("github_repo", DEFAULT_GITHUB_REPO)
        for branch in ("main", "master"):
            try:
                url = f"https://raw.githubusercontent.com/{repo}/{branch}/version.json"
                req = urllib.request.Request(url, headers={"User-Agent": "AI-3D-Studio"})
                with urllib.request.urlopen(req, timeout=6) as r:
                    if r.status == 200:
                        data = json.loads(r.read().decode("utf-8"))
                        remote = data.get("version", "").replace("v", "").strip()
                        local = APP_VERSION.replace("v", "").strip()
                        notes = data.get("releaseNotes", [])
                        if isinstance(notes, list):
                            notes = "\n".join(notes)
                        return {
                            "success": True,
                            "current_version": APP_VERSION,
                            "latest_version": f"v{remote}",
                            "has_update": remote != local and remote,
                            "release_notes": notes,
                            "download_url": data.get("updateUrl", ""),
                        }
            except Exception:
                pass
        return {"success": True, "current_version": APP_VERSION,
                "latest_version": APP_VERSION, "has_update": False,
                "release_notes": "Bạn đang dùng phiên bản mới nhất.", "download_url": ""}

    def apply_update(self, download_url):
        if not download_url:
            return {"success": False, "error": "URL không hợp lệ!"}
        try:
            tmp_zip = os.path.join(APP_DIR, "_update.zip")
            ex_dir = os.path.join(APP_DIR, "_update_ex")
            req = urllib.request.Request(download_url, headers={"User-Agent": "AI-3D-Studio"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp_zip, "wb") as f:
                shutil.copyfileobj(r, f)
            if os.path.exists(ex_dir):
                shutil.rmtree(ex_dir)
            with zipfile.ZipFile(tmp_zip) as z:
                z.extractall(ex_dir)
            items = os.listdir(ex_dir)
            src = os.path.join(ex_dir, items[0]) if (len(items) == 1 and os.path.isdir(os.path.join(ex_dir, items[0]))) else ex_dir

            # Safe copy: STRICTLY PRESERVE .venv
            SKIP_ITEMS = {".venv", "__pycache__", "output_app", "config.json", "_update.zip", "_update_ex"}

            def _safe_copy_tree(s_dir, d_dir):
                os.makedirs(d_dir, exist_ok=True)
                for item in os.listdir(s_dir):
                    if item in SKIP_ITEMS:
                        continue
                    sp = os.path.join(s_dir, item)
                    dp = os.path.join(d_dir, item)
                    if os.path.isdir(sp):
                        _safe_copy_tree(sp, dp)
                    else:
                        shutil.copy2(sp, dp)

            _safe_copy_tree(src, APP_DIR)

            for p in (tmp_zip, ex_dir):
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                    else:
                        shutil.rmtree(p)
                except Exception:
                    pass
            return {"success": True, "message": "Cập nhật thành công! Đang khởi động lại…"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def restart_app(self):
        def _do():
            time.sleep(1)
            os.system(f'start "" "{sys.executable}" "{os.path.join(APP_DIR, "app.py")}"')
            os._exit(0)
        threading.Thread(target=_do, daemon=True).start()
        return True


# ── HTML / CSS / JS ──────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>AI 3D Studio – RTX 3050 & img2threejs & Meshy AAA</title>
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;user-select:none}
body{background:#0a0c12;color:#e2e8f0;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Header ── */
header{height:52px;background:#131722;border-bottom:1px solid #232936;display:flex;align-items:center;justify-content:space-between;padding:0 18px;flex-shrink:0;z-index:20}
.logo{display:flex;align-items:center;gap:8px;font-weight:700;font-size:16px;color:#60a5fa}
.logo-chip{background:linear-gradient(135deg,#2563eb,#7c3aed);padding:3px 8px;border-radius:5px;color:#fff;font-size:10px;font-weight:800}
.ver{font-size:11px;color:#64748b;background:#1a2030;padding:2px 7px;border-radius:4px;border:1px solid #2d3748}
.hdr-right{display:flex;align-items:center;gap:8px}
.gpu-pill{font-size:11px;padding:5px 12px;border-radius:20px;display:flex;align-items:center;gap:6px;font-weight:600;background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.35)}
.dot{width:7px;height:7px;border-radius:50%;background:#10b981;box-shadow:0 0 6px #10b981;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.btn-hdr{background:#1a2030;border:1px solid #2d3748;color:#94a3b8;padding:5px 11px;border-radius:7px;font-size:11px;font-weight:600;cursor:pointer;transition:.15s}
.btn-hdr:hover{background:#232936;color:#e2e8f0;border-color:#3b82f6}

/* ── Layout ── */
.layout{flex:1;display:flex;min-height:0}

/* ── Sidebar ── */
.sidebar{width:380px;background:#0e1017;border-right:1px solid #1e2433;padding:14px;display:flex;flex-direction:column;gap:10px;overflow-y:auto;flex-shrink:0}

/* ── Creation Source Switcher (Image vs Text) ── */
.src-switcher{display:flex;background:#131825;padding:3px;border-radius:9px;border:1px solid #242d40;gap:4px}
.src-tab{flex:1;padding:8px;border:none;border-radius:7px;background:transparent;color:#94a3b8;font-size:11.5px;font-weight:700;cursor:pointer;transition:.18s;display:flex;align-items:center;justify-content:center;gap:5px}
.src-tab.active{background:linear-gradient(135deg,#2563eb,#4f46e5);color:#fff;box-shadow:0 2px 8px rgba(37,99,235,.4)}

/* ── 5 Engine tabs ── */
.tabs{display:grid;grid-template-columns:repeat(auto-fit, minmax(105px, 1fr));background:#07090e;padding:3px;border-radius:8px;border:1px solid #1e2433;gap:3px}
.tab{padding:6px 4px;border:none;border-radius:6px;background:transparent;color:#64748b;font-size:10px;font-weight:700;cursor:pointer;transition:.15s;text-align:center;white-space:nowrap}
.tab.on{background:linear-gradient(135deg,#1d4ed8,#6d28d9);color:#fff;box-shadow:0 2px 8px rgba(37,99,235,.35)}
.tab.img2three.on{background:linear-gradient(135deg,#0284c7,#10b981);color:#fff;box-shadow:0 2px 8px rgba(2,132,199,.4)}
.tab.meshy.on{background:linear-gradient(135deg,#7c3aed,#db2777);color:#fff;box-shadow:0 2px 8px rgba(219,39,119,.4)}
.tab.tripo.on{background:linear-gradient(135deg,#e11d48,#7c3aed);box-shadow:0 2px 10px rgba(225,29,72,.4)}
.tab.cloud.on{background:linear-gradient(135deg,#059669,#2563eb);box-shadow:0 2px 8px rgba(16,185,129,.35)}

/* ── Drop zone ── */
.drop{border:2px dashed #2d3748;border-radius:10px;padding:12px;text-align:center;cursor:pointer;background:#111622;min-height:130px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;transition:.2s}
.drop:hover{border-color:#3b82f6;background:#161f30}
.drop img{max-width:100%;max-height:120px;object-fit:contain;border-radius:7px;display:none}
.drop-hint b{color:#60a5fa;display:block;font-size:13px;margin-bottom:2px}
.drop-hint p{color:#64748b;font-size:11px}

/* ── Text Input Box ── */
.text-box{display:flex;flex-direction:column;gap:7px}
.txt-prompt{width:100%;background:#111622;border:1px solid #2d3748;color:#f1f5f9;padding:9px;border-radius:8px;font-size:12px;resize:none;height:65px;outline:none;line-height:1.4}
.txt-prompt:focus{border-color:#3b82f6}
.tags-wrap{display:flex;flex-wrap:wrap;gap:4px}
.tag-btn{background:#151c2c;border:1px solid #27344d;color:#93c5fd;font-size:10px;padding:4px 7px;border-radius:6px;cursor:pointer;transition:.15s}
.tag-btn:hover{background:#1e293b;border-color:#38bdf8;color:#fff}

/* Concept preview box */
.concept-preview{display:none;background:#131825;border:1px solid #25334d;border-radius:8px;padding:8px;align-items:center;gap:8px}
.concept-preview img{width:60px;height:60px;object-fit:contain;border-radius:6px;background:#0b0d14}
.concept-info{font-size:11px;color:#cbd5e1;line-height:1.4}
.concept-info b{color:#38bdf8}

/* ── Info card ── */
.info-card{background:#111622;border:1px solid #1e2433;border-radius:8px;padding:8px 11px;font-size:11px;display:flex;flex-direction:column;gap:2px}
.ic-title{color:#f1f5f9;font-weight:600;font-size:11.5px}
.ic-desc{color:#38bdf8;line-height:1.4}
.ic-desc b{color:#34d399}

/* ── Settings group ── */
.sg{display:flex;flex-direction:column;gap:3px}
.sg label{font-size:11px;font-weight:600;color:#94a3b8}
select,input[type=text]{background:#111622;border:1px solid #2d3748;color:#e2e8f0;padding:6px 9px;border-radius:7px;font-size:11.5px;outline:none;width:100%}
.chk-row{display:flex;align-items:center;gap:7px;font-size:11px;color:#cbd5e1;cursor:pointer}
.chk-row input{cursor:pointer}

/* ── Generate button ── */
.btn-gen{background:linear-gradient(135deg,#1d4ed8,#6d28d9);color:#fff;border:none;padding:12px;border-radius:8px;font-size:12.5px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:7px;box-shadow:0 4px 14px rgba(37,99,235,.3);transition:.2s}
.btn-gen.img2three-mode{background:linear-gradient(135deg,#0284c7,#10b981);box-shadow:0 4px 14px rgba(2,132,199,.35)}
.btn-gen.meshy-mode{background:linear-gradient(135deg,#7c3aed,#db2777);box-shadow:0 4px 14px rgba(219,39,119,.35)}
.btn-gen.tripo-mode{background:linear-gradient(135deg,#e11d48,#7c3aed);box-shadow:0 4px 14px rgba(225,29,72,.35)}
.btn-gen.cloud-mode{background:linear-gradient(135deg,#059669,#2563eb);box-shadow:0 4px 14px rgba(16,185,129,.3)}
.btn-gen:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 6px 18px rgba(37,99,235,.4)}
.btn-gen:disabled{background:#1a2030;color:#475569;cursor:not-allowed;box-shadow:none;transform:none}

/* ── Progress block ── */
.prog-block{background:#111622;border:1px solid #1e2433;border-radius:8px;padding:9px 11px;display:flex;flex-direction:column;gap:5px;min-height:52px}
.prog-text{font-size:11.5px;color:#94a3b8;display:flex;align-items:center;gap:7px;min-height:18px}
.prog-bar-wrap{height:4px;background:#1e2433;border-radius:2px;overflow:hidden;display:none}
.prog-bar{height:100%;width:0%;background:linear-gradient(90deg,#3b82f6,#10b981);border-radius:2px;transition:width .3s ease}
.spin{width:12px;height:12px;border:2px solid #2d3748;border-top-color:#38bdf8;border-radius:50%;animation:sp .7s linear infinite;display:none;flex-shrink:0}
@keyframes sp{to{transform:rotate(360deg)}}

/* ── Main Stage (Toolbar + 3D Viewport) ── */
.main-stage{flex:1;display:flex;flex-direction:column;min-width:0;height:100%;position:relative}

/* ── Dedicated Stage Toolbar: 100% accessible ── */
.stage-toolbar{height:48px;background:#0d111a;border-bottom:1px solid #1e2536;display:flex;align-items:center;justify-content:space-between;padding:0 14px;flex-shrink:0;z-index:10}
.tool-group{display:flex;align-items:center;gap:6px}
.tool-label{font-size:11px;font-weight:600;color:#94a3b8;display:flex;align-items:center;gap:4px;margin-right:2px}

/* Lighting Preset Buttons */
.light-btn{background:#131825;border:1px solid #253147;color:#94a3b8;padding:5px 9px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;transition:.15s;display:flex;align-items:center;gap:4px}
.light-btn:hover{background:#1c263c;color:#f1f5f9;border-color:#38bdf8}
.light-btn.active{background:linear-gradient(135deg,#1e3a8a,#2563eb);color:#fff;border-color:#60a5fa;box-shadow:0 0 10px rgba(59,130,246,.3)}

/* Action Buttons */
.btn-act{background:#151c2a;border:1px solid #2d3748;color:#94a3b8;padding:6px 11px;border-radius:7px;font-size:11px;font-weight:600;cursor:pointer;transition:all .18s;display:flex;align-items:center;gap:5px}
.btn-act:hover{background:#1e2a3e;border-color:#38bdf8;color:#f1f5f9;transform:translateY(-1px)}
.btn-act.ready{background:linear-gradient(135deg,#132847,#1a263c);border-color:#38bdf8;color:#38bdf8;box-shadow:0 0 10px rgba(56,189,248,.25)}
.btn-act.ready:hover{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;border-color:#60a5fa}

/* ── Viewport ── */
.vp{flex:1;background:radial-gradient(circle at 50% 50%,#151f33 0%,#070910 100%);position:relative;display:flex;align-items:center;justify-content:center;min-width:0;overflow:hidden}
model-viewer{width:100%;height:100%;--poster-color:transparent;position:relative;z-index:1}
.vp-empty{position:absolute;display:flex;flex-direction:column;align-items:center;gap:10px;color:#334155;pointer-events:none;z-index:2}
.vp-empty svg{width:64px;height:64px;opacity:.4}

.hint{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);background:rgba(11,13,19,.85);backdrop-filter:blur(8px);padding:6px 18px;border-radius:16px;font-size:11px;color:#64748b;pointer-events:none;border:1px solid rgba(255,255,255,.07);z-index:5}

/* ── Modal ── */
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.75);backdrop-filter:blur(5px);display:none;align-items:center;justify-content:center;z-index:99}
.modal{background:#111622;border:1px solid #2d3748;border-radius:12px;width:540px;max-width:92vw;padding:22px;display:flex;flex-direction:column;gap:13px;box-shadow:0 20px 40px rgba(0,0,0,.5)}
.m-hdr{display:flex;justify-content:space-between;align-items:center}
.m-title{font-size:15px;font-weight:700;color:#60a5fa}
.m-x{background:none;border:none;color:#64748b;font-size:18px;cursor:pointer}
.notes{background:#080a10;border:1px solid #1e2433;border-radius:7px;padding:10px;font-size:11px;color:#94a3b8;max-height:160px;overflow-y:auto;white-space:pre-wrap;line-height:1.5;display:none}
.m-foot{display:flex;justify-content:flex-end;gap:8px}
.btn-m{padding:7px 14px;border-radius:7px;font-size:12px;font-weight:600;cursor:pointer;border:none}
.btn-m-pri{background:#2563eb;color:#fff}
.btn-m-sec{background:#1e2433;color:#94a3b8}

/* ── Banner ── */
.banner{background:rgba(37,99,235,.12);border:1px solid rgba(37,99,235,.3);border-radius:8px;padding:8px 12px;font-size:11px;color:#93c5fd;display:none;align-items:center;gap:7px}

/* ── Library Header Button & Badges ── */
.btn-lib-hdr{background:linear-gradient(135deg,#1e3a8a,#2563eb);color:#fff;border:1px solid #3b82f6;display:flex;align-items:center;gap:6px;padding:5px 12px;border-radius:7px;font-size:11px;font-weight:700;cursor:pointer;transition:.18s;box-shadow:0 2px 8px rgba(37,99,235,.25)}
.btn-lib-hdr:hover{background:linear-gradient(135deg,#2563eb,#3b82f6);transform:translateY(-1px);box-shadow:0 4px 12px rgba(37,99,235,.4)}
.badge-pill{background:#38bdf8;color:#0f172a;font-size:10px;font-weight:800;padding:1px 6px;border-radius:10px;line-height:1.3}

/* ── 3D Library Modal (Full screen sleek grid) ── */
.modal-lib{width:960px;max-width:95vw;height:84vh;max-height:740px;display:flex;flex-direction:column;gap:12px;padding:20px;overflow:hidden}
.lib-header{display:flex;justify-content:space-between;align-items:center;padding-bottom:10px;border-bottom:1px solid #1e2638;flex-shrink:0}
.lib-title-row{display:flex;align-items:center;gap:10px}
.lib-title{font-size:16px;font-weight:800;color:#60a5fa;display:flex;align-items:center;gap:8px}
.lib-stats{font-size:12px;color:#94a3b8;background:#151c2a;border:1px solid #28354b;padding:3px 10px;border-radius:6px}

.lib-controls{display:flex;flex-direction:column;gap:10px;flex-shrink:0}
.lib-search-row{display:flex;gap:10px;align-items:center}
.lib-search-input{flex:1;background:#0d111a;border:1px solid #28354b;color:#f1f5f9;padding:8px 12px;border-radius:8px;font-size:12px;outline:none;transition:.15s}
.lib-search-input:focus{border-color:#38bdf8;box-shadow:0 0 8px rgba(56,189,248,.25)}
.btn-lib-folder{background:#151c2a;border:1px solid #28354b;color:#94a3b8;padding:8px 12px;border-radius:8px;font-size:11.5px;font-weight:600;cursor:pointer;white-space:nowrap;transition:.15s}
.btn-lib-folder:hover{background:#1e293b;border-color:#38bdf8;color:#f1f5f9}

.lib-chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.lib-chip{background:#0e1320;border:1px solid #1e2638;color:#94a3b8;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;cursor:pointer;transition:.15s;display:flex;align-items:center;gap:5px}
.lib-chip:hover{background:#182236;color:#e2e8f0;border-color:#38bdf8}
.lib-chip.active{background:linear-gradient(135deg,#2563eb,#3b82f6);color:#fff;border-color:#60a5fa;box-shadow:0 2px 8px rgba(37,99,235,.3)}
.chip-num{background:rgba(0,0,0,.3);padding:1px 6px;border-radius:10px;font-size:9.5px;font-weight:700}

/* Grid & Cards */
.lib-grid-wrap{flex:1;overflow-y:auto;min-height:0;padding-right:4px}
.lib-grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(260px, 1fr));gap:14px;padding:4px 0}
.lib-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:240px;color:#64748b;gap:10px;text-align:center}

.lib-card{background:#0f1422;border:1px solid #1e273a;border-radius:10px;overflow:hidden;display:flex;flex-direction:column;transition:all .18s;position:relative}
.lib-card:hover{border-color:#3b82f6;transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.4)}

.lib-card-thumb{height:140px;background:#06080e;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}
.lib-card-thumb img{width:100%;height:100%;object-fit:cover;transition:transform .2s}
.lib-card:hover .lib-card-thumb img{transform:scale(1.05)}
.lib-thumb-fallback{color:#334155;display:flex;flex-direction:column;align-items:center;gap:4px;font-size:11px}

.lib-badge-engine{position:absolute;top:8px;left:8px;font-size:10px;font-weight:700;padding:3px 7px;border-radius:5px;backdrop-filter:blur(6px);box-shadow:0 2px 6px rgba(0,0,0,.4);white-space:nowrap;max-width:180px;overflow:hidden;text-overflow:ellipsis}
.badge-local{background:rgba(16,185,129,.85);color:#fff}
.badge-img2three{background:rgba(2,132,199,.85);color:#fff}
.badge-meshy{background:rgba(192,38,211,.85);color:#fff}
.badge-tripo{background:rgba(225,29,72,.85);color:#fff}
.badge-cloud{background:rgba(14,165,233,.85);color:#fff}

.lib-badge-size{position:absolute;bottom:8px;right:8px;background:rgba(10,12,18,.8);border:1px solid rgba(255,255,255,.1);font-size:10px;font-weight:600;color:#cbd5e1;padding:2px 6px;border-radius:4px;backdrop-filter:blur(4px)}

.lib-card-body{padding:10px 12px;display:flex;flex-direction:column;gap:6px;flex:1}
.lib-card-name{font-size:12.5px;font-weight:700;color:#f1f5f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lib-card-sub{font-size:10.5px;color:#64748b;display:flex;justify-content:space-between;align-items:center}

.lib-card-actions{display:flex;gap:4px;margin-top:auto;padding-top:8px;border-top:1px solid #1a2233}
.lib-btn-act{flex:1;background:#151c2a;border:1px solid #25334c;color:#cbd5e1;padding:5px 6px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;text-align:center;transition:.15s;white-space:nowrap}
.lib-btn-act:hover{background:#1e2a3f;border-color:#38bdf8;color:#fff}
.lib-btn-pri{background:linear-gradient(135deg,#1d4ed8,#2563eb);border-color:#3b82f6;color:#fff}
.lib-btn-pri:hover{background:linear-gradient(135deg,#2563eb,#3b82f6)}
.lib-btn-del{flex:0 0 28px;background:#1c1418;border-color:#4a1e28;color:#f87171}
.lib-btn-del:hover{background:#dc2626;color:#fff;border-color:#ef4444}
</style>
</head>
<body>

<header>
  <div style="display:flex;align-items:center;gap:9px">
    <div class="logo"><span class="logo-chip">3D AI</span>AI 3D Studio</div>
    <span class="ver" id="ver">v1.9.0</span>
  </div>
  <div class="hdr-right">
    <div class="gpu-pill"><div class="dot"></div><span id="gpuTxt">Đang nạp card GPU…</span></div>
    <button class="btn-lib-hdr" onclick="openLibrary()" title="Mở Thư viện quản lý các mô hình 3D đã tạo">
      🏛️ Thư viện 3D <span class="badge-pill" id="libBadgeHdr">0</span>
    </button>
    <button class="btn-hdr" onclick="openSettings()">⚙️ Cài đặt &amp; API Key</button>
    <button class="btn-hdr" onclick="openUpdate()">🔄 Cập nhật</button>
  </div>
</header>

<div class="layout">
  <div class="sidebar">
    <!-- Source Switcher: Image or Text -->
    <div class="src-switcher">
      <button class="src-tab active" id="srcTabImg" onclick="switchSource('image')">🖼️ Từ Hình Ảnh</button>
      <button class="src-tab" id="srcTabTxt" onclick="switchSource('text')">✍️ Từ Văn Bản</button>
    </div>

    <!-- 5 Engine tabs -->
    <div class="tabs">
      <button class="tab on" id="tabLocal" onclick="setMode('local')">⚡ RTX 3050</button>
      <button class="tab img2three" id="tabImg2Three" onclick="setMode('img2threejs')">🎨 img2threejs</button>
      <button class="tab meshy" id="tabMeshy" onclick="setMode('meshy')">💎 Meshy AAA</button>
      <button class="tab tripo" id="tabTripo" onclick="setMode('tripo')">🚀 Tripo3D</button>
      <button class="tab cloud" id="tabCloud" onclick="setMode('cloud')">🌐 Hybrid</button>
    </div>

    <!-- Not-ready banner -->
    <div class="banner" id="banner">
      <span>⏳</span>
      <span id="bannerTxt">Model AI đang nạp vào GPU RTX 3050 (~vài giây). Sẵn sàng!</span>
    </div>

    <!-- 1. IMAGE MODE CONTAINER -->
    <div id="imageBox">
      <div class="drop" id="drop" onclick="pickImage()">
        <img id="prev" alt="preview">
        <div class="drop-hint" id="dropHint">
          <b>Chọn ảnh 2D từ máy tính</b>
          <p>Nhấn để nạp ảnh PNG / JPG / WebP</p>
        </div>
      </div>
    </div>

    <!-- 2. TEXT MODE CONTAINER -->
    <div id="textBox" style="display:none" class="text-box">
      <textarea id="promptInput" class="txt-prompt" placeholder="Nhập mô tả bằng tiếng Việt hoặc tiếng Anh (ví dụ: Quả chuối vàng chín mọng, Ghế sofa bọc da sang trọng, Siêu xe Lamborghini, Thanh kiếm hiệp sĩ…)"></textarea>
      <div class="tags-wrap">
        <button class="tag-btn" onclick="setPrompt('Quả chuối vàng chín mọng')">🍌 Chuối vàng</button>
        <button class="tag-btn" onclick="setPrompt('Chiếc ghế sofa bọc da sang trọng')">🛋️ Ghế sofa</button>
        <button class="tag-btn" onclick="setPrompt('Chiếc xe thể thao Ferrari màu đỏ')">🏎️ Siêu xe</button>
        <button class="tag-btn" onclick="setPrompt('Thanh kiếm hiệp sĩ bằng thép sáng loáng')">⚔️ Kiếm hiệp sĩ</button>
        <button class="tag-btn" onclick="setPrompt('Bình hoa gốm sứ cổ điển xanh trắng')">🏺 Bình gốm</button>
      </div>
      <!-- Concept preview when generated -->
      <div class="concept-preview" id="conceptBox">
        <img id="conceptImg" alt="concept">
        <div class="concept-info">
          <b>Ý tưởng 2D AI đã phác họa</b><br>
          <span id="conceptTxt" style="font-size:10px;color:#94a3b8">Đang chuyển vào nhân GPU 3D...</span>
        </div>
      </div>
    </div>

    <!-- Info card -->
    <div class="info-card">
      <span class="ic-title" id="icTitle">⚡ GPU Cục bộ – NVIDIA RTX 3050 (100% Offline)</span>
      <span class="ic-desc" id="icDesc">Chạy 100% trên card máy, <b>không cần mạng, không hết lượt</b>, nướng vân UV 1024px mịn màng.</span>
    </div>

    <!-- Local settings -->
    <div id="localSet">
      <div class="sg" style="margin-bottom:7px">
        <label>Chất lượng hình học &amp; Bề mặt RTX 3050:</label>
        <select id="quality">
          <option value="pbr_1024" selected>💎 Đỉnh cao PBR 1024px + Làm mịn Taubin (~20s) – Mịn màng, vân nét (Khuyên dùng)</option>
          <option value="ultra_320">📐 Chi tiết góc cạnh Ultra HD 320 (~10s) – 43.000 điểm lưới</option>
          <option value="fast_256">⚡ Siêu tốc Vertex Colors 256 (~3s) – Màu sắc rực rỡ</option>
        </select>
      </div>
      <div class="chk-row" style="margin-bottom:6px">
        <input type="checkbox" id="chkSmooth" checked>
        <label class="chk-row" for="chkSmooth">✨ Làm mịn Taubin (giữ nguyên khối lượng, khử phẳng sọc)</label>
      </div>
    </div>

    <!-- img2threejs settings & showcase gallery -->
    <div id="img2threeSet" style="display:none">
      <div style="background:rgba(2,132,199,.15);border:1px solid rgba(2,132,199,.35);border-radius:8px;padding:8px 10px;font-size:11px;color:#38bdf8;line-height:1.4;margin-bottom:6px">
        🎨 <b>img2threejs (Reconstruction-by-Code + TRELLIS AI):</b><br>
        Tái tạo mô hình 3D chuẩn xác cao (PBR GLB + OBJ) tối ưu hóa cho Three.js, Blender và Unity.
      </div>
      <div style="background:#131824;border:1px solid #1e2536;border-radius:8px;padding:8px 10px;font-size:10.5px;color:#cbd5e1;line-height:1.5">
        🌟 <b>MÔ HÌNH NỔI TIẾNG (IMG2THREEJS SHOWCASE):</b><br>
        <span style="color:#94a3b8">Bấm để mở xem trực tiếp mô hình mẫu 3D trên trình duyệt:</span>
        <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px">
          <button class="tag-btn" onclick="openDemo('talon-doppler-ruby')">🔪 Dao Talon Ruby</button>
          <button class="tag-btn" onclick="openDemo('awp-medusa-v2')">🔫 Súng AWP Medusa</button>
          <button class="tag-btn" onclick="openDemo('electric-mouse-mascot')">⚡ Pikachu Mascot</button>
          <button class="tag-btn" onclick="openDemo('doraemon-house')">🏠 Nhà Doraemon</button>
          <button class="tag-btn" onclick="openDemo('sony-wf1000xm3')">🎧 Tai nghe Sony WF</button>
          <button class="tag-btn" onclick="openDemo('crown-chest')">👑 Rương Hoàng Gia</button>
          <button class="tag-btn" onclick="openDemo('glock-ghost-protocol')">🔫 Súng Glock Ghost</button>
          <button class="tag-btn" onclick="openDemo('bmx-endurance')">🚲 Xe đạp BMX</button>
        </div>
        <div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center">
          <a href="javascript:void(0)" onclick="openDemoGallery()" style="color:#38bdf8;font-weight:700;text-decoration:underline">🌐 Mở img2threejs.io Gallery</a>
          <button class="tag-btn" onclick="openImg2ThreeCode()" style="color:#34d399">📁 Mã nguồn cục bộ</button>
        </div>
      </div>
    </div>

    <!-- Meshy settings -->
    <div id="meshySet" style="display:none">
      <div style="background:rgba(124,58,237,.15);border:1px solid rgba(124,58,237,.35);border-radius:8px;padding:8px 10px;font-size:11px;color:#f472b6;line-height:1.4;margin-bottom:6px">
        💎 <b>Chất lượng Game AAA Siêu Thực (Meshy AI):</b><br>
        Tạo mô hình PBR đa lớp (Albedo, Normal, Roughness, Metallic) đẹp xuất sắc.
      </div>
      <div style="background:#131824;border:1px solid #1e2536;border-radius:8px;padding:8px 10px;font-size:10.5px;color:#cbd5e1;line-height:1.5">
        🎁 <b>TẶNG 200 CREDITS MIỄN PHÍ MỖI THÁNG:</b><br>
        1. Nhấn <a href="javascript:void(0)" onclick="openMeshyWeb()" style="color:#60a5fa;font-weight:700;text-decoration:underline">Mở trang meshy.ai</a> đăng ký miễn phí.<br>
        2. Vào <b>Settings -> API Keys</b> tạo key (msy_...).<br>
        3. Vào <b>⚙️ Cài đặt</b> dán key để tạo mô hình AAA trực tiếp!
      </div>
    </div>

    <!-- Tripo3D settings -->
    <div id="tripoSet" style="display:none">
      <div style="background:#131824;border:1px solid #1e2536;border-radius:8px;padding:8px 10px;font-size:10.5px;color:#cbd5e1;line-height:1.5">
        💡 <b>Tripo3D Web (300 Credits Miễn Phí):</b><br>
        1. Nhấn <a href="javascript:void(0)" onclick="openTripoWeb()" style="color:#60a5fa;font-weight:700;text-decoration:underline">Mở platform.tripo3d.ai</a> tạo 3D.<br>
        2. Tải file <b>.glb</b> về máy tính.<br>
        3. Nhấn <b>'📂 Nạp 3D ngoài'</b> trên thanh công cụ để mở và xuất sang Blender/Unity!
      </div>
    </div>

    <!-- Cloud settings -->
    <div id="cloudSet" style="display:none">
      <div class="sg" style="margin-bottom:7px">
        <label>Chất lượng Cloud Multi-View</label>
        <select id="cloudQ">
          <option value="pbr_1024" selected>🌟 Tái tạo cao cấp (50 bước – Chi tiết đa góc)</option>
          <option value="fast_256">⚡ Tiết kiệm Quota (30 bước)</option>
        </select>
      </div>
    </div>

    <button class="btn-gen" id="btnGen" onclick="generate()">
      ⚡ BẮT ĐẦU TẠO 3D (RTX 3050 OFFLINE)
    </button>

    <!-- Progress block -->
    <div class="prog-block">
      <div class="prog-text">
        <div class="spin" id="spin"></div>
        <span id="progTxt">Chọn ảnh hoặc nhập văn bản để tạo mô hình 3D.</span>
      </div>
      <div class="prog-bar-wrap" id="barWrap">
        <div class="prog-bar" id="bar"></div>
      </div>
    </div>
  </div>

  <!-- Main Stage: Toolbar + 3D Viewport -->
  <div class="main-stage">
    <!-- Top Action Toolbar: 100% ACCESSIBLE & CLICKABLE -->
    <div class="stage-toolbar">
      <!-- Lighting Presets (Buttons) -->
      <div class="tool-group">
        <span class="tool-label">💡 Ánh sáng:</span>
        <button class="light-btn active" id="lbtnStudio" onclick="setLighting('studio')">✨ Studio ACES</button>
        <button class="light-btn" id="lbtnCinema" onclick="setLighting('aces')">🎬 Cinema</button>
        <button class="light-btn" id="lbtnSoft" onclick="setLighting('soft')">☀️ Tự nhiên</button>
      </div>

      <!-- Action Buttons -->
      <div class="tool-group">
        <button class="btn-act ready" onclick="openLibrary()" style="background:linear-gradient(135deg,#1e3a8a,#2563eb);color:#fff;border-color:#60a5fa" title="Xem và quản lý toàn bộ mô hình 3D trong thư viện">
          🏛️ Thư viện (<span id="libBadgeToolbar">0</span>)
        </button>
        <button class="btn-act ready" id="btnLoadLast" onclick="loadLastModel()" style="display:none;background:linear-gradient(135deg,#0284c7,#2563eb);color:#fff" title="Xem lại mô hình 3D vừa tạo trong phiên trước">
          ↺ Xem mô hình trước
        </button>
        <button class="btn-act ready" id="btnImport" onclick="importModel()" title="Nạp file 3D GLB/OBJ từ máy tính hoặc tải về từ web">
          📂 Nạp 3D ngoài
        </button>
        <button class="btn-act ready" id="btnGlb" onclick="doExport('glb')" title="Lưu định dạng GLB chuẩn Unity và Game Engine">
          📦 Lưu GLB
        </button>
        <button class="btn-act ready" id="btnObj" onclick="doExport('obj')" title="Lưu định dạng OBJ kèm vật liệu cho Blender và Maya">
          📦 Lưu OBJ
        </button>
        <button class="btn-act ready" id="btnDir" onclick="openDir()" title="Mở thư mục chứa file đã tạo trên máy tính">
          📁 Mở thư mục
        </button>
      </div>
    </div>

    <!-- 3D Canvas Area -->
    <div class="vp">
      <div class="vp-empty" id="empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
          <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
          <line x1="12" y1="22.08" x2="12" y2="12"/>
        </svg>
        <p style="font-size:14px;font-weight:600;color:#64748b">Mô hình 3D xoay 360° sẽ hiển thị tại đây</p>
        <p style="font-size:12px;color:#475569">Chọn ảnh hoặc nhập văn bản rồi nhấn Bắt đầu tạo 3D</p>
      </div>

      <model-viewer id="mv"
        camera-controls
        touch-action="pan-y"
        auto-rotate
        auto-rotate-delay="1000"
        rotation-per-second="22deg"
        tone-mapping="aces"
        shadow-intensity="1.5"
        shadow-softness="0.5"
        exposure="1.15"
        bounds="tight"
        camera-orbit="0deg 75deg auto"
        field-of-view="35deg"
        style="display:none">
      </model-viewer>

      <div class="hint" id="hint" style="display:none">
        🖱️ Chuột trái: Xoay 360° &nbsp;|&nbsp; Con lăn: Zoom &nbsp;|&nbsp; Chuột phải: Di chuyển góc nhìn &nbsp;|&nbsp; Bấm Studio ACES đổi ánh sáng
      </div>
    </div>
  </div>
</div>

<!-- Settings modal -->
<div class="modal-bg" id="mSet">
  <div class="modal">
    <div class="m-hdr">
      <span class="m-title">⚙️ Cài đặt API Key (Miễn phí 100%)</span>
      <button class="m-x" onclick="closeSettings()">&times;</button>
    </div>

    <!-- Meshy API Key (Top recommendation) -->
    <div class="sg">
      <label>💎 Meshy AI API Key (Tặng 200 credits miễn phí/tháng – Chất lượng Game AAA):</label>
      <input type="text" id="meshyKey" placeholder="msy_xxxxxxxxxxxxxxxxxxxxxxxx" style="font-family:monospace">
      <p style="font-size:10.5px;color:#94a3b8;margin-top:2px;line-height:1.4">
        👉 Đăng ký miễn phí tại <b style="color:#60a5fa">meshy.ai</b> -> Settings -> API Keys để nhận key.
      </p>
    </div>

    <!-- Tripo3D API Key -->
    <div class="sg" style="margin-top:4px">
      <label>🚀 Tripo3D API Key (Tùy chọn):</label>
      <input type="text" id="tripoKey" placeholder="tsk_xxxxxxxxxxxxxxxxxxxxxxxx" style="font-family:monospace">
    </div>

    <!-- Hugging Face Token -->
    <div class="sg" style="margin-top:4px">
      <label>☁️ Hugging Face Token (Tùy chọn):</label>
      <input type="text" id="hfTok" placeholder="hf_xxxxxxxxxxxxxxxx" style="font-family:monospace">
    </div>

    <div class="m-foot" style="margin-top:8px">
      <button class="btn-m btn-m-sec" onclick="closeSettings()">Hủy</button>
      <button class="btn-m btn-m-pri" onclick="saveSettings()">💾 Lưu cài đặt</button>
    </div>
  </div>
</div>

<!-- Update modal -->
<div class="modal-bg" id="mUpd">
  <div class="modal">
    <div class="m-hdr">
      <span class="m-title">🔄 Cập nhật từ GitHub</span>
      <button class="m-x" onclick="closeUpdate()">&times;</button>
    </div>
    <div id="updTxt" style="font-size:12px;color:#64748b">Nhấn "Kiểm tra" để đồng bộ với GitHub.</div>
    <div class="notes" id="updNotes"></div>
    <div class="m-foot">
      <button class="btn-m btn-m-sec" onclick="closeUpdate()">Đóng</button>
      <button class="btn-m btn-m-sec" onclick="checkUpd()">Kiểm tra bản mới</button>
      <button class="btn-m btn-m-pri" id="btnApply" onclick="applyUpd()" style="display:none">🚀 Cập nhật ngay</button>
    </div>
  </div>
</div>

<!-- 3D Model Library modal -->
<div class="modal-bg" id="mLib">
  <div class="modal modal-lib">
    <div class="lib-header">
      <div class="lib-title-row">
        <div class="lib-title">
          <span>🏛️</span> Thư viện Mô hình 3D
        </div>
        <div class="lib-stats" id="libStats">
          Đang quét thư viện…
        </div>
      </div>
      <button class="m-x" onclick="closeLibrary()">&times;</button>
    </div>

    <div class="lib-controls">
      <div class="lib-search-row">
        <input type="text" class="lib-search-input" id="libSearchInput" placeholder="🔍 Tìm kiếm theo tên mô hình, câu lệnh prompt hoặc nguồn tạo..." oninput="filterLib()">
        <button class="btn-lib-folder" onclick="openLibraryRoot()" title="Mở thư mục chứa toàn bộ mô hình 3D trên ổ đĩa">
          📁 Mở thư mục gốc
        </button>
      </div>
      <div class="lib-chips">
        <button class="lib-chip active" id="chipAll" onclick="setLibFilter('all')">
          Tất cả <span class="chip-num" id="cntAll">0</span>
        </button>
        <button class="lib-chip" id="chipLocal" onclick="setLibFilter('local')">
          ⚡ RTX 3050 <span class="chip-num" id="cntLocal">0</span>
        </button>
        <button class="lib-chip" id="chipImg2Three" onclick="setLibFilter('img2threejs')">
          🎨 img2threejs <span class="chip-num" id="cntImg2Three">0</span>
        </button>
        <button class="lib-chip" id="chipMeshy" onclick="setLibFilter('meshy')">
          💎 Meshy AAA <span class="chip-num" id="cntMeshy">0</span>
        </button>
        <button class="lib-chip" id="chipTripo" onclick="setLibFilter('tripo')">
          🚀 Tripo3D <span class="chip-num" id="cntTripo">0</span>
        </button>
        <button class="lib-chip" id="chipCloud" onclick="setLibFilter('cloud')">
          🌐 Hybrid <span class="chip-num" id="cntCloud">0</span>
        </button>
      </div>
    </div>

    <div class="lib-grid-wrap">
      <div class="lib-grid" id="libGrid"></div>
    </div>
  </div>
</div>

<script>
/* ── state ── */
let imgPath = null, lastFolder = null, dlUrl = null, curMode = 'local', curSource = 'image';
let _modelReady = false;

/* ── called from Python to push progress ── */
window._setProgress = function(msg, pct) {
  document.getElementById('progTxt').textContent = msg;
  const barWrap = document.getElementById('barWrap');
  const bar = document.getElementById('bar');
  if (pct >= 0) {
    barWrap.style.display = 'block';
    bar.style.width = pct + '%';
  }
};

window._showConcept = function(dataUrl) {
  document.getElementById('conceptImg').src = dataUrl;
  document.getElementById('conceptBox').style.display = 'flex';
  document.getElementById('conceptTxt').textContent = 'Đang đưa vào nhân GPU 3D…';
};

/* ── source switcher (Image vs Text) ── */
function switchSource(s) {
  curSource = s;
  if (s === 'image') {
    document.getElementById('srcTabImg').classList.add('active');
    document.getElementById('srcTabTxt').classList.remove('active');
    document.getElementById('imageBox').style.display = 'block';
    document.getElementById('textBox').style.display = 'none';
  } else {
    document.getElementById('srcTabTxt').classList.add('active');
    document.getElementById('srcTabImg').classList.remove('active');
    document.getElementById('textBox').style.display = 'flex';
    document.getElementById('imageBox').style.display = 'none';
  }
  updateGenBtnText();
}

function updateGenBtnText() {
  const btn = document.getElementById('btnGen');
  if (curMode === 'local') {
    btn.className = 'btn-gen';
    btn.textContent = curSource === 'image' ? '⚡ BẮT ĐẦU TẠO 3D (RTX 3050 OFFLINE)' : '✍️ BẮT ĐẦU TẠO 3D TỪ VĂN BẢN (RTX 3050)';
  } else if (curMode === 'img2threejs') {
    btn.className = 'btn-gen img2three-mode';
    btn.textContent = curSource === 'image' ? '🎨 TẠO 3D VỚI IMG2THREEJS (TRELLIS)' : '✍️ TẠO 3D TỪ VĂN BẢN (IMG2THREEJS)';
  } else if (curMode === 'meshy') {
    btn.className = 'btn-gen meshy-mode';
    btn.textContent = curSource === 'image' ? '💎 TẠO 3D GAME AAA (MESHY AI)' : '✍️ TẠO 3D TỪ CHỮ GAME AAA (MESHY AI)';
  } else if (curMode === 'tripo') {
    btn.className = 'btn-gen tripo-mode';
    btn.textContent = '🚀 TẠO 3D STUDIO (TRIPO3D)';
  } else {
    btn.className = 'btn-gen cloud-mode';
    btn.textContent = '🌐 TẠO 3D HYBRID (CLOUD)';
  }
}

function setPrompt(txt) {
  document.getElementById('promptInput').value = txt;
}

/* ── init ── */
window.addEventListener('pywebviewready', async () => {
  const d = await window.pywebview.api.get_init_data();
  document.getElementById('ver').textContent = d.version;
  document.getElementById('gpuTxt').textContent = d.hardware.has_gpu
    ? 'GPU: ' + d.hardware.name + ' (' + d.hardware.vram_gb + ' GB)'
    : d.hardware.status_text;

  if (d.config) {
    if (d.config.hf_token) document.getElementById('hfTok').value = d.config.hf_token;
    if (d.config.tripo_api_key) document.getElementById('tripoKey').value = d.config.tripo_api_key;
    if (d.config.meshy_api_key) document.getElementById('meshyKey').value = d.config.meshy_api_key;
  }

  _modelReady = d.model_ready;

  if (d.latest_model && d.latest_model.has_model) {
    lastFolder = d.latest_model.folder;
    const btnLast = document.getElementById('btnLoadLast');
    if (btnLast) btnLast.style.display = 'inline-flex';
    window._setProgress('⚡ Studio sẵn sàng! Chọn ảnh hoặc nhập văn bản để tạo mô hình 3D.', -1);
  } else {
    window._setProgress('⚡ Studio sẵn sàng! Chọn ảnh hoặc nhập văn bản để tạo mô hình 3D.', -1);
  }

  updateLibBadge();
});

async function loadLastModel() {
  window._setProgress('Đang nạp mô hình phiên trước…', 50);
  const r = await window.pywebview.api.load_latest_model();
  if (r && r.success) {
    lastFolder = r.folder;
    const mv = document.getElementById('mv');
    mv.src = r.glb_data;
    mv.style.display = 'block';
    document.getElementById('empty').style.display = 'none';
    document.getElementById('hint').style.display = 'block';
    window._setProgress('✨ Đã nạp mô hình 3D hoàn chỉnh. Sẵn sàng lưu GLB / OBJ!', 100);
  } else {
    window._setProgress('❌ ' + (r.error || 'Không tìm thấy mô hình'), -1);
  }
}

/* ── mode switcher ── */
function setMode(m) {
  curMode = m;
  ['tabLocal','tabImg2Three','tabMeshy','tabTripo','tabCloud'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.className = 'tab' + (id === 'tabImg2Three' ? ' img2three' : (id === 'tabMeshy' ? ' meshy' : (id === 'tabTripo' ? ' tripo' : (id === 'tabCloud' ? ' cloud' : ''))));
  });
  ['localSet','img2threeSet','meshySet','tripoSet','cloudSet'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });

  if (m === 'local') {
    document.getElementById('tabLocal').classList.add('on');
    document.getElementById('localSet').style.display = '';
    document.getElementById('icTitle').textContent = '⚡ GPU Cục bộ – NVIDIA RTX 3050 (100% Offline)';
    document.getElementById('icDesc').innerHTML = 'Tạo 3D bằng card RTX 3050 máy bạn. <b>Không cần mạng, không hết lượt</b>, nướng vân UV 1024px mịn màng.';
  } else if (m === 'img2threejs') {
    document.getElementById('tabImg2Three').classList.add('on');
    document.getElementById('img2threeSet').style.display = '';
    document.getElementById('icTitle').textContent = '🎨 img2threejs – Tái tạo 3D bằng Code & TRELLIS';
    document.getElementById('icDesc').innerHTML = 'Công nghệ tái tạo mô hình 3D cho Three.js, Blender và Unity từ hình ảnh hoặc ý tưởng văn bản.';
  } else if (m === 'meshy') {
    document.getElementById('tabMeshy').classList.add('on');
    document.getElementById('meshySet').style.display = '';
    document.getElementById('icTitle').textContent = '💎 Meshy AI Pro (Chất lượng Game AAA Siêu Thực)';
    document.getElementById('icDesc').innerHTML = 'Được tặng <b>200 credits miễn phí mỗi tháng</b>. Vân PBR phản chiếu ánh sáng chân thực 100%.';
  } else if (m === 'tripo') {
    document.getElementById('tabTripo').classList.add('on');
    document.getElementById('tripoSet').style.display = '';
    document.getElementById('icTitle').textContent = '🚀 Tripo3D Studio (300 Credits Web)';
    document.getElementById('icDesc').innerHTML = 'Tạo trên web <b>platform.tripo3d.ai</b> rồi nạp vào bằng nút <b>📂 Nạp 3D ngoài</b>.';
  } else {
    document.getElementById('tabCloud').classList.add('on');
    document.getElementById('cloudSet').style.display = '';
    document.getElementById('icTitle').textContent = '🌐 Chế độ Hybrid Cloud Multi-View';
    document.getElementById('icDesc').innerHTML = 'Dùng máy chủ Cloud tái tạo 6 góc nhìn đa chiều 360°.';
  }
  updateGenBtnText();
}

/* ── image pick ── */
async function pickImage() {
  window._setProgress('Đang mở hộp thoại chọn ảnh…', -1);
  const r = await window.pywebview.api.select_image();
  if (r && r.path) {
    imgPath = r.path;
    document.getElementById('prev').src = r.dataUrl;
    document.getElementById('prev').style.display = 'block';
    document.getElementById('dropHint').style.display = 'none';
    window._setProgress('Đã chọn: ' + r.name + ' – Nhấn nút Bắt đầu tạo 3D!', -1);
  } else {
    window._setProgress('Chưa chọn ảnh.', -1);
  }
}

/* ── generate (Non-blocking async task with 400ms polling) ── */
async function generate() {
  const quality = curMode === 'local'
    ? document.getElementById('quality').value
    : document.getElementById('cloudQ').value;
  const smooth = document.getElementById('chkSmooth').checked;

  if (curSource === 'image' && !imgPath) {
    window._setProgress('⚠️ Hãy nhấn vào khung chọn ảnh trước!', -1);
    await pickImage();
    if (!imgPath) return;
  }

  if (curSource === 'text') {
    const prompt = document.getElementById('promptInput').value.trim();
    if (!prompt) {
      window._setProgress('⚠️ Vui lòng nhập mô tả hoặc bấm chọn 1 gợi ý bên dưới!', -1);
      document.getElementById('promptInput').focus();
      return;
    }
  }

  const btn = document.getElementById('btnGen');
  const spin = document.getElementById('spin');
  const barWrap = document.getElementById('barWrap');
  const bar = document.getElementById('bar');

  btn.disabled = true;
  spin.style.display = 'inline-block';
  barWrap.style.display = 'block';
  bar.style.width = '2%';
  window._setProgress('Đang khởi động tiến trình xử lý…', 5);

  function finishRun() {
    btn.disabled = false;
    spin.style.display = 'none';
  }

  try {
    const prompt = (curSource === 'text') ? document.getElementById('promptInput').value.trim() : '';
    const startRes = (curSource === 'text')
      ? await window.pywebview.api.start_generate_from_text(prompt, quality, smooth, curMode)
      : await window.pywebview.api.start_generate_3d(imgPath, curMode, 'auto', (quality === 'pbr_1024'), smooth, quality);

    if (!startRes || !startRes.task_id) {
      window._setProgress('❌ Không thể khởi tạo tác vụ: ' + (startRes && startRes.error ? startRes.error : 'Lỗi không xác định'), -1);
      finishRun();
      return;
    }

    const taskId = startRes.task_id;
    const pollInterval = setInterval(async () => {
      try {
        const task = await window.pywebview.api.poll_task(taskId);
        if (!task || task.status === 'not_found') {
          return;
        }
        if (task.status === 'running') {
          if (task.msg) {
            window._setProgress(task.msg, (task.pct !== undefined && task.pct >= 0) ? task.pct : -1);
          }
        } else if (task.status === 'done') {
          clearInterval(pollInterval);
          const r = task.result;
          if (r && r.success) {
            lastFolder = r.folder;
            const mv = document.getElementById('mv');
            mv.src = r.glb_data;
            mv.style.display = 'block';
            document.getElementById('empty').style.display = 'none';
            document.getElementById('hint').style.display = 'block';
            barWrap.style.display = 'block';
            bar.style.width = '100%';
            window._setProgress('✅ Thành công! (' + r.engine_used + ') – Sẵn sàng lưu file!', 100);
            if (r.concept_data) {
              document.getElementById('conceptTxt').textContent = '✅ Đã hoàn tất mô hình 3D!';
            }
            updateLibBadge();
          } else {
            window._setProgress('❌ ' + ((r && r.error) ? r.error : 'Lỗi không xác định'), -1);
            bar.style.width = '0%';
          }
          finishRun();
        } else if (task.status === 'error') {
          clearInterval(pollInterval);
          window._setProgress('❌ ' + (task.error || 'Lỗi không xác định'), -1);
          bar.style.width = '0%';
          finishRun();
        }
      } catch (pollErr) {
        console.warn('Poll error:', pollErr);
      }
    }, 400);

  } catch(e) {
    window._setProgress('❌ Lỗi ngoại lệ: ' + e, -1);
    finishRun();
  }
}

/* ── lighting ── */
function setLighting(mode) {
  const mv = document.getElementById('mv');
  ['lbtnStudio','lbtnCinema','lbtnSoft'].forEach(id => document.getElementById(id).classList.remove('active'));

  if (mode === 'studio') {
    document.getElementById('lbtnStudio').classList.add('active');
    mv.setAttribute('tone-mapping', 'aces');
    mv.setAttribute('exposure', '1.15');
    mv.setAttribute('shadow-intensity', '1.5');
    mv.setAttribute('shadow-softness', '0.5');
    window._setProgress('💡 Chế độ ánh sáng: Studio ACES (Chuẩn thực tế 100%)', -1);
  } else if (mode === 'aces') {
    document.getElementById('lbtnCinema').classList.add('active');
    mv.setAttribute('tone-mapping', 'aces');
    mv.setAttribute('exposure', '1.0');
    mv.setAttribute('shadow-intensity', '2.2');
    mv.setAttribute('shadow-softness', '0.3');
    window._setProgress('💡 Chế độ ánh sáng: Điện ảnh Cinema (Đậm nét & Tương phản cao)', -1);
  } else {
    document.getElementById('lbtnSoft').classList.add('active');
    mv.setAttribute('tone-mapping', 'neutral');
    mv.setAttribute('exposure', '1.2');
    mv.setAttribute('shadow-intensity', '0.8');
    mv.setAttribute('shadow-softness', '0.8');
    window._setProgress('💡 Chế độ ánh sáng: Tự nhiên Studio (Mềm mại dịu mắt)', -1);
  }
}

/* ── export / folder ── */
async function doExport(t) {
  window._setProgress('📦 Đang mở hộp thoại lưu file ' + t.toUpperCase() + '…', -1);
  const r = await window.pywebview.api.export_file(t);
  if (r.success) {
    window._setProgress('✅ Đã lưu file ' + t.toUpperCase() + ': ' + r.saved_path, -1);
  } else if (!r.canceled) {
    window._setProgress('❌ ' + r.error, -1);
  } else {
    window._setProgress('Đã hủy lưu file.', -1);
  }
}

async function openDir() {
  window._setProgress('📂 Đang mở thư mục chứa mô hình…', -1);
  await window.pywebview.api.open_folder(lastFolder);
}

async function importModel() {
  window._setProgress('Đang mở hộp thoại nạp file 3D (GLB/OBJ)…', -1);
  const r = await window.pywebview.api.load_external_model();
  if (r && r.success) {
    lastFolder = r.folder;
    const mv = document.getElementById('mv');
    mv.src = r.glb_data;
    mv.style.display = 'block';
    document.getElementById('empty').style.display = 'none';
    document.getElementById('hint').style.display = 'block';
    window._setProgress('✅ Đã nạp thành công mô hình: ' + r.filename + ' – Sẵn sàng lưu GLB / OBJ!', 100);
  } else if (r && r.error) {
    window._setProgress('❌ ' + r.error, -1);
  } else {
    window._setProgress('Đã hủy chọn file.', -1);
  }
}

function openTripoWeb() {
  window.pywebview.api.open_external_url('https://platform.tripo3d.ai');
}

function openMeshyWeb() {
  window.pywebview.api.open_external_url('https://meshy.ai');
}

function openDemo(demoId) {
  window.pywebview.api.open_external_url('https://img2threejs.io/#/demo/' + demoId);
}

function openDemoGallery() {
  window.pywebview.api.open_external_url('https://img2threejs.io/');
}

function openImg2ThreeCode() {
  window.pywebview.api.open_img2threejs_dir();
}

/* ── settings ── */
function openSettings() { document.getElementById('mSet').style.display='flex'; }
function closeSettings() { document.getElementById('mSet').style.display='none'; }
async function saveSettings() {
  await window.pywebview.api.save_settings(
    document.getElementById('hfTok').value,
    document.getElementById('tripoKey').value,
    document.getElementById('meshyKey').value
  );
  closeSettings();
  window._setProgress('✅ Đã lưu cài đặt API Key thành công.', -1);
}

/* ── update ── */
function openUpdate() { document.getElementById('mUpd').style.display='flex'; }
function closeUpdate() { document.getElementById('mUpd').style.display='none'; }
async function checkUpd() {
  document.getElementById('updTxt').textContent = 'Đang kiểm tra từ máy chủ GitHub…';
  document.getElementById('btnApply').style.display='none';
  document.getElementById('updNotes').style.display='none';
  const r = await window.pywebview.api.check_updates();
  if (r.has_update) {
    document.getElementById('updTxt').innerHTML =
      "<span style='color:#10b981;font-weight:700'>🎉 Phát hiện bản mới: "+r.latest_version+" (Hiện tại: "+r.current_version+")</span>";
    if (r.release_notes) {
      document.getElementById('updNotes').textContent = r.release_notes;
      document.getElementById('updNotes').style.display='block';
    }
    dlUrl = r.download_url;
    document.getElementById('btnApply').style.display='inline-block';
  } else {
    document.getElementById('updTxt').innerHTML =
      "<span style='color:#38bdf8'>✓ Bạn đang sử dụng phiên bản mới nhất ("+r.current_version+")</span>";
  }
}

async function applyUpd() {
  if (!dlUrl) return;
  document.getElementById('updTxt').textContent = 'Đang tải gói cập nhật và cài đặt…';
  document.getElementById('btnApply').disabled = true;
  const r = await window.pywebview.api.apply_update(dlUrl);
  if (r.success) {
    document.getElementById('updTxt').innerHTML =
      "<span style='color:#10b981;font-weight:700'>✓ "+r.message+"</span>";
    setTimeout(() => window.pywebview.api.restart_app(), 2000);
  } else {
    document.getElementById('updTxt').innerHTML =
      "<span style='color:#ef4444'>❌ "+r.error+"</span>";
    document.getElementById('btnApply').disabled = false;
  }
}

/* ── 3D Library State & Controller ── */
let _libItems = [];
let _curLibFilter = 'all';

async function updateLibBadge() {
  try {
    const res = await window.pywebview.api.get_library_items();
    if (res && res.success) {
      _libItems = res.items || [];
      const count = res.count || 0;
      const hBadge = document.getElementById('libBadgeHdr');
      const tBadge = document.getElementById('libBadgeToolbar');
      if (hBadge) hBadge.textContent = count;
      if (tBadge) tBadge.textContent = count;
      updateFilterCounts();
    }
  } catch (e) {
    console.warn('updateLibBadge error:', e);
  }
}

function updateFilterCounts() {
  const counts = { all: _libItems.length, local: 0, img2threejs: 0, meshy: 0, tripo: 0, cloud: 0 };
  _libItems.forEach(it => {
    const k = it.engine_key || 'local';
    if (counts[k] !== undefined) counts[k]++;
  });
  const cAll = document.getElementById('cntAll'); if (cAll) cAll.textContent = counts.all;
  const cLoc = document.getElementById('cntLocal'); if (cLoc) cLoc.textContent = counts.local;
  const cImg = document.getElementById('cntImg2Three'); if (cImg) cImg.textContent = counts.img2threejs;
  const cMsy = document.getElementById('cntMeshy'); if (cMsy) cMsy.textContent = counts.meshy;
  const cTrp = document.getElementById('cntTripo'); if (cTrp) cTrp.textContent = counts.tripo;
  const cCld = document.getElementById('cntCloud'); if (cCld) cCld.textContent = counts.cloud;
}

async function openLibrary() {
  document.getElementById('mLib').style.display = 'flex';
  await refreshLibraryData();
}

function closeLibrary() {
  document.getElementById('mLib').style.display = 'none';
}

async function refreshLibraryData() {
  const stats = document.getElementById('libStats');
  if (stats) stats.textContent = 'Đang tải dữ liệu…';
  const res = await window.pywebview.api.get_library_items();
  if (res && res.success) {
    _libItems = res.items || [];
    if (stats) stats.innerHTML = '<b>' + res.count + '</b> mô hình • Dung lượng: <b>' + res.total_size_mb + ' MB</b>';
    const hBadge = document.getElementById('libBadgeHdr');
    const tBadge = document.getElementById('libBadgeToolbar');
    if (hBadge) hBadge.textContent = res.count;
    if (tBadge) tBadge.textContent = res.count;
    updateFilterCounts();
    renderLibGrid();
  } else {
    if (stats) stats.textContent = 'Lỗi nạp thư viện';
  }
}

function setLibFilter(cat) {
  _curLibFilter = cat;
  ['chipAll','chipLocal','chipImg2Three','chipMeshy','chipTripo','chipCloud'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active');
  });
  if (cat === 'all') { const el = document.getElementById('chipAll'); if (el) el.classList.add('active'); }
  else if (cat === 'local') { const el = document.getElementById('chipLocal'); if (el) el.classList.add('active'); }
  else if (cat === 'img2threejs') { const el = document.getElementById('chipImg2Three'); if (el) el.classList.add('active'); }
  else if (cat === 'meshy') { const el = document.getElementById('chipMeshy'); if (el) el.classList.add('active'); }
  else if (cat === 'tripo') { const el = document.getElementById('chipTripo'); if (el) el.classList.add('active'); }
  else if (cat === 'cloud') { const el = document.getElementById('chipCloud'); if (el) el.classList.add('active'); }
  renderLibGrid();
}

function filterLib() {
  renderLibGrid();
}

function renderLibGrid() {
  const grid = document.getElementById('libGrid');
  if (!grid) return;
  const searchInput = document.getElementById('libSearchInput');
  const query = (searchInput ? searchInput.value : '').toLowerCase().trim();

  const filtered = _libItems.filter(item => {
    if (_curLibFilter !== 'all' && item.engine_key !== _curLibFilter) {
      return false;
    }
    if (query) {
      const matchName = (item.name || '').toLowerCase().includes(query);
      const matchPrompt = (item.prompt || '').toLowerCase().includes(query);
      const matchEng = (item.engine || '').toLowerCase().includes(query);
      const matchId = (item.id || '').toLowerCase().includes(query);
      if (!matchName && !matchPrompt && !matchEng && !matchId) return false;
    }
    return true;
  });

  if (filtered.length === 0) {
    grid.innerHTML = '<div class="lib-empty" style="grid-column: 1/-1">' +
      '<svg style="width:48px;height:48px;opacity:.3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">' +
      '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>' +
      '</svg>' +
      '<p style="font-size:13px;font-weight:600">Không tìm thấy mô hình 3D nào</p>' +
      '<p style="font-size:11px;color:#475569">Hãy thử đổi từ khóa tìm kiếm hoặc chọn bộ lọc khác.</p>' +
      '</div>';
    return;
  }

  let html = '';
  filtered.forEach(it => {
    let badgeClass = 'badge-local';
    if (it.engine_key === 'img2threejs') badgeClass = 'badge-img2three';
    else if (it.engine_key === 'meshy') badgeClass = 'badge-meshy';
    else if (it.engine_key === 'tripo') badgeClass = 'badge-tripo';
    else if (it.engine_key === 'cloud') badgeClass = 'badge-cloud';

    const thumbHtml = it.thumb_url
      ? '<img src="' + it.thumb_url + '" alt="' + (it.name || '3D') + '" loading="lazy">'
      : '<div class="lib-thumb-fallback">' +
        '<svg style="width:36px;height:36px;opacity:.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">' +
        '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>' +
        '</svg>' +
        '<span>3D Model</span>' +
        '</div>';

    const safeName = (it.name || 'Mô hình 3D').replace(/"/g, '&quot;');
    const safeId = it.id;

    html += '<div class="lib-card" id="card-' + safeId + '">' +
      '<div class="lib-card-thumb">' +
      thumbHtml +
      '<div class="lib-badge-engine ' + badgeClass + '">' + it.engine + '</div>' +
      '<div class="lib-badge-size">' + it.glb_size + '</div>' +
      '</div>' +
      '<div class="lib-card-body">' +
      '<div class="lib-card-name" title="' + safeName + '">' + safeName + '</div>' +
      '<div class="lib-card-sub">' +
      '<span>📅 ' + it.created_str + '</span>' +
      '<span>' + (it.source === 'text' ? '✍️ Chữ' : '🖼️ Ảnh') + '</span>' +
      '</div>' +
      '<div class="lib-card-actions">' +
      '<button class="lib-btn-act lib-btn-pri" onclick="loadFromLibrary(\'' + safeId + '\')" title="Xem mô hình xoay 360°">👁️ Xem 3D</button>' +
      '<button class="lib-btn-act" onclick="exportFromLibrary(\'' + safeId + '\', \'glb\')" title="Lưu file GLB về máy tính">📦 GLB</button>' +
      (it.has_obj ? '<button class="lib-btn-act" onclick="exportFromLibrary(\'' + safeId + '\', \'obj\')" title="Lưu file OBJ">📦 OBJ</button>' : '') +
      '<button class="lib-btn-act" onclick="openItemFolder(\'' + safeId + '\')" title="Mở thư mục chứa file">📁</button>' +
      '<button class="lib-btn-act lib-btn-del" onclick="deleteFromLibrary(\'' + safeId + '\', \'' + safeName + '\')" title="Xóa mô hình này để giải phóng dung lượng SSD">🗑️</button>' +
      '</div>' +
      '</div>' +
      '</div>';
  });
  grid.innerHTML = html;
}

async function loadFromLibrary(id) {
  window._setProgress('Đang nạp mô hình từ thư viện…', 40);
  const r = await window.pywebview.api.load_library_item(id);
  if (r && r.success) {
    lastFolder = r.folder;
    const mv = document.getElementById('mv');
    mv.src = r.glb_data;
    mv.style.display = 'block';
    document.getElementById('empty').style.display = 'none';
    document.getElementById('hint').style.display = 'block';
    window._setProgress('✨ Đã nạp thành công \'' + r.name + '\' (' + r.engine + ') từ Thư viện!', 100);
    closeLibrary();
  } else {
    alert('❌ Không thể nạp mô hình: ' + ((r && r.error) ? r.error : 'Lỗi không xác định'));
  }
}

async function deleteFromLibrary(id, name) {
  if (!confirm('Bạn có chắc chắn muốn xóa mô hình "' + name + '" khỏi ổ đĩa?\n\nThao tác này sẽ xóa vĩnh viễn và giải phóng dung lượng SSD.')) {
    return;
  }
  const r = await window.pywebview.api.delete_library_item(id);
  if (r && r.success) {
    await refreshLibraryData();
    window._setProgress('🗑️ Đã xóa mô hình "' + name + '" thành công!', -1);
  } else {
    alert('❌ Không thể xóa: ' + ((r && r.error) ? r.error : 'Lỗi không xác định'));
  }
}

async function exportFromLibrary(id, type) {
  const r = await window.pywebview.api.export_library_item(id, type);
  if (r && r.success) {
    window._setProgress('✅ Đã lưu file ' + type.toUpperCase() + ': ' + r.saved_path, -1);
  } else if (r && !r.canceled) {
    alert('❌ Lỗi lưu file: ' + r.error);
  }
}

async function openItemFolder(id) {
  await window.pywebview.api.open_library_item_folder(id);
}

function openLibraryRoot() {
  window.pywebview.api.open_folder();
}
</script>
</body>
</html>"""


def main():
    api = AppApi()
    window = webview.create_window(
        title=f"AI 3D Studio {APP_VERSION}",
        html=HTML,
        js_api=api,
        width=1240,
        height=830,
        min_size=(980, 650),
    )
    api.set_window(window)
    webview.start(debug=False)


if __name__ == "__main__":
    main()
