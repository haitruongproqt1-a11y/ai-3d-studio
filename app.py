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
os.environ["HY3DGEN_MODELS"] = os.path.join(CACHE_DIR, "hy3dgen")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["U2NET_HOME"] = os.path.join(CACHE_DIR, "u2net")

# Add paths
sys.path.insert(0, APP_DIR)
sys.path.append(os.path.join(APP_DIR, "TripoSR"))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import webview
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
import trimesh
import cv2
import rembg

logging.basicConfig(level=logging.INFO)

APP_VERSION = "v2.0.1"
DEFAULT_GITHUB_REPO = "haitruongproqt1-a11y/ai-3d-studio"
OUTPUT_DIR = os.path.join(APP_DIR, "output_app")
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
HUNYUAN_MODEL_DIR = os.path.join(CACHE_DIR, "hy3dgen", "tencent", "Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HUNYUAN_MODEL_DIR, exist_ok=True)

# ── Hardware Detection ──────────────────────────────────────────────────────
def detect_hardware():
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            disp_name = gpu_name if gpu_name.lower().startswith("nvidia") else f"NVIDIA {gpu_name}"
            return {
                "has_gpu": True,
                "device": "cuda:0",
                "name": gpu_name,
                "vram_gb": total_vram_gb,
                "preset": "gpu_high" if total_vram_gb >= 5.5 else "gpu_low",
                "status_text": f"{disp_name} ({total_vram_gb} GB)",
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

# ── Model state holders ───────────────────────────────────────────────────────
model = None                # TripoSR
_model_ready = threading.Event()
hunyuan_pipeline = None     # Hunyuan3D-2 Turbo DiT
_hunyuan_ready = threading.Event()

def is_hunyuan_downloaded():
    config_p = os.path.join(HUNYUAN_MODEL_DIR, "config.yaml")
    model_p = os.path.join(HUNYUAN_MODEL_DIR, "model.fp16.safetensors")
    return os.path.exists(config_p) and os.path.exists(model_p) and os.path.getsize(model_p) >= 3_822_580_000

def load_ai_model():
    """Load TripoSR for Fast Engine with mutual VRAM cleanup."""
    global model, hunyuan_pipeline
    if model is not None and _model_ready.is_set():
        return True
    try:
        logging.info(f"Loading TripoSR on {current_device}…")
        if hunyuan_pipeline is not None:
            try:
                del hunyuan_pipeline
                hunyuan_pipeline = None
                _hunyuan_ready.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(8192)
        model.to(current_device)
        _model_ready.set()
        logging.info("TripoSR model loaded successfully!")
        return True
    except Exception as e:
        logging.error(f"Failed to load TripoSR: {e}")
        return False

def load_hunyuan_model():
    """Load Hunyuan3D-2 Turbo DiT Pipeline for Realistic High-Fidelity Engine."""
    global hunyuan_pipeline, model
    if hunyuan_pipeline is not None and _hunyuan_ready.is_set():
        return True

    if not is_hunyuan_downloaded():
        logging.warning("Hunyuan3D Turbo weights not yet fully downloaded.")
        return False

    try:
        logging.info("Loading Hunyuan3D-2 Turbo Pipeline onto GPU…")
        if model is not None:
            try:
                del model
                model = None
                _model_ready.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini",
            subfolder="hunyuan3d-dit-v2-mini-turbo",
            device=current_device if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True
        )
        _hunyuan_ready.set()
        logging.info("Hunyuan3D-2 Turbo DiT Pipeline loaded successfully!")
        return True
    except Exception as e:
        logging.exception(f"Failed to load Hunyuan3D Turbo: {e}")
        return False

# ── Config helpers ───────────────────────────────────────────────────────────
def load_config():
    cfg = {
        "github_repo": DEFAULT_GITHUB_REPO,
        "default_engine": "turbo"
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
        logging.error(f"save_config error: {e}")

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

# ── Texture & Library helpers ───────────────────────────────────────────────
def enhance_texture(img: Image.Image) -> Image.Image:
    try:
        img = ImageEnhance.Color(img).enhance(1.30)
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    except Exception:
        pass
    return img

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

def _save_model_metadata(item_dir, name="", engine="RTX Đẳng Cấp (Hunyuan3D Turbo)", prompt="", source="image", input_img_path=None):
    try:
        os.makedirs(item_dir, exist_ok=True)
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
            total_bytes += glb_bytes

            obj = os.path.join(p, "model.obj")
            obj_bytes = os.path.getsize(obj) if os.path.exists(obj) else 0
            total_bytes += obj_bytes

            meta_file = os.path.join(p, "meta.json")
            meta = {}
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            engine = meta.get("engine", "")
            if not engine:
                if "hy" in d.lower() or "turbo" in d.lower():
                    engine = "RTX Đẳng Cấp (Hunyuan3D Turbo)"
                else:
                    engine = "RTX Siêu Tốc (TripoSR)"

            engine_key = "turbo" if ("hunyuan" in engine.lower() or "turbo" in engine.lower() or "đẳng cấp" in engine.lower()) else "triposr"

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
            engine = "RTX 3D Studio"
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
            "hunyuan_ready": _hunyuan_ready.is_set(),
            "hunyuan_downloaded": is_hunyuan_downloaded(),
            "config": self.config,
            "latest_model": latest,
        }

    def get_hunyuan_status(self):
        model_p = os.path.join(HUNYUAN_MODEL_DIR, "model.fp16.safetensors")
        dl_bytes = os.path.getsize(model_p) if os.path.exists(model_p) else 0
        total_bytes = 3822584202
        pct = round((dl_bytes / total_bytes) * 100, 1) if total_bytes else 0
        return {
            "downloaded": is_hunyuan_downloaded(),
            "ready": _hunyuan_ready.is_set(),
            "dl_bytes": dl_bytes,
            "total_bytes": total_bytes,
            "pct": pct,
            "dl_str": f"{format_file_size(dl_bytes)} / {format_file_size(total_bytes)}"
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

    # ── live progress helper ─────────────────────────────────────────────────
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
    def start_generate_3d(self, file_path, engine="turbo",
                          mc_resolution=None, bake_tex=False,
                          smooth=True, quality="pbr_1024",
                          num_steps=10, octree_res=160):
        task_id = str(time.time_ns())
        self._tasks[task_id] = {
            "status": "running", "msg": "Đang khởi động tiến trình GPU…",
            "pct": 5, "result": None, "error": None
        }

        def _worker():
            try:
                res = self.generate_3d(
                    file_path, engine=engine, mc_resolution=mc_resolution,
                    bake_tex=bake_tex, smooth=smooth, quality=quality,
                    num_steps=num_steps, octree_res=octree_res, task_id=task_id
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

    def start_generate_from_text(self, prompt, engine="turbo", quality="pbr_1024",
                                 smooth=True, num_steps=10, octree_res=160):
        task_id = str(time.time_ns())
        self._tasks[task_id] = {
            "status": "running", "msg": "Đang phân tích câu lệnh văn bản…",
            "pct": 5, "result": None, "error": None
        }

        def _worker():
            try:
                res = self.generate_from_text(
                    prompt, engine=engine, quality=quality, smooth=smooth,
                    num_steps=num_steps, octree_res=octree_res, task_id=task_id
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
        self._progress("🤖 AI đang phân đoạn & tách nền u2net…", 10)
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

    def _preprocess_hunyuan(self, file_path: str, task_id=None) -> Image.Image:
        self._progress("🤖 AI đang phân đoạn & tách sạch nền trong suốt 100%…", 10, task_id=task_id)
        orig = Image.open(file_path)
        try:
            os.environ["U2NET_HOME"] = os.path.join(CACHE_DIR, "u2net")
            clean_rgba = rembg.remove(orig)
        except Exception as e:
            logging.warning(f"rembg remove warning: {e}")
            clean_rgba = remove_background(orig.convert("RGB"))

        if clean_rgba.mode != "RGBA":
            clean_rgba = clean_rgba.convert("RGBA")

        # Crop to subject bounds with 5% padding so Hunyuan3D centers perfectly on the subject
        arr = np.array(clean_rgba)
        if arr.shape[2] == 4:
            alpha = arr[:, :, 3]
            coords = np.nonzero(alpha > 15)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                h, w = y_max - y_min, x_max - x_min
                pad_y = max(4, int(h * 0.05))
                pad_x = max(4, int(w * 0.05))
                y0 = max(0, y_min - pad_y)
                y1 = min(arr.shape[0], y_max + pad_y)
                x0 = max(0, x_min - pad_x)
                x1 = min(arr.shape[1], x_max + pad_x)
                clean_rgba = clean_rgba.crop((x0, y0, x1, y1))

        return clean_rgba

    # ── TEXT TO 3D PIPELINE ──────────────────────────────────────────────────
    def generate_from_text(self, prompt: str, engine="turbo", quality="pbr_1024",
                           smooth=True, num_steps=10, octree_res=256, task_id=None):
        prompt = prompt.strip()
        if not prompt:
            return {"success": False, "error": "Vui lòng nhập mô tả văn bản cần tạo 3D!"}

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

            req = urllib.request.Request(url, headers={"User-Agent": "AI-3D-Studio/2.0.0"})
            with urllib.request.urlopen(req, timeout=25) as r, open(concept_file, "wb") as f_img:
                shutil.copyfileobj(r, f_img)

            # Send concept image to UI preview immediately
            with open(concept_file, "rb") as f_img:
                b64_img = base64.b64encode(f_img.read()).decode()
            concept_data_url = f"data:image/png;base64,{b64_img}"

            if self._window:
                safe_url = json.dumps(concept_data_url)
                self._window.evaluate_js(f"window._showConcept({safe_url});")

            # 3. Feed directly into selected RTX engine!
            if engine in ("turbo", "hunyuan3d"):
                self._progress("🐉 Đưa hình phác họa vào RTX Hunyuan3D Turbo tái tạo 3D…", 28, task_id=task_id)
                res = self._gen_hunyuan3d_turbo(
                    concept_file, num_steps=num_steps, octree_res=octree_res,
                    target_dir=item_dir, task_id=task_id
                )
            else:
                self._progress("⚡ Đưa hình phác họa vào GPU RTX TripoSR tái tạo 3D…", 28, task_id=task_id)
                res = self._gen_local(
                    concept_file, quality=quality, smooth=smooth,
                    target_dir=item_dir, task_id=task_id
                )

            if res.get("success"):
                res["concept_data"] = concept_data_url
                res["prompt"] = prompt
                _save_model_metadata(
                    item_dir,
                    name=prompt,
                    engine=res.get("engine_used", "RTX (Từ văn bản)"),
                    prompt=prompt,
                    source="text",
                    input_img_path=concept_file
                )
            return res

        except Exception as e:
            logging.exception("generate_from_text error")
            return {"success": False, "error": f"Lỗi tạo ảnh phác họa: {e}\n(Bạn có thể chuyển sang thẻ 'Từ Hình Ảnh' để nạp ảnh trực tiếp)"}

    # ── MAIN 3D GENERATION ROUTER ────────────────────────────────────────────
    def generate_3d(self, file_path, engine="turbo",
                    mc_resolution=None, bake_tex=False,
                    smooth=True, quality="pbr_1024",
                    num_steps=10, octree_res=160, task_id=None):
        if engine in ("turbo", "hunyuan3d"):
            return self._gen_hunyuan3d_turbo(
                file_path, num_steps=num_steps, octree_res=octree_res, task_id=task_id
            )
        return self._gen_local(file_path, quality=quality, smooth=smooth, task_id=task_id)

    # ── ENGINE 1: LOCAL FAST (RTX TRIPOSR ~15s) ──────────────────────────────
    def _gen_local(self, file_path, quality="pbr_1024", smooth=True, target_dir=None, task_id=None):
        global model, current_device
        if model is None or not _model_ready.is_set():
            self._progress("⏳ Đang nạp TripoSR vào GPU RTX 3050 (~15s)…", 5, task_id=task_id)
            ok = load_ai_model()
            if not ok or model is None:
                return {"success": False, "error": "Model TripoSR nạp thất bại. Vui lòng kiểm tra dung lượng VRAM!"}

        try:
            do_bake = (quality == "pbr_1024" or quality == "bake")
            mc_res = 320 if quality == "ultra_320" else 256

            self._progress("🖼 Khử bóng đổ, tách viền & tiền xử lý ảnh…", 15, task_id=task_id)
            image = self._preprocess(file_path)

            if target_dir:
                item_dir = target_dir
            else:
                ts = int(time.time())
                item_dir = os.path.join(OUTPUT_DIR, str(ts))
                os.makedirs(item_dir, exist_ok=True)

            image.save(os.path.join(item_dir, "input.png"))

            self._progress(f"🧠 RTX 3050 suy luận Tensor không gian ({HARDWARE_INFO['name']})…", 30, task_id=task_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                scene_codes = model([image], device=current_device)

            self._progress(f"⚙️ Tái tạo lưới Marching Cubes ({mc_res}x{mc_res} Voxels)…", 45, task_id=task_id)
            meshes = model.extract_mesh(
                scene_codes,
                has_vertex_color=(not do_bake),
                resolution=mc_res,
            )

            # Taubin non-shrinking smoothing
            if smooth:
                self._progress("✨ Làm mịn bề mặt Taubin (khử bậc thang, giữ nguyên thể tích)…", 55, task_id=task_id)
                try:
                    trimesh.smoothing.filter_taubin(meshes[0], lamb=0.5, nu=-0.53, iterations=10)
                except Exception:
                    pass

            glb_path = os.path.join(item_dir, "model.glb")
            obj_path = os.path.join(item_dir, "model.obj")

            # ── HD UV ATLAS TEXTURE BAKING (1024x1024) ──
            if do_bake:
                self._progress("🎨 AI đang trải UV & nướng vân Atlas 1024x1024 HD…", 65, task_id=task_id)
                mesh_trimesh = meshes[0]
                try:
                    import xatlas
                    v = mesh_trimesh.vertices
                    f = mesh_trimesh.faces
                    atlas = xatlas.Atlas()
                    atlas.add_mesh(v, f)
                    chart_options = xatlas.ChartOptions()
                    chart_options.max_iterations = 2
                    pack_options = xatlas.PackOptions()
                    pack_options.width = 1024
                    pack_options.height = 1024
                    pack_options.padding = 2
                    atlas.generate(chart_options=chart_options, pack_options=pack_options)
                    vmapping, indices, uvs = atlas[0]

                    v_cam = vmapping.copy()
                    zs = v[v_cam, 2]
                    v_cam = v_cam[np.argsort(zs)]
                    u_coords = np.clip(((v[v_cam, 0] + 0.5) * 1023), 0, 1023).astype(int)
                    v_coords = np.clip(((1.0 - (v[v_cam, 1] + 0.5)) * 1023), 0, 1023).astype(int)

                    img_arr = np.array(image.resize((1024, 1024), Image.Resampling.LANCZOS))
                    colors = img_arr[v_coords, u_coords]

                    tex_img = np.full((1024, 1024, 3), 128, dtype=np.uint8)
                    tex_u = np.clip((uvs[np.argsort(zs), 0] * 1023), 0, 1023).astype(int)
                    tex_v = np.clip(((1.0 - uvs[np.argsort(zs), 1]) * 1023), 0, 1023).astype(int)
                    tex_img[tex_v, tex_u] = colors[:, :3]

                    mask = np.any(tex_img != 128, axis=-1).astype(np.uint8)
                    tex_img = cv2.inpaint(tex_img, 1 - mask, 3, cv2.INPAINT_TELEA)

                    pil_tex = Image.fromarray(tex_img)
                    pil_tex = enhance_texture(pil_tex)

                    material = trimesh.visual.material.PBRMaterial(
                        baseColorTexture=pil_tex,
                        roughnessFactor=0.45,
                        metallicFactor=0.08,
                    )
                    baked_mesh = trimesh.Trimesh(
                        vertices=v[vmapping],
                        faces=indices,
                        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
                        process=False,
                    )
                    meshes = [baked_mesh]
                except Exception as e_bake:
                    logging.warning(f"Bake texture fallback: {e_bake}")

            self._progress("💾 Đang xuất tệp mô hình GLB và OBJ…", 88, task_id=task_id)
            meshes[0].export(glb_path)
            meshes[0].export(obj_path)

            self.last_glb = glb_path
            self.last_obj = obj_path
            self.last_folder = item_dir

            _save_model_metadata(
                item_dir,
                name=os.path.splitext(os.path.basename(file_path))[0],
                engine="RTX Siêu Tốc (TripoSR)",
                source="image",
                input_img_path=file_path
            )

            self._progress("✅ Hoàn tất! Mô hình 3D sẵn sàng.", 100, task_id=task_id)
            with open(glb_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{b64}",
                "glb_path": glb_path,
                "obj_path": obj_path,
                "folder": item_dir,
                "engine_used": "RTX Siêu Tốc (TripoSR)"
            }

        except Exception as e:
            logging.exception("_gen_local error")
            return {"success": False, "error": f"Lỗi tạo 3D RTX: {e}"}

    # ── ENGINE 2: LOCAL REALISTIC (HUNYUAN3D-2 TURBO DIT FLOW MATCHING) ─────
    def _gen_hunyuan3d_turbo(self, file_path, num_steps=10, octree_res=160, target_dir=None, task_id=None):
        global hunyuan_pipeline

        if not is_hunyuan_downloaded():
            model_p = os.path.join(HUNYUAN_MODEL_DIR, "model.fp16.safetensors")
            dl_bytes = os.path.getsize(model_p) if os.path.exists(model_p) else 0
            total_bytes = 3822584202
            pct = round((dl_bytes / total_bytes) * 100, 1)
            return {
                "success": False,
                "error": f"Mô hình Hunyuan3D-2 Turbo đang được tải về ổ đĩa trong nền: {format_file_size(dl_bytes)} / {format_file_size(total_bytes)} ({pct}%).\n\n👉 Vui lòng đợi trong giây lát hoặc chuyển sang '⚡ RTX Siêu Tốc (TripoSR)' để tạo ngay lập tức 100% offline!"
            }

        if hunyuan_pipeline is None or not _hunyuan_ready.is_set():
            self._progress("⏳ Đang nạp Hunyuan3D-2 Turbo vào GPU RTX 3050 (8%)…", 8, task_id=task_id)
            ok = load_hunyuan_model()
            if not ok or hunyuan_pipeline is None:
                return {"success": False, "error": "Không thể nạp mô hình Hunyuan3D-2 Turbo vào VRAM GPU."}

        try:
            image = self._preprocess_hunyuan(file_path, task_id=task_id)

            if target_dir:
                item_dir = target_dir
            else:
                ts = int(time.time())
                item_dir = os.path.join(OUTPUT_DIR, f"hy_{ts}")
                os.makedirs(item_dir, exist_ok=True)

            image.save(os.path.join(item_dir, "input.png"))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_steps = int(num_steps)
            self._progress(f"🐉 RTX 3050 suy luận Flow Matching: Bước 0/{total_steps} (25%)…", 25, task_id=task_id)

            def step_cb(step_idx, t, outputs):
                cur = step_idx + 1
                pct = int(25 + (cur / total_steps) * 45)
                if cur == total_steps:
                    self._progress(
                        "⚙️ RTX 3050 đang giải mã không gian ShapeVAE & trích xuất bề mặt mesh (72%)…",
                        72,
                        task_id=task_id
                    )
                else:
                    self._progress(
                        f"🐉 RTX 3050 suy luận Flow Matching: Bước {cur}/{total_steps} ({pct}%)…",
                        pct,
                        task_id=task_id
                    )

            with torch.no_grad():
                mesh_outputs = hunyuan_pipeline(
                    image=image,
                    num_inference_steps=total_steps,
                    octree_resolution=int(octree_res),
                    num_chunks=24000,
                    output_type="trimesh",
                    enable_pbar=False,
                    callback=step_cb,
                    callback_steps=1
                )

            self._progress("⚙️ Đang giải mã ShapeVAE & trích xuất bề mặt 3D (75%)…", 75, task_id=task_id)
            mesh = mesh_outputs[0]
            if isinstance(mesh, list):
                mesh = mesh[0]

            # ── HD COLOR & TEXTURE PROJECTION ──
            self._progress("🎨 AI đang chiếu màu & ánh sáng gốc lên mô hình 3D (85%)…", 85, task_id=task_id)
            try:
                img_w, img_h = image.size
                img_np = np.array(image)
                v = mesh.vertices.copy()
                vn = mesh.vertex_normals
                min_b = v.min(axis=0)
                max_b = v.max(axis=0)
                span = max_b - min_b
                span[span == 0] = 1.0

                u = np.clip(((v[:, 0] - min_b[0]) / span[0]) * (img_w - 1), 0, img_w - 1).astype(int)
                y_norm = (v[:, 1] - min_b[1]) / span[1]
                v_coord = np.clip((1.0 - y_norm) * (img_h - 1), 0, img_h - 1).astype(int)
                sampled_colors = img_np[v_coord, u].copy()

                # Fix back projection bleed: If normal points backwards, blend with smooth ambient tone
                is_back = vn[:, 2] < -0.1
                if np.any(is_back):
                    valid_mask = sampled_colors[:, 3] > 128
                    body_tone = np.median(sampled_colors[valid_mask], axis=0) if np.any(valid_mask) else np.array([200, 200, 200, 255])
                    blend = np.clip((-vn[is_back, 2] - 0.1) / 0.9, 0.0, 0.85)[:, None]
                    sampled_colors[is_back] = (sampled_colors[is_back] * (1.0 - blend) + body_tone * blend).astype(np.uint8)

                mesh.visual.vertex_colors = sampled_colors
            except Exception as e_col:
                logging.warning(f"Color projection fallback: {e_col}")

            self._progress("💾 Đang xuất tệp mô hình GLB và OBJ sắc nét (95%)…", 95, task_id=task_id)
            glb_path = os.path.join(item_dir, "model.glb")
            obj_path = os.path.join(item_dir, "model.obj")

            mesh.export(glb_path)
            mesh.export(obj_path)

            self.last_glb = glb_path
            self.last_obj = obj_path
            self.last_folder = item_dir

            _save_model_metadata(
                item_dir,
                name=os.path.splitext(os.path.basename(file_path))[0],
                engine="RTX Đẳng Cấp (Hunyuan3D Turbo)",
                source="image",
                input_img_path=file_path
            )

            self._progress("✅ Hoàn tất! Mô hình 3D Hunyuan Turbo sẵn sàng (100%).", 100, task_id=task_id)
            with open(glb_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{b64}",
                "glb_path": glb_path,
                "obj_path": obj_path,
                "folder": item_dir,
                "engine_used": "RTX Đẳng Cấp (Hunyuan3D Turbo)"
            }
        except Exception as e:
            logging.exception("Hunyuan3D Turbo generation error")
            return {"success": False, "error": f"Lỗi tạo 3D Hunyuan Turbo: {e}"}

    # ── file operations ──────────────────────────────────────────────────────
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
                return {"success": False, "error": "Chưa có mô hình 3D! Vui lòng tạo mô hình trước."}

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
                    for extra in ("texture.png", "model.mtl", "normal.png", "material_0.png", "material.mtl"):
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
                "glb_path": local_glb,
                "obj_path": self.last_obj,
                "folder": item_dir,
                "filename": os.path.basename(fp)
            }
        except Exception as e:
            return {"success": False, "error": f"Không thể đọc file 3D: {e}"}

    def open_external_url(self, url):
        try:
            import webbrowser
            webbrowser.open(url)
            return True
        except Exception as e:
            logging.error(f"open_external_url: {e}")
            return False

    # ── Auto-update OTA ──────────────────────────────────────────────────────
    def check_updates(self):
        repo = self.config.get("github_repo", DEFAULT_GITHUB_REPO)
        url = f"https://raw.githubusercontent.com/{repo}/main/version.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AI-3D-Studio"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
            latest = str(data.get("version", APP_VERSION)).strip()
            rel_notes_raw = data.get("releaseNotes") or data.get("release_notes", "")
            if isinstance(rel_notes_raw, list):
                rel_notes = "\n".join(f"• {note}" for note in rel_notes_raw)
            else:
                rel_notes = str(rel_notes_raw)
            dl_url = data.get("ota_url") or data.get("download_url") or ""
            has_update = latest.lstrip("v") != APP_VERSION.strip().lstrip("v")
            return {
                "has_update": has_update,
                "current_version": APP_VERSION,
                "latest_version": f"v{latest.lstrip('v')}",
                "release_notes": rel_notes,
                "download_url": dl_url,
            }
        except Exception as e:
            logging.warning(f"check_updates error: {e}")
            return {"has_update": False, "error": str(e), "current_version": APP_VERSION}

    def apply_update(self, download_url):
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
<title>AI 3D Studio – RTX Siêu Tốc & RTX Đẳng Cấp (100% Offline GPU)</title>
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

/* ── Creation Source Switcher ── */
.src-switcher{display:flex;background:#131825;padding:3px;border-radius:9px;border:1px solid #242d40;gap:4px}
.src-tab{flex:1;padding:8px;border:none;border-radius:7px;background:transparent;color:#94a3b8;font-size:11.5px;font-weight:700;cursor:pointer;transition:.18s;display:flex;align-items:center;justify-content:center;gap:5px}
.src-tab.active{background:linear-gradient(135deg,#2563eb,#4f46e5);color:#fff;box-shadow:0 2px 8px rgba(37,99,235,.4)}

/* ── 2 Local Offline RTX Engine Tabs ── */
.tabs{display:grid;grid-template-columns:1fr 1fr;background:#07090e;padding:3px;border-radius:8px;border:1px solid #1e2433;gap:3px}
.tab{padding:8px 6px;border:none;border-radius:6px;background:transparent;color:#64748b;font-size:11px;font-weight:700;cursor:pointer;transition:.15s;text-align:center;white-space:nowrap}
.tab.on{background:linear-gradient(135deg,#1d4ed8,#2563eb);color:#fff;box-shadow:0 2px 8px rgba(37,99,235,.35)}
.tab.turbo.on{background:linear-gradient(135deg,#7c3aed,#db2777);color:#fff;box-shadow:0 2px 10px rgba(219,39,119,.4)}

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
.btn-gen{background:linear-gradient(135deg,#7c3aed,#db2777);color:#fff;border:none;padding:12px;border-radius:8px;font-size:12.5px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:7px;box-shadow:0 4px 14px rgba(219,39,119,.35);transition:.2s}
.btn-gen.triposr-mode{background:linear-gradient(135deg,#1d4ed8,#2563eb);box-shadow:0 4px 14px rgba(37,99,235,.3)}
.btn-gen:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 6px 18px rgba(219,39,119,.45)}
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
.badge-turbo{background:rgba(192,38,211,.85);color:#fff}
.badge-triposr{background:rgba(16,185,129,.85);color:#fff}

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
    <span class="ver" id="ver">v2.0.0</span>
  </div>
  <div class="hdr-right">
    <div class="gpu-pill"><div class="dot"></div><span id="gpuTxt">Đang nạp card GPU…</span></div>
    <button class="btn-lib-hdr" onclick="openLibrary()" title="Mở Thư viện quản lý các mô hình 3D đã tạo">
      🏛️ Thư viện 3D <span class="badge-pill" id="libBadgeHdr">0</span>
    </button>
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

    <!-- 2 Engine tabs: Offline RTX Only -->
    <div class="tabs">
      <button class="tab turbo on" id="tabTurbo" onclick="setMode('turbo')">🐉 RTX Đẳng Cấp (Hunyuan Turbo)</button>
      <button class="tab" id="tabTripoSR" onclick="setMode('triposr')">⚡ RTX Siêu Tốc (TripoSR)</button>
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
      <textarea id="promptInput" class="txt-prompt" placeholder="Nhập mô tả bằng tiếng Việt hoặc tiếng Anh (ví dụ: Quả chuối vàng chín mọng, Ghế sofa bọc da sang trọng, Siêu xe Ferrari, Thanh kiếm hiệp sĩ…)"></textarea>
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
      <span class="ic-title" id="icTitle">🐉 RTX Đẳng Cấp – Tencent Hunyuan3D-2 Turbo</span>
      <span class="ic-desc" id="icDesc">Kiến trúc DiT Flow Matching thế hệ mới, <b>tách khối sắc nét, mô hình thực và chất lượng cao</b>, không bị biến dạng hay dính bệt.</span>
    </div>

    <!-- Hunyuan3D Turbo Settings -->
    <div id="turboSet">
      <div class="sg" style="margin-bottom:7px">
        <label>Số bước suy luận Flow Matching (RTX 3050):</label>
        <select id="turboSteps">
          <option value="10" selected>🚀 10 bước Turbo (~35s) – Mịn màng, cân đối hoàn hảo (Khuyên dùng)</option>
          <option value="15">💎 15 bước Ultra (~50s) – Tăng cường độ nét cấu trúc phức tạp</option>
          <option value="8">⚡ 8 bước Fast (~25s) – Xem nhanh</option>
        </select>
      </div>
      <div class="sg" style="margin-bottom:7px">
        <label>Độ phân giải không gian Octree:</label>
        <select id="turboOctree">
          <option value="160" selected>⚡ 160 Octree (~1 phút) – Cực nhanh, chuẩn nhẹ cho Game & Mixamo Rigging</option>
          <option value="192">🚀 192 Octree (~2 phút) – Cân bằng sắc nét & tốc độ (Khuyên dùng)</option>
          <option value="256">💎 256 Octree (~6-8 phút) – Siêu chi tiết, lưới dày</option>
        </select>
      </div>
    </div>

    <!-- TripoSR Fast Settings -->
    <div id="triposrSet" style="display:none">
      <div class="sg" style="margin-bottom:7px">
        <label>Chất lượng hình học TripoSR:</label>
        <select id="quality">
          <option value="pbr_1024" selected>💎 PBR 1024px + Làm mịn Taubin (~15s) – Mịn màng</option>
          <option value="ultra_320">📐 Ultra HD 320 (~10s) – 43.000 điểm lưới</option>
          <option value="fast_256">⚡ Siêu tốc Vertex Colors 256 (~3s)</option>
        </select>
      </div>
      <div class="chk-row" style="margin-bottom:6px">
        <input type="checkbox" id="chkSmooth" checked>
        <label class="chk-row" for="chkSmooth">✨ Làm mịn Taubin (khử bậc thang, giữ thể tích)</label>
      </div>
    </div>

    <button class="btn-gen" id="btnGen" onclick="generate()">
      🐉 BẮT ĐẦU TẠO 3D (RTX ĐẲNG CẤP)
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
    <!-- Top Action Toolbar -->
    <div class="stage-toolbar">
      <!-- Lighting Presets -->
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
        <button class="btn-act ready" id="btnLoadLast" onclick="loadLastModel()" style="display:none;background:linear-gradient(135deg,#0284c7,#2563eb);color:#fff" title="Xem lại mô hình 3D vừa tạo">
          ↺ Xem mô hình trước
        </button>
        <button class="btn-act ready" id="btnImport" onclick="importModel()" title="Nạp file 3D GLB/OBJ từ máy tính">
          📂 Nạp 3D ngoài
        </button>
        <button class="btn-act ready" id="btnGlb" onclick="doExport('glb')" title="Lưu định dạng GLB">
          📦 Lưu GLB
        </button>
        <button class="btn-act ready" id="btnObj" onclick="doExport('obj')" title="Lưu định dạng OBJ cho Blender/Maya">
          📦 Lưu OBJ
        </button>
        <button class="btn-act ready" id="btnDir" onclick="openDir()" title="Mở thư mục chứa file đã tạo">
          📁 Mở thư mục
        </button>
        <button class="btn-act ready" id="btnMixamo" onclick="openMixamoModal()" style="background:linear-gradient(135deg,#d97706,#b45309);color:#fff;border-color:#f59e0b" title="Gắn khung xương tự động và tạo động tác Game qua Mixamo">
          🦴 Gắn Xương (Mixamo)
        </button>
      </div>
    </div>

    <!-- Viewport -->
    <div class="vp">
      <model-viewer id="mv"
        camera-controls
        auto-rotate
        auto-rotate-delay="3000"
        rotation-per-second="25deg"
        interaction-prompt="none"
        shadow-intensity="1.5"
        shadow-softness="0.5"
        exposure="1.15"
        tone-mapping="aces"
        style="display:none">
      </model-viewer>

      <div class="vp-empty" id="empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
          <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
          <line x1="12" y1="22.08" x2="12" y2="12"/>
        </svg>
        <p style="font-size:13px;font-weight:600">Khung xem mô hình 3D thực tế</p>
        <p style="font-size:11px;color:#475569">Nạp ảnh hoặc nhập văn bản ở cột trái để tạo mô hình</p>
      </div>

      <div class="hint" id="hint" style="display:none">
        🖱️ Chuột trái: Xoay 360° | Chuột phải: Di chuyển | Con lăn: Thu phóng
      </div>
    </div>
  </div>
</div>

<!-- ── 3D Library Modal (Full screen sleek grid) ── -->
<div class="modal-bg" id="mLib">
  <div class="modal modal-lib">
    <div class="lib-header">
      <div class="lib-title-row">
        <span class="lib-title">🏛️ Thư viện Mô hình 3D Studio</span>
        <span class="lib-stats" id="libStats">0 mô hình • 0 MB</span>
      </div>
      <button class="m-x" onclick="closeLibrary()">✕</button>
    </div>

    <div class="lib-controls">
      <div class="lib-search-row">
        <input type="text" class="lib-search-input" id="libSearchInput" placeholder="🔍 Tìm kiếm mô hình theo tên, câu lệnh hoặc mã ID..." oninput="filterLib()">
        <button class="btn-lib-folder" onclick="openLibraryRoot()">📁 Mở thư mục lưu trữ</button>
      </div>
      <div class="lib-chips">
        <button class="lib-chip active" id="chipAll" onclick="setLibFilter('all')">Tất cả <span class="chip-num" id="cntAll">0</span></button>
        <button class="lib-chip" id="chipTurbo" onclick="setLibFilter('turbo')">🐉 RTX Đẳng Cấp <span class="chip-num" id="cntTurbo">0</span></button>
        <button class="lib-chip" id="chipTripoSR" onclick="setLibFilter('triposr')">⚡ RTX Siêu Tốc <span class="chip-num" id="cntTripoSR">0</span></button>
      </div>
    </div>

    <div class="lib-grid-wrap">
      <div class="lib-grid" id="libGrid"></div>
    </div>
  </div>
</div>

<!-- Update Modal -->
<div class="modal-bg" id="mUpd">
  <div class="modal">
    <div class="m-hdr">
      <span class="m-title">Kiểm tra Cập nhật</span>
      <button class="m-x" onclick="closeUpdate()">✕</button>
    </div>
    <p id="updTxt" style="font-size:12px;color:#94a3b8">Đang kiểm tra từ GitHub…</p>
    <div class="notes" id="updNotes"></div>
    <div class="m-foot">
      <button class="btn-m btn-m-sec" onclick="closeUpdate()">Đóng</button>
      <button class="btn-m btn-m-pri" id="btnApply" style="display:none" onclick="applyUpd()">Cài đặt ngay</button>
    </div>
  </div>
</div>

<!-- Mixamo Auto-Rigging Modal -->
<div class="modal-bg" id="mMixamo">
  <div class="modal" style="width:620px">
    <div class="m-hdr">
      <span class="m-title" style="color:#f59e0b;display:flex;align-items:center;gap:7px">
        🦴 Gắn Xương Tự Động & Động Tác Game (Adobe Mixamo)
      </span>
      <button class="m-x" onclick="closeMixamoModal()">✕</button>
    </div>
    <div style="font-size:12px;color:#cbd5e1;line-height:1.6;display:flex;flex-direction:column;gap:10px">
      <p>Adobe Mixamo là nền tảng gắn khung xương nhân vật 3D <b>tự động số 1 thế giới</b> và cung cấp hơn <b>2.500 động tác Game hoàn toàn miễn phí</b>.</p>
      
      <div style="background:#0b0e14;border:1px solid #1e2638;border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:8px">
        <b style="color:#60a5fa">Quy trình 3 bước cực kỳ đơn giản:</b>
        <div>1️⃣ Nhấn nút <b>"📁 Mở thư mục chứa file OBJ"</b> bên dưới để lấy file <code>model.obj</code>.</div>
        <div>2️⃣ Nhấn nút <b>"🌐 Mở Adobe Mixamo"</b>, đăng nhập miễn phí, bấm nút <b>Upload Character</b> rồi kéo thả file vào.</div>
        <div>3️⃣ Kéo 5 điểm khớp (Cằm, 2 Cổ tay, 2 Đầu gối, Háng) -> Chọn động tác Game bạn thích (Chạy, nhảy, chiến đấu...) và tải file <b>FBX</b> nạp thẳng vào Game!</div>
      </div>
    </div>
    <div class="m-foot" style="gap:8px">
      <button class="btn-m btn-m-sec" onclick="closeMixamoModal()">Đóng</button>
      <button class="btn-m btn-m-sec" onclick="openDir()">📁 Mở thư mục chứa file OBJ</button>
      <button class="btn-m" style="background:linear-gradient(135deg,#d97706,#b45309);color:#fff;border:none" onclick="openMixamoWeb()">🌐 Mở Adobe Mixamo</button>
    </div>
  </div>
</div>

<script>
let curSource = 'image';
let curMode = 'turbo';
let imgPath = null;
let lastFolder = '';
let dlUrl = '';

/* ── Progress helper exposed to Python ── */
window._setProgress = function(msg, pct) {
  const t = document.getElementById('progTxt');
  const w = document.getElementById('barWrap');
  const b = document.getElementById('bar');
  const s = document.getElementById('spin');
  if (t) t.textContent = msg;
  if (pct !== undefined && pct >= 0) {
    if (w) w.style.display = 'block';
    if (b) b.style.width = pct + '%';
    if (pct < 100) {
      if (s) s.style.display = 'inline-block';
    } else {
      if (s) s.style.display = 'none';
    }
  } else {
    if (s) s.style.display = 'none';
  }
};

window._showConcept = function(dataUrl) {
  const box = document.getElementById('conceptBox');
  const img = document.getElementById('conceptImg');
  const txt = document.getElementById('conceptTxt');
  if (box && img) {
    img.src = dataUrl;
    box.style.display = 'flex';
    if (txt) txt.textContent = 'Đang chuyển vào nhân GPU RTX 3050 tái tạo 3D…';
  }
};

/* ── Init ── */
window.addEventListener('pywebviewready', async () => {
  try {
    const init = await window.pywebview.api.get_init_data();
    if (init) {
      if (init.version) {
        document.getElementById('ver').textContent = init.version;
      }
      if (init.hardware) {
        document.getElementById('gpuTxt').textContent = init.hardware.status_text;
      }
      if (init.latest_model && init.latest_model.has_model) {
        lastFolder = init.latest_model.folder;
        document.getElementById('btnLoadLast').style.display = 'inline-flex';
      }
      updateLibBadge();
    }
  } catch(e) {
    console.warn('init error:', e);
  }
});

/* ── View latest model from previous session ── */
async function loadLastModel() {
  window._setProgress('Đang nạp lại mô hình từ phiên trước…', 30);
  const r = await window.pywebview.api.load_latest_model();
  if (r && r.success) {
    lastFolder = r.folder;
    const mv = document.getElementById('mv');
    mv.src = r.glb_data;
    mv.style.display = 'block';
    document.getElementById('empty').style.display = 'none';
    document.getElementById('hint').style.display = 'block';
    window._setProgress('✨ Đã nạp lại mô hình 3D từ phiên trước!', 100);
  } else {
    window._setProgress('❌ ' + ((r && r.error) ? r.error : 'Không thể nạp mô hình'), -1);
  }
}

/* ── source switch ── */
function switchSource(src) {
  curSource = src;
  const tabImg = document.getElementById('srcTabImg');
  const tabTxt = document.getElementById('srcTabTxt');
  const boxImg = document.getElementById('imageBox');
  const boxTxt = document.getElementById('textBox');

  if (src === 'image') {
    tabImg.classList.add('active');
    tabTxt.classList.remove('active');
    boxImg.style.display = 'block';
    boxTxt.style.display = 'none';
  } else {
    tabTxt.classList.add('active');
    tabImg.classList.remove('active');
    boxTxt.style.display = 'flex';
    boxImg.style.display = 'none';
  }
  updateGenBtnText();
}

function setPrompt(text) {
  const p = document.getElementById('promptInput');
  p.value = text;
  p.focus();
}

function updateGenBtnText() {
  const btn = document.getElementById('btnGen');
  if (curMode === 'turbo') {
    btn.className = 'btn-gen';
    btn.innerHTML = (curSource === 'text') ? '🐉 BẮT ĐẦU TẠO 3D (TỪ VĂN BẢN)' : '🐉 BẮT ĐẦU TẠO 3D (RTX ĐẲNG CẤP)';
  } else {
    btn.className = 'btn-gen triposr-mode';
    btn.innerHTML = (curSource === 'text') ? '⚡ BẮT ĐẦU TẠO 3D (TỪ VĂN BẢN)' : '⚡ BẮT ĐẦU TẠO 3D (RTX SIÊU TỐC)';
  }
}

/* ── engine switch ── */
function setMode(m) {
  curMode = m;
  document.getElementById('tabTurbo').classList.remove('on');
  document.getElementById('tabTripoSR').classList.remove('on');
  document.getElementById('turboSet').style.display = 'none';
  document.getElementById('triposrSet').style.display = 'none';

  const progTxt = document.getElementById('progTxt');
  const barWrap = document.getElementById('barWrap');
  const bar = document.getElementById('bar');
  const spin = document.getElementById('spin');
  if (barWrap) barWrap.style.display = 'none';
  if (bar) bar.style.width = '0%';
  if (spin) spin.style.display = 'none';

  if (m === 'turbo') {
    document.getElementById('tabTurbo').classList.add('on');
    document.getElementById('turboSet').style.display = '';
    document.getElementById('icTitle').textContent = '🐉 RTX Đẳng Cấp – Tencent Hunyuan3D-2 Turbo';
    document.getElementById('icDesc').innerHTML = 'Kiến trúc DiT Flow Matching thế hệ mới, <b>tách khối sắc nét, mô hình thực và chất lượng cao</b>, không bị biến dạng hay dính bệt.';
    if (window.pywebview && window.pywebview.api) {
      window.pywebview.api.get_hunyuan_status().then(st => {
        if (st && !st.downloaded) {
          if (progTxt) progTxt.innerHTML = `<span style="color:#f59e0b;font-weight:600">⏳ Mô hình Hunyuan Turbo đang tải trong nền: ${st.dl_str} (${st.pct}%). Bạn có thể dùng '⚡ RTX Siêu Tốc' ngay lập tức!</span>`;
        } else {
          if (progTxt) progTxt.innerHTML = '<span style="color:#f472b6;font-weight:600">🐉 Đã chọn RTX Đẳng Cấp: Mô hình 3D thực tế chi tiết cao. Bấm nút bên dưới để tạo!</span>';
        }
      }).catch(() => {
        if (progTxt) progTxt.innerHTML = '<span style="color:#f472b6;font-weight:600">🐉 Đã chọn RTX Đẳng Cấp: Mô hình 3D thực tế chi tiết cao. Bấm nút bên dưới để tạo!</span>';
      });
    } else {
      if (progTxt) progTxt.innerHTML = '<span style="color:#f472b6;font-weight:600">🐉 Đã chọn RTX Đẳng Cấp: Mô hình 3D thực tế chi tiết cao. Bấm nút bên dưới để tạo!</span>';
    }
  } else {
    document.getElementById('tabTripoSR').classList.add('on');
    document.getElementById('triposrSet').style.display = '';
    document.getElementById('icTitle').textContent = '⚡ RTX Siêu Tốc – TripoSR Offline (~15s)';
    document.getElementById('icDesc').innerHTML = 'Tạo 3D siêu nhanh trong <b>~15 giây</b> trên GPU RTX 3050. Phù hợp phác thảo nhanh ý tưởng và thử nghiệm.';
    if (progTxt) progTxt.innerHTML = '<span style="color:#34d399;font-weight:600">⚡ Đã chọn RTX Siêu Tốc: Tạo nhanh chỉ ~15 giây. Bấm nút bên dưới để tạo!</span>';
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
  const quality = document.getElementById('quality').value;
  const smooth = document.getElementById('chkSmooth').checked;
  const numSteps = document.getElementById('turboSteps').value;
  const octreeRes = document.getElementById('turboOctree').value;

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
  window._setProgress('Đang khởi động tiến trình GPU…', 5);

  function finishRun() {
    btn.disabled = false;
    spin.style.display = 'none';
  }

  try {
    const prompt = (curSource === 'text') ? document.getElementById('promptInput').value.trim() : '';
    const startRes = (curSource === 'text')
      ? await window.pywebview.api.start_generate_from_text(prompt, curMode, quality, smooth, numSteps, octreeRes)
      : await window.pywebview.api.start_generate_3d(imgPath, curMode, 'auto', (quality === 'pbr_1024'), smooth, quality, numSteps, octreeRes);

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
    window._setProgress('💡 Chế độ ánh sáng: Studio ACES (Chuẩn thực tế)', -1);
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
    window._setProgress('💡 Chế độ ánh sáng: Tự nhiên Studio (Mềm mại)', -1);
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

/* ── update ── */
function openUpdate() {
  document.getElementById('mUpd').style.display = 'flex';
  checkUpd();
}
function closeUpdate() { document.getElementById('mUpd').style.display = 'none'; }
async function checkUpd() {
  document.getElementById('updTxt').innerHTML = "<span style='color:#38bdf8'>⏳ Đang kết nối máy chủ GitHub kiểm tra phiên bản mới…</span>";
  document.getElementById('btnApply').style.display = 'none';
  document.getElementById('updNotes').style.display = 'none';
  try {
    const r = await window.pywebview.api.check_updates();
    if (r.has_update) {
      document.getElementById('updTxt').innerHTML =
        "<span style='color:#10b981;font-weight:700'>🎉 Phát hiện bản mới: " + r.latest_version + " (Hiện tại: " + r.current_version + ")</span>";
      if (r.release_notes) {
        document.getElementById('updNotes').textContent = r.release_notes;
        document.getElementById('updNotes').style.display = 'block';
      }
      dlUrl = r.download_url;
      document.getElementById('btnApply').style.display = 'inline-block';
    } else if (r.error) {
      document.getElementById('updTxt').innerHTML =
        "<span style='color:#ef4444'>⚠️ Lỗi kiểm tra: " + r.error + "</span><br><button class='btn-m btn-m-sec' style='margin-top:8px' onclick='checkUpd()'>🔄 Thử lại</button>";
    } else {
      document.getElementById('updTxt').innerHTML =
        "<span style='color:#38bdf8'>✓ Bạn đang sử dụng phiên bản mới nhất (" + r.current_version + ")</span>";
    }
  } catch (err) {
    document.getElementById('updTxt').innerHTML =
      "<span style='color:#ef4444'>⚠️ Lỗi kết nối: " + err + "</span><br><button class='btn-m btn-m-sec' style='margin-top:8px' onclick='checkUpd()'>🔄 Thử lại</button>";
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
  const counts = { all: _libItems.length, turbo: 0, triposr: 0 };
  _libItems.forEach(it => {
    const k = it.engine_key || 'triposr';
    if (counts[k] !== undefined) counts[k]++;
  });
  const cAll = document.getElementById('cntAll'); if (cAll) cAll.textContent = counts.all;
  const cTrb = document.getElementById('cntTurbo'); if (cTrb) cTrb.textContent = counts.turbo;
  const cTsr = document.getElementById('cntTripoSR'); if (cTsr) cTsr.textContent = counts.triposr;
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
  ['chipAll','chipTurbo','chipTripoSR'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active');
  });
  if (cat === 'all') { const el = document.getElementById('chipAll'); if (el) el.classList.add('active'); }
  else if (cat === 'turbo') { const el = document.getElementById('chipTurbo'); if (el) el.classList.add('active'); }
  else if (cat === 'triposr') { const el = document.getElementById('chipTripoSR'); if (el) el.classList.add('active'); }
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
    const badgeClass = (it.engine_key === 'turbo') ? 'badge-turbo' : 'badge-triposr';

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

function openMixamoModal() {
  document.getElementById('mMixamo').style.display = 'flex';
}
function closeMixamoModal() {
  document.getElementById('mMixamo').style.display = 'none';
}
function openMixamoWeb() {
  if (window.pywebview && window.pywebview.api) {
    window.pywebview.api.open_external_url('https://www.mixamo.com');
  } else {
    window.open('https://www.mixamo.com', '_blank');
  }
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
