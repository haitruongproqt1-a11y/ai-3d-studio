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

logging.basicConfig(level=logging.INFO)

APP_VERSION = "v1.4.2"
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
    try:
        logging.info(f"Loading TripoSR on {current_device}…")
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        # 8192 is the sweet spot: fast enough for inference, low VRAM
        model.renderer.set_chunk_size(8192)
        model.to(current_device)
        _model_ready.set()
        logging.info("Model loaded ✓")
    except Exception as e:
        logging.error(f"Model load failed: {e}")

def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"github_repo": DEFAULT_GITHUB_REPO, "hf_token": ""}

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"save_config: {e}")


# ── Texture helpers ─────────────────────────────────────────────────────────
def enhance_texture(img: Image.Image) -> Image.Image:
    try:
        img = ImageEnhance.Color(img).enhance(1.25)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=110, threshold=3))
    except Exception:
        pass
    return img


# ── AppApi ──────────────────────────────────────────────────────────────────
class AppApi:
    def __init__(self):
        self._window = None
        self.last_glb = ""
        self.last_obj = ""
        self.last_folder = ""
        self.config = load_config()

    def set_window(self, w):
        self._window = w

    # ── called once on startup ──────────────────────────────────────────────
    def get_init_data(self):
        return {
            "version": APP_VERSION,
            "hardware": HARDWARE_INFO,
            "model_ready": _model_ready.is_set(),
            "config": self.config,
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

    # ── model ready check ───────────────────────────────────────────────────
    def is_model_ready(self):
        return _model_ready.is_set()

    # ── live progress helper ─────────────────────────────────────────────────
    def _progress(self, msg: str, pct: int = -1):
        """Push a status message and optional progress % to the UI."""
        if self._window:
            safe = msg.replace("'", "\\'").replace("\n", " ")
            self._window.evaluate_js(f"window._setProgress('{safe}', {pct});")

    # ── image preprocessing ──────────────────────────────────────────────────
    def _preprocess(self, file_path: str) -> Image.Image:
        orig = Image.open(file_path)
        if orig.mode in ("RGBA", "LA"):
            alpha = np.array(orig)[:, :, -1]
            if (alpha < 240).any() and (alpha > 15).any():
                self._progress("🖼 Dùng kênh alpha sẵn có…", 10)
                img = resize_foreground(orig, 0.85)
                arr = np.array(img).astype(np.float32) / 255.0
                arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
                return Image.fromarray((arr * 255).astype(np.uint8))
        self._progress("🤖 AI đang tách nền…", 10)
        img = remove_background(orig.convert("RGB"))
        img = resize_foreground(img, 0.85)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        return Image.fromarray((arr * 255).astype(np.uint8))

    # ── main 3D generation entry ────────────────────────────────────────────
    def generate_3d(self, file_path, engine="local",
                    mc_resolution=None, bake_tex=False,
                    smooth=True, quality="fast"):
        if engine == "cloud":
            return self._gen_cloud(file_path, quality)
        return self._gen_local(file_path, mc_resolution, bake_tex, smooth, quality)

    # ── LOCAL GPU ────────────────────────────────────────────────────────────
    def _gen_local(self, file_path, mc_resolution, bake_tex, smooth, quality):
        global model, current_device
        if not _model_ready.is_set():
            return {"success": False, "error": "Model AI chưa sẵn sàng. Đợi thêm vài giây rồi thử lại!"}

        try:
            # ── resolution preset ──────────────────────────────────────────
            if mc_resolution is None or mc_resolution == "auto":
                if quality == "fast":
                    mc_resolution = 128
                elif quality == "hq":
                    mc_resolution = 256
                else:
                    mc_resolution = 256
            mc_resolution = int(mc_resolution)

            # ── preprocess ─────────────────────────────────────────────────
            self._progress("🖼 Đang xử lý ảnh…", 5)
            image = self._preprocess(file_path)

            ts = int(time.time())
            item_dir = os.path.join(OUTPUT_DIR, str(ts))
            os.makedirs(item_dir, exist_ok=True)
            image.save(os.path.join(item_dir, "input.png"))

            # ── TripoSR inference ──────────────────────────────────────────
            self._progress("🧠 AI đang phân tích ảnh 2D…", 20)
            with torch.no_grad():
                scene_codes = model([image], device=current_device)

            # ── mesh extraction ────────────────────────────────────────────
            self._progress("⚙️ Đang tái tạo hình khối 3D…", 45)
            # has_vertex_color=True → vertex colors from TripoSR (fast, no UV baking needed)
            # has_vertex_color=False → raw geometry only (used before bake)
            meshes = model.extract_mesh(
                scene_codes,
                has_vertex_color=(not bake_tex),
                resolution=mc_resolution,
            )

            # ── optional smoothing ─────────────────────────────────────────
            if smooth:
                self._progress("✨ Làm mịn bề mặt Taubin…", 60)
                try:
                    trimesh.smoothing.filter_taubin(meshes[0], lamb=0.5, nu=0.5, iterations=8)
                except Exception as e:
                    logging.warning(f"Smooth skipped: {e}")

            out_obj = os.path.join(item_dir, "model.obj")
            out_glb = os.path.join(item_dir, "model.glb")
            out_tex = os.path.join(item_dir, "texture.png")

            if bake_tex and HARDWARE_INFO["has_gpu"]:
                # ── BAKE TEXTURE (high-quality, slower ~30-40s at 512px) ──
                self._progress("🎨 Đang bake texture UV (512px)…", 65)
                from tsr.bake_texture import bake_texture as _bake
                import xatlas
                bake = _bake(meshes[0], model, scene_codes[0], 512)   # 512 not 1024 → 4x faster
                xatlas.export(
                    out_obj,
                    meshes[0].vertices[bake["vmapping"]],
                    bake["indices"],
                    bake["uvs"],
                    meshes[0].vertex_normals[bake["vmapping"]],
                )
                self._progress("🖌 Tăng cường màu sắc PBR…", 85)
                tex = Image.fromarray((bake["colors"] * 255).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)
                tex = enhance_texture(tex)
                tex.save(out_tex)
                loaded = trimesh.load(out_obj)
                mat = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=tex, roughnessFactor=0.35, metallicFactor=0.05
                )
                loaded.visual.material = mat
                loaded.export(out_glb)
                engine_label = "GPU RTX 3050 + Bake UV 512px PBR"
            else:
                # ── FAST PATH: vertex colors (no UV baking needed) ────────
                self._progress("💾 Đang xuất GLB…", 80)
                meshes[0].export(out_glb)
                meshes[0].export(out_obj)
                engine_label = "GPU RTX 3050 (Vertex Colors – Siêu Nhanh)"

            self.last_glb = out_glb
            self.last_obj = out_obj
            self.last_folder = item_dir

            self._progress("✅ Hoàn tất! Đang tải mô hình 3D lên…", 95)
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

    # ── CLOUD ────────────────────────────────────────────────────────────────
    def _gen_cloud(self, file_path, quality="fast"):
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            return {"success": False, "error": "Thiếu thư viện gradio_client. Vui lòng dùng chế độ GPU Offline!"}

        try:
            hf_token = self.config.get("hf_token", "").strip() or None
            self._progress("☁️ Đang kết nối Cloud…", 5)
            client = Client("TencentARC/InstantMesh", token=hf_token)

            self._progress("🖼 Đang tải ảnh lên Cloud và tách nền…", 15)
            prep = client.predict(input_image=handle_file(file_path), api_name="/preprocess")

            steps = 50 if quality == "hq" else 30   # InstantMesh min=30
            self._progress(f"🔮 Cloud đang tạo đa góc nhìn ({steps} bước)…", 35)
            client.predict(input_image=handle_file(prep), sample_steps=steps,
                           sample_seed=42, api_name="/generate_mvs")

            self._progress("⚙️ Cloud đang ghép Mesh 3D…", 70)
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

            self._progress("✅ Hoàn tất! Đang tải mô hình lên…", 95)
            with open(local_glb, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{b64}",
                "glb_path": local_glb, "obj_path": local_obj, "folder": item_dir,
                "engine_used": f"Cloud Multi-View ({steps} Steps)",
            }
        except Exception as e:
            logging.exception("_gen_cloud error")
            msg = str(e)
            if "quota" in msg.lower() or "ZeroGPU" in msg:
                msg = ("⚠️ Cloud hết quota ZeroGPU miễn phí tạm thời.\n"
                       "👉 Chuyển sang GPU Offline (RTX 3050) để tạo ngay không giới hạn, "
                       "hoặc thêm HF Token miễn phí trong Cài đặt!")
            return {"success": False, "error": msg}

    # ── settings ─────────────────────────────────────────────────────────────
    def save_settings(self, hf_token=None):
        if hf_token is not None:
            self.config["hf_token"] = hf_token.strip()
        save_config(self.config)
        return {"success": True}

    # ── file ops ──────────────────────────────────────────────────────────────
    def open_folder(self, folder=None):
        target = folder or self.last_folder or OUTPUT_DIR
        if os.path.exists(target):
            os.system(f'explorer.exe "{target}"')
        return True

    def export_file(self, file_type="glb"):
        src = self.last_glb if file_type == "glb" else self.last_obj
        if not src or not os.path.exists(src):
            return {"success": False, "error": "Chưa có file nào được tạo!"}
        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=f"model_{int(time.time())}.{file_type}",
            file_types=[f"3D Model (*.{file_type})"],
        )
        if result:
            dst = result if isinstance(result, str) else result[0]
            shutil.copy2(src, dst)
            return {"success": True, "saved_path": dst}
        return {"success": False, "canceled": True}

    # ── OTA update ────────────────────────────────────────────────────────────
    def check_updates(self):
        repo = self.config.get("github_repo", DEFAULT_GITHUB_REPO)
        for branch in ("main", "master"):
            try:
                url = f"https://raw.githubusercontent.com/{repo}/{branch}/version.json"
                req = urllib.request.Request(url, headers={"User-Agent": "AI-3D-Studio"})
                with urllib.request.urlopen(req, timeout=6) as r:
                    if r.status == 200:
                        data = json.loads(r.read())
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
            SKIP = {".venv", "output_app", "config.json", "_update.zip", "_update_ex"}
            for name in os.listdir(src):
                if name in SKIP:
                    continue
                s = os.path.join(src, name)
                d = os.path.join(APP_DIR, name)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
                elif os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.copytree(s, d)
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


# ── HTML ─────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>AI 3D Studio</title>
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;user-select:none}
body{background:#0b0d13;color:#e2e8f0;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Header ── */
header{height:52px;background:#131722;border-bottom:1px solid #232936;display:flex;align-items:center;justify-content:space-between;padding:0 18px;flex-shrink:0}
.logo{display:flex;align-items:center;gap:8px;font-weight:700;font-size:16px;color:#60a5fa}
.logo-chip{background:linear-gradient(135deg,#2563eb,#7c3aed);padding:3px 8px;border-radius:5px;color:#fff;font-size:10px;font-weight:800}
.ver{font-size:11px;color:#64748b;background:#1a2030;padding:2px 7px;border-radius:4px;border:1px solid #2d3748}
.hdr-right{display:flex;align-items:center;gap:8px}
.gpu-pill{font-size:11px;padding:5px 12px;border-radius:20px;display:flex;align-items:center;gap:6px;font-weight:600;background:rgba(16,185,129,.12);color:#34d399;border:1px solid rgba(16,185,129,.25)}
.dot{width:7px;height:7px;border-radius:50%;background:#10b981;box-shadow:0 0 6px #10b981;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.btn-hdr{background:#1a2030;border:1px solid #2d3748;color:#94a3b8;padding:5px 11px;border-radius:7px;font-size:11px;font-weight:600;cursor:pointer;transition:.15s}
.btn-hdr:hover{background:#232936;color:#e2e8f0;border-color:#3b82f6}

/* ── Layout ── */
.layout{flex:1;display:flex;min-height:0}

/* ── Sidebar ── */
.sidebar{width:360px;background:#0f1117;border-right:1px solid #1e2433;padding:14px;display:flex;flex-direction:column;gap:11px;overflow-y:auto;flex-shrink:0}

/* ── Mode tabs ── */
.tabs{display:flex;background:#080a10;padding:3px;border-radius:8px;border:1px solid #1e2433;gap:3px}
.tab{flex:1;padding:7px 4px;border:none;border-radius:6px;background:transparent;color:#64748b;font-size:11px;font-weight:700;cursor:pointer;transition:.15s;text-align:center}
.tab.on{background:linear-gradient(135deg,#1d4ed8,#6d28d9);color:#fff;box-shadow:0 2px 8px rgba(37,99,235,.35)}
.tab.cloud.on{background:linear-gradient(135deg,#059669,#2563eb);box-shadow:0 2px 8px rgba(16,185,129,.35)}

/* ── Drop zone ── */
.drop{border:2px dashed #2d3748;border-radius:10px;padding:14px;text-align:center;cursor:pointer;background:#111622;min-height:160px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:7px;transition:.2s}
.drop:hover{border-color:#3b82f6;background:#161f30}
.drop img{max-width:100%;max-height:140px;object-fit:contain;border-radius:7px;display:none}
.drop-hint b{color:#60a5fa;display:block;font-size:13px;margin-bottom:2px}
.drop-hint p{color:#475569;font-size:11px}

/* ── Info card ── */
.info-card{background:#111622;border:1px solid #1e2433;border-radius:8px;padding:9px 11px;font-size:11px;display:flex;flex-direction:column;gap:3px}
.ic-title{color:#f1f5f9;font-weight:600;font-size:12px}
.ic-desc{color:#38bdf8;line-height:1.4}
.ic-desc b{color:#34d399}

/* ── Settings group ── */
.sg{display:flex;flex-direction:column;gap:4px}
.sg label{font-size:11px;font-weight:600;color:#64748b}
select,input[type=text]{background:#111622;border:1px solid #2d3748;color:#e2e8f0;padding:7px 10px;border-radius:7px;font-size:12px;outline:none;width:100%}
.chk-row{display:flex;align-items:center;gap:7px;font-size:11px;color:#94a3b8;cursor:pointer}
.chk-row input{cursor:pointer}

/* ── Generate button ── */
.btn-gen{background:linear-gradient(135deg,#1d4ed8,#6d28d9);color:#fff;border:none;padding:13px;border-radius:9px;font-size:13px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:7px;box-shadow:0 4px 14px rgba(37,99,235,.3);transition:.2s}
.btn-gen.cloud-mode{background:linear-gradient(135deg,#059669,#2563eb);box-shadow:0 4px 14px rgba(16,185,129,.3)}
.btn-gen:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 6px 18px rgba(37,99,235,.4)}
.btn-gen:disabled{background:#1a2030;color:#475569;cursor:not-allowed;box-shadow:none;transform:none}

/* ── Progress block ── */
.prog-block{background:#111622;border:1px solid #1e2433;border-radius:8px;padding:10px 12px;display:flex;flex-direction:column;gap:6px;min-height:56px}
.prog-text{font-size:12px;color:#94a3b8;display:flex;align-items:center;gap:7px;min-height:20px}
.prog-bar-wrap{height:4px;background:#1e2433;border-radius:2px;overflow:hidden;display:none}
.prog-bar{height:100%;width:0%;background:linear-gradient(90deg,#3b82f6,#7c3aed);border-radius:2px;transition:width .4s ease}
.spin{width:13px;height:13px;border:2px solid #2d3748;border-top-color:#38bdf8;border-radius:50%;animation:sp .7s linear infinite;display:none;flex-shrink:0}
@keyframes sp{to{transform:rotate(360deg)}}

/* ── Viewport ── */
.vp{flex:1;background:radial-gradient(circle at 50% 50%,#161f30 0%,#070910 100%);position:relative;display:flex;align-items:center;justify-content:center;min-width:0}
model-viewer{width:100%;height:100%;--poster-color:transparent}
.vp-empty{position:absolute;display:flex;flex-direction:column;align-items:center;gap:10px;color:#334155;pointer-events:none}
.vp-empty svg{width:56px;height:56px;opacity:.4}

/* ── Viewport overlays ── */
.light-bar{position:absolute;top:14px;left:14px;display:flex;align-items:center;gap:7px;background:rgba(11,13,19,.8);backdrop-filter:blur(8px);padding:5px 11px;border-radius:7px;border:1px solid rgba(255,255,255,.08);font-size:11px}
.light-bar select{padding:2px 7px;font-size:11px;width:auto}
.act-bar{position:absolute;top:14px;right:14px;display:flex;gap:7px}
.btn-act{background:rgba(11,13,19,.8);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.1);color:#cbd5e1;padding:6px 11px;border-radius:7px;font-size:11px;font-weight:600;cursor:pointer;transition:.15s;display:flex;align-items:center;gap:4px}
.btn-act:hover:not(:disabled){background:#1e2433;border-color:#38bdf8}
.btn-act:disabled{opacity:.3;cursor:not-allowed}
.hint{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);background:rgba(11,13,19,.8);backdrop-filter:blur(8px);padding:6px 18px;border-radius:16px;font-size:11px;color:#64748b;pointer-events:none;border:1px solid rgba(255,255,255,.07)}

/* ── Modal ── */
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.7);backdrop-filter:blur(5px);display:none;align-items:center;justify-content:center;z-index:99}
.modal{background:#111622;border:1px solid #2d3748;border-radius:12px;width:480px;max-width:90vw;padding:22px;display:flex;flex-direction:column;gap:14px;box-shadow:0 20px 40px rgba(0,0,0,.5)}
.m-hdr{display:flex;justify-content:space-between;align-items:center}
.m-title{font-size:15px;font-weight:700;color:#60a5fa}
.m-x{background:none;border:none;color:#64748b;font-size:18px;cursor:pointer}
.notes{background:#080a10;border:1px solid #1e2433;border-radius:7px;padding:10px;font-size:11px;color:#94a3b8;max-height:160px;overflow-y:auto;white-space:pre-wrap;line-height:1.5;display:none}
.m-foot{display:flex;justify-content:flex-end;gap:8px}
.btn-m{padding:7px 14px;border-radius:7px;font-size:12px;font-weight:600;cursor:pointer;border:none}
.btn-m-pri{background:#2563eb;color:#fff}
.btn-m-sec{background:#1e2433;color:#94a3b8}

/* ── Model-not-ready banner ── */
.banner{background:rgba(37,99,235,.12);border:1px solid rgba(37,99,235,.3);border-radius:8px;padding:8px 12px;font-size:11px;color:#93c5fd;display:none;align-items:center;gap:7px}
</style>
</head>
<body>

<header>
  <div style="display:flex;align-items:center;gap:9px">
    <div class="logo"><span class="logo-chip">3D AI</span>AI 3D Studio</div>
    <span class="ver" id="ver">v1.4.1</span>
  </div>
  <div class="hdr-right">
    <div class="gpu-pill"><div class="dot"></div><span id="gpuTxt">Đang phát hiện…</span></div>
    <button class="btn-hdr" onclick="openSettings()">⚙️ Cài đặt</button>
    <button class="btn-hdr" onclick="openUpdate()">🔄 Cập nhật</button>
  </div>
</header>

<div class="layout">
  <div class="sidebar">
    <!-- Mode tabs -->
    <div class="tabs">
      <button class="tab on" id="tabLocal" onclick="setMode('local')">⚡ GPU Offline (RTX 3050)</button>
      <button class="tab cloud" id="tabCloud" onclick="setMode('cloud')">🌟 Cloud Multi-View</button>
    </div>

    <!-- Not-ready banner -->
    <div class="banner" id="banner">
      <span>⏳</span>
      <span id="bannerTxt">Model AI đang nạp vào GPU, thường mất ~10 giây. Bạn có thể chọn ảnh trước!</span>
    </div>

    <!-- Drop zone -->
    <div class="drop" id="drop" onclick="pickImage()">
      <img id="prev" alt="preview">
      <div class="drop-hint" id="dropHint">
        <b>Chọn ảnh 2D</b>
        <p>Nhấn để nạp ảnh PNG / JPG / WebP từ máy</p>
      </div>
    </div>

    <!-- Info card -->
    <div class="info-card">
      <span class="ic-title" id="icTitle">⚡ GPU Cục bộ – NVIDIA RTX 3050 (Offline)</span>
      <span class="ic-desc" id="icDesc">Xử lý ngay trên card đồ họa, <b>siêu nhanh</b>, không cần mạng, không giới hạn số lần tạo.</span>
    </div>

    <!-- Local settings -->
    <div id="localSet">
      <div class="sg" style="margin-bottom:8px">
        <label>Chất lượng & Tốc độ</label>
        <select id="quality">
          <option value="fast" selected>⚡ Nhanh (~5–8 giây) – Màu đỉnh đỉnh</option>
          <option value="hq">🔬 Cao (~12–18 giây) – Chi tiết sắc nét hơn</option>
        </select>
      </div>
      <div class="chk-row" style="margin-bottom:6px">
        <input type="checkbox" id="chkSmooth" checked>
        <label class="chk-row" for="chkSmooth">✨ Làm mịn bề mặt Taubin (khử sần sùi)</label>
      </div>
      <div class="chk-row" style="margin-bottom:8px">
        <input type="checkbox" id="chkBake">
        <label class="chk-row" for="chkBake">🎨 Bake Texture UV 512px PBR (+~30 giây)</label>
      </div>
      <p style="font-size:10px;color:#475569;line-height:1.4">
        ✓ AI tách nền tự động bảo toàn 100% chi tiết màu sắc.<br>
        ✓ Mặc định dùng Vertex Colors – nhanh &amp; đẹp cho Unity Game.
      </p>
    </div>

    <!-- Cloud settings -->
    <div id="cloudSet" style="display:none">
      <div class="sg" style="margin-bottom:7px">
        <label>Chất lượng Cloud</label>
        <select id="cloudQ">
          <option value="fast" selected>⚡ Nhanh (~20 giây) – Tiết kiệm quota</option>
          <option value="hq">🔬 Cao (~35 giây) – Chi tiết hơn</option>
        </select>
      </div>
      <p style="font-size:10px;color:#475569;line-height:1.4">
        ✓ Số bước tính toán được tối ưu để không vượt giới hạn ZeroGPU miễn phí.<br>
        ✓ Thêm HF Token trong Cài đặt nếu thường xuyên bị hết quota.
      </p>
    </div>

    <button class="btn-gen" id="btnGen" onclick="generate()" disabled>
      ⚡ BẮT ĐẦU TẠO 3D (RTX 3050)
    </button>

    <!-- Progress block -->
    <div class="prog-block">
      <div class="prog-text">
        <div class="spin" id="spin"></div>
        <span id="progTxt">Nạp ảnh để bắt đầu.</span>
      </div>
      <div class="prog-bar-wrap" id="barWrap">
        <div class="prog-bar" id="bar"></div>
      </div>
    </div>
  </div>

  <!-- Viewport -->
  <div class="vp">
    <div class="light-bar">
      <span>💡</span>
      <select id="lightSel" onchange="changeLight()">
        <option value="studio" selected>Studio ACES</option>
        <option value="aces">Điện ảnh</option>
        <option value="soft">Tự nhiên</option>
      </select>
    </div>

    <div class="act-bar">
      <button class="btn-act" id="btnGlb" onclick="doExport('glb')" disabled>📦 Lưu GLB (Unity)</button>
      <button class="btn-act" id="btnObj" onclick="doExport('obj')" disabled>📦 Lưu OBJ (Blender)</button>
      <button class="btn-act" id="btnDir" onclick="openDir()" disabled>📂 Mở thư mục</button>
    </div>

    <div class="vp-empty" id="empty">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
        <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
        <line x1="12" y1="22.08" x2="12" y2="12"/>
      </svg>
      <p style="font-size:13px">Mô hình 3D xoay 360° sẽ hiện ở đây</p>
    </div>

    <model-viewer id="mv" camera-controls auto-rotate auto-rotate-delay="800"
      rotation-per-second="20deg" tone-mapping="aces" shadow-intensity="1.8"
      shadow-softness="0.6" exposure="1.2" style="display:none"></model-viewer>

    <div class="hint" id="hint" style="display:none">
      Chuột trái: Xoay &nbsp;|&nbsp; Con lăn: Zoom &nbsp;|&nbsp; Chuột phải: Di chuyển
    </div>
  </div>
</div>

<!-- Settings modal -->
<div class="modal-bg" id="mSet">
  <div class="modal">
    <div class="m-hdr">
      <span class="m-title">⚙️ Cài đặt</span>
      <button class="m-x" onclick="closeSettings()">&times;</button>
    </div>
    <div class="sg">
      <label>Hugging Face Token (tuỳ chọn – miễn phí):</label>
      <input type="text" id="hfTok" placeholder="hf_xxxxxxxxxxxxxxxx" style="font-family:monospace">
      <p style="font-size:10px;color:#64748b;margin-top:3px;line-height:1.4">
        Đăng ký miễn phí tại <b style="color:#60a5fa">huggingface.co/settings/tokens</b> để nhận thêm quota Cloud. Không cần điền nếu dùng GPU Offline.
      </p>
    </div>
    <div class="m-foot">
      <button class="btn-m btn-m-sec" onclick="closeSettings()">Hủy</button>
      <button class="btn-m btn-m-pri" onclick="saveSettings()">💾 Lưu</button>
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

<script>
/* ── state ── */
let imgPath = null, lastFolder = null, dlUrl = null, curMode = 'local';
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

/* ── init ── */
window.addEventListener('pywebviewready', async () => {
  const d = await window.pywebview.api.get_init_data();
  document.getElementById('ver').textContent = d.version;
  document.getElementById('gpuTxt').textContent = d.hardware.has_gpu
    ? 'GPU: ' + d.hardware.name + ' (' + d.hardware.vram_gb + ' GB)'
    : d.hardware.status_text;

  if (d.config && d.config.hf_token)
    document.getElementById('hfTok').value = d.config.hf_token;

  _modelReady = d.model_ready;
  if (!_modelReady) {
    document.getElementById('banner').style.display = 'flex';
    pollModel();
  }
});

async function pollModel() {
  const ready = await window.pywebview.api.is_model_ready();
  if (ready) {
    _modelReady = true;
    document.getElementById('banner').style.display = 'none';
    if (imgPath) document.getElementById('btnGen').disabled = false;
  } else {
    setTimeout(pollModel, 1500);
  }
}

/* ── mode switcher ── */
function setMode(m) {
  curMode = m;
  const isLocal = m === 'local';
  document.getElementById('tabLocal').className = 'tab' + (isLocal ? ' on' : '');
  document.getElementById('tabCloud').className = 'tab cloud' + (!isLocal ? ' on' : '');
  document.getElementById('localSet').style.display = isLocal ? '' : 'none';
  document.getElementById('cloudSet').style.display = isLocal ? 'none' : '';
  if (isLocal) {
    document.getElementById('icTitle').textContent = '⚡ GPU Cục bộ – NVIDIA RTX 3050 (Offline)';
    document.getElementById('icDesc').innerHTML = 'Xử lý ngay trên card đồ họa, <b>siêu nhanh</b>, không cần mạng, không giới hạn số lần tạo.';
    document.getElementById('btnGen').className = 'btn-gen';
    document.getElementById('btnGen').textContent = '⚡ BẮT ĐẦU TẠO 3D (RTX 3050)';
  } else {
    document.getElementById('icTitle').textContent = '🌟 Cloud Multi-View (Miễn phí)';
    document.getElementById('icDesc').innerHTML = 'Dùng GPU Cloud miễn phí của Hugging Face tái tạo <b>nhiều góc nhìn</b> cho độ chi tiết cao.';
    document.getElementById('btnGen').className = 'btn-gen cloud-mode';
    document.getElementById('btnGen').textContent = '🌟 BẮT ĐẦU TẠO 3D TRÊN CLOUD';
  }
}

/* ── image pick ── */
async function pickImage() {
  window._setProgress('Đang mở hộp thoại…', -1);
  const r = await window.pywebview.api.select_image();
  if (r && r.path) {
    imgPath = r.path;
    document.getElementById('prev').src = r.dataUrl;
    document.getElementById('prev').style.display = 'block';
    document.getElementById('dropHint').style.display = 'none';
    if (_modelReady || curMode === 'cloud')
      document.getElementById('btnGen').disabled = false;
    window._setProgress('Đã nạp: ' + r.name, -1);
  } else {
    window._setProgress('Chưa chọn ảnh.', -1);
  }
}

/* ── generate ── */
async function generate() {
  if (!imgPath) return;
  const quality = curMode === 'local'
    ? document.getElementById('quality').value
    : document.getElementById('cloudQ').value;
  const smooth = document.getElementById('chkSmooth').checked;
  const bake   = document.getElementById('chkBake').checked;

  document.getElementById('btnGen').disabled = true;
  document.getElementById('drop').style.pointerEvents = 'none';
  document.getElementById('spin').style.display = 'inline-block';
  document.getElementById('barWrap').style.display = 'block';
  document.getElementById('bar').style.width = '0%';
  window._setProgress('Đang khởi động…', 2);

  try {
    const r = await window.pywebview.api.generate_3d(
      imgPath, curMode, 'auto', bake, smooth, quality
    );
    if (r.success) {
      lastFolder = r.folder;
      const mv = document.getElementById('mv');
      mv.src = r.glb_data;
      mv.style.display = 'block';
      document.getElementById('empty').style.display = 'none';
      document.getElementById('hint').style.display = 'block';
      ['btnGlb','btnObj','btnDir'].forEach(id => document.getElementById(id).disabled = false);
      document.getElementById('barWrap').style.display = 'block';
      document.getElementById('bar').style.width = '100%';
      window._setProgress('✅ Thành công! (' + r.engine_used + ') – Chuột trái để xoay.', 100);
    } else {
      window._setProgress('❌ ' + r.error, -1);
      document.getElementById('bar').style.width = '0%';
    }
  } catch(e) {
    window._setProgress('❌ Lỗi ngoại lệ: ' + e, -1);
  } finally {
    document.getElementById('btnGen').disabled = false;
    document.getElementById('drop').style.pointerEvents = 'auto';
    document.getElementById('spin').style.display = 'none';
  }
}

/* ── lighting ── */
function changeLight() {
  const mv = document.getElementById('mv');
  const v = document.getElementById('lightSel').value;
  const cfg = {studio:['aces','1.3','1.8'],aces:['aces','1.0','2.2'],soft:['neutral','1.1','1.2']};
  const [tm,ex,sh] = cfg[v];
  mv.setAttribute('tone-mapping',tm);
  mv.setAttribute('exposure',ex);
  mv.setAttribute('shadow-intensity',sh);
}

/* ── export / folder ── */
async function doExport(t) {
  window._setProgress('Đang lưu file ' + t.toUpperCase() + '…', -1);
  const r = await window.pywebview.api.export_file(t);
  if (r.success) window._setProgress('✅ Đã lưu: ' + r.saved_path, -1);
  else if (!r.canceled) window._setProgress('❌ ' + r.error, -1);
}
async function openDir() { await window.pywebview.api.open_folder(lastFolder); }

/* ── settings ── */
function openSettings() { document.getElementById('mSet').style.display='flex'; }
function closeSettings() { document.getElementById('mSet').style.display='none'; }
async function saveSettings() {
  await window.pywebview.api.save_settings(document.getElementById('hfTok').value);
  closeSettings();
  window._setProgress('✅ Đã lưu cài đặt.', -1);
}

/* ── update ── */
function openUpdate() { document.getElementById('mUpd').style.display='flex'; }
function closeUpdate() { document.getElementById('mUpd').style.display='none'; }
async function checkUpd() {
  document.getElementById('updTxt').textContent = 'Đang kiểm tra…';
  document.getElementById('btnApply').style.display='none';
  document.getElementById('updNotes').style.display='none';
  const r = await window.pywebview.api.check_updates();
  if (r.has_update) {
    document.getElementById('updTxt').innerHTML =
      "<span style='color:#10b981;font-weight:700'>🎉 Bản mới: "+r.latest_version+" (hiện tại: "+r.current_version+")</span>";
    if (r.release_notes) {
      document.getElementById('updNotes').textContent = r.release_notes;
      document.getElementById('updNotes').style.display='block';
    }
    dlUrl = r.download_url;
    document.getElementById('btnApply').style.display='inline-block';
  } else {
    document.getElementById('updTxt').innerHTML =
      "<span style='color:#38bdf8'>✓ Đang dùng phiên bản mới nhất ("+r.current_version+")</span>";
  }
}
async function applyUpd() {
  if (!dlUrl) return;
  document.getElementById('updTxt').textContent = 'Đang tải và cài đặt…';
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
</script>
</body>
</html>"""


def main():
    # Load model in background so the UI is immediately responsive
    threading.Thread(target=load_ai_model, daemon=True).start()

    api = AppApi()
    window = webview.create_window(
        title=f"AI 3D Studio {APP_VERSION}",
        html=HTML,
        js_api=api,
        width=1200,
        height=800,
        min_size=(920, 620),
    )
    api.set_window(window)
    webview.start(debug=False)


if __name__ == "__main__":
    main()
