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
from PIL import Image
import torch
import webview

# Add TripoSR path
sys.path.append(os.path.join(os.path.dirname(__file__), "TripoSR"))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
from tsr.bake_texture import bake_texture
import trimesh
import cv2

try:
    from gradio_client import Client, handle_file
except ImportError:
    Client = None

logging.basicConfig(level=logging.INFO)

APP_VERSION = "v1.2.0"
DEFAULT_GITHUB_REPO = "haitruongproqt1-a11y/ai-3d-studio"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "output_app")
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardware Detection
def detect_hardware():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            total_vram_gb = round(total_vram_bytes / (1024 ** 3), 1)
            preset = "gpu_high" if total_vram_gb >= 5.5 else "gpu_low"
            return {
                "has_gpu": True,
                "device": "cuda:0",
                "name": gpu_name,
                "vram_gb": total_vram_gb,
                "preset": preset,
                "status_text": f"GPU: NVIDIA {gpu_name} ({total_vram_gb} GB)",
                "recommendation": "GPU Sẵn sàng - Hỗ trợ làm mịn bề mặt & Texture Atlas"
            }
        except Exception:
            pass

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    return {
        "has_gpu": False,
        "device": "cpu",
        "name": f"CPU ({cpu_count} luồng)",
        "vram_gb": 0,
        "preset": "cpu_fallback",
        "status_text": f"Chế độ CPU ({cpu_count} luồng) - Máy yếu",
        "recommendation": "Tự động kích hoạt cấu hình siêu nhẹ tránh đơ máy"
    }

HARDWARE_INFO = detect_hardware()
current_device = HARDWARE_INFO["device"]
model = None

def load_ai_model():
    global model
    try:
        logging.info(f"Loading TripoSR model on {current_device}...")
        model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
        chunk_size = 8192 if (current_device.startswith("cuda") and HARDWARE_INFO.get("vram_gb", 0) >= 5.5) else (2048 if current_device.startswith("cuda") else 1024)
        model.renderer.set_chunk_size(chunk_size)
        model.to(current_device)
        logging.info("Model loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading model: {e}")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "github_repo": DEFAULT_GITHUB_REPO,
        "hf_token": ""
    }

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving config: {e}")


class AppApi:
    def __init__(self):
        self._window = None
        self.last_glb_path = ""
        self.last_obj_path = ""
        self.config = load_config()

    def set_window(self, window):
        self._window = window

    def get_init_data(self):
        return {
            "version": APP_VERSION,
            "hardware": HARDWARE_INFO,
            "config": self.config
        }

    def save_settings(self, hf_token, github_repo):
        self.config["hf_token"] = (hf_token or "").strip()
        if github_repo:
            self.config["github_repo"] = github_repo.strip()
        save_config(self.config)
        return {"success": True, "message": "Đã lưu cài đặt!"}

    def select_image(self):
        if not self._window:
            return None
        file_types = ('Image Files (*.png;*.jpg;*.jpeg;*.webp)', 'All Files (*.*)')
        result = self._window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types)
        if result and len(result) > 0:
            file_path = result[0]
            try:
                with open(file_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(file_path)[1].lower().replace(".", "")
                if ext == "jpg":
                    ext = "jpeg"
                data_url = f"data:image/{ext};base64,{b64}"
                return {"path": file_path, "dataUrl": data_url, "name": os.path.basename(file_path)}
            except Exception as e:
                return {"error": str(e)}
        return None

    def clean_checkerboard_mask(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return Image.open(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        chair_mask = (s > 28) & (v > 25)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        chair_mask = cv2.morphologyEx(chair_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        mask_ratio = np.sum(chair_mask > 0) / (chair_mask.shape[0] * chair_mask.shape[1])
        if 0.04 < mask_ratio < 0.92:
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = chair_mask
            return Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))
        else:
            return Image.open(img_path)

    # ======== GENERATE 3D (LOCAL TAUBIN SMOOTH + HUGGING FACE FREE CLOUD) ========
    def generate_3d(self, file_path, engine="local", mc_resolution=None, bake_texture_flag=None, smooth_mesh=True):
        if engine == "cloud_free":
            return self._generate_cloud_free(file_path)
        else:
            return self._generate_local(file_path, mc_resolution, bake_texture_flag, smooth_mesh)

    def _generate_local(self, file_path, mc_resolution, bake_texture_flag, smooth_mesh):
        global model, current_device
        if not model:
            return {"success": False, "error": "AI Model đang nạp, vui lòng bấm lại sau 5 giây!"}
        
        try:
            if mc_resolution is None or mc_resolution == "auto":
                mc_resolution = 256 if HARDWARE_INFO["has_gpu"] else 128

            if bake_texture_flag is None:
                bake_texture_flag = True if HARDWARE_INFO["has_gpu"] else False

            cleaned_img = self.clean_checkerboard_mask(file_path)
            
            if cleaned_img.mode == "RGBA":
                raw_np = np.array(cleaned_img).astype(np.float32) / 255.0
                image_np = raw_np[:, :, :3] * raw_np[:, :, 3:4] + (1 - raw_np[:, :, 3:4]) * 0.5
                image = Image.fromarray((image_np * 255.0).astype(np.uint8))
            else:
                image = remove_background(cleaned_img)
                image = resize_foreground(image, 0.85)
                image_np = np.array(image).astype(np.float32) / 255.0
                image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
                image = Image.fromarray((image_np * 255.0).astype(np.uint8))

            timestamp = int(time.time())
            item_dir = os.path.join(OUTPUT_DIR, str(timestamp))
            os.makedirs(item_dir, exist_ok=True)
            image.save(os.path.join(item_dir, "input.png"))

            with torch.no_grad():
                scene_codes = model([image], device=current_device)

            meshes = model.extract_mesh(scene_codes, not bake_texture_flag, resolution=int(mc_resolution))
            
            # Surface Taubin Smoothing to make banana and organic shapes smooth & realistic
            if smooth_mesh:
                try:
                    trimesh.smoothing.filter_taubin(meshes[0], lamb=0.5, nu=-0.53, iterations=10)
                except Exception as e:
                    logging.warning(f"Smoothing skipped: {e}")

            out_obj_path = os.path.join(item_dir, "model.obj")
            out_glb_path = os.path.join(item_dir, "model.glb")
            out_tex_path = os.path.join(item_dir, "texture.png")

            if bake_texture_flag and HARDWARE_INFO["has_gpu"]:
                import xatlas
                bake_output = bake_texture(meshes[0], model, scene_codes[0], 1024)
                xatlas.export(out_obj_path, meshes[0].vertices[bake_output["vmapping"]], bake_output["indices"], bake_output["uvs"], meshes[0].vertex_normals[bake_output["vmapping"]])
                tex_img = Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)
                tex_img.save(out_tex_path)
                
                loaded = trimesh.load(out_obj_path)
                loaded.visual = trimesh.visual.TextureVisuals(image=tex_img, uv=loaded.visual.uv)
                loaded.export(out_glb_path)
            else:
                meshes[0].export(out_glb_path)
                meshes[0].export(out_obj_path)

            self.last_glb_path = out_glb_path
            self.last_obj_path = out_obj_path

            with open(out_glb_path, "rb") as f:
                glb_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{glb_b64}",
                "glb_path": out_glb_path,
                "obj_path": out_obj_path,
                "folder": item_dir,
                "engine_used": f"Local GPU (RTX 3050 + Làm Mịn Taubin)"
            }
        except Exception as e:
            logging.exception("Error in _generate_local")
            return {"success": False, "error": str(e)}

    def _generate_cloud_free(self, file_path):
        """Generates 3D using 100% FREE InstantMesh / Hunyuan3D on Hugging Face Spaces"""
        if Client is None:
            return {"success": False, "error": "Chưa cài đặt gradio_client!"}
        
        try:
            hf_token = self.config.get("hf_token", "").strip() or None
            logging.info("Connecting to Free Hugging Face InstantMesh Space...")
            client = Client("TencentARC/InstantMesh", hf_token=hf_token)
            
            # Step 1: Preprocess image
            prep_res = client.predict(
                input_image=handle_file(file_path),
                api_name="/preprocess"
            )
            
            # Step 2: Generate Multiview
            logging.info("Generating multi-view representation...")
            client.predict(
                input_image=handle_file(prep_res),
                sample_steps=40,
                sample_seed=42,
                api_name="/generate_mvs"
            )
            
            # Step 3: Make 3D Mesh
            logging.info("Reconstructing 3D textured mesh...")
            obj_path, glb_path = client.predict(api_name="/make3d")
            
            timestamp = int(time.time())
            item_dir = os.path.join(OUTPUT_DIR, f"hf_free_{timestamp}")
            os.makedirs(item_dir, exist_ok=True)
            
            local_glb = os.path.join(item_dir, "model.glb")
            local_obj = os.path.join(item_dir, "model.obj")
            shutil.copy2(glb_path, local_glb)
            shutil.copy2(obj_path, local_obj)
            
            self.last_glb_path = local_glb
            self.last_obj_path = local_obj
            
            with open(local_glb, "rb") as f:
                glb_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{glb_b64}",
                "glb_path": local_glb,
                "obj_path": local_obj,
                "folder": item_dir,
                "engine_used": "Cloud Miễn Phí (Hugging Face InstantMesh 3D)"
            }
        except Exception as e:
            logging.exception("Error in _generate_cloud_free")
            return {"success": False, "error": f"Lỗi Cloud Miễn Phí: {str(e)}"}

    def open_folder(self, folder_path=None):
        target = folder_path if folder_path else OUTPUT_DIR
        if os.path.exists(target):
            os.system(f'explorer.exe "{target}"')
            return True
        return False

    def export_file(self, file_type="glb"):
        src = self.last_glb_path if file_type == "glb" else self.last_obj_path
        if not src or not os.path.exists(src):
            return {"success": False, "error": "Chưa có file nào được tạo!"}
        
        ext = f"*.{file_type}"
        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=f"model_{int(time.time())}.{file_type}",
            file_types=[f"3D Model ({ext})"]
        )
        if result:
            dest = result if isinstance(result, str) else result[0]
            shutil.copy2(src, dest)
            return {"success": True, "saved_path": dest}
        return {"success": False, "canceled": True}

    # ================= GITHUB AUTO-UPDATE SYSTEM =================
    def check_updates(self):
        repo = self.config.get("github_repo", DEFAULT_GITHUB_REPO)
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        req = urllib.request.Request(api_url, headers={"User-Agent": "AI-3D-Studio-Updater"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latest_tag = data.get("tag_name", "").strip()
                body = data.get("body", "Không có ghi chú phiên bản.")
                
                zip_url = ""
                for asset in data.get("assets", []):
                    if asset.get("name", "").endswith(".zip"):
                        zip_url = asset.get("browser_download_url", "")
                        break

                has_update = (latest_tag != APP_VERSION and latest_tag != "")
                return {
                    "success": True,
                    "current_version": APP_VERSION,
                    "latest_version": latest_tag if latest_tag else APP_VERSION,
                    "has_update": has_update,
                    "release_notes": body,
                    "download_url": zip_url,
                    "repo": repo
                }
        except urllib.error.HTTPError as e:
            return {"success": False, "error": f"Lỗi HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"success": False, "error": f"Không thể kết nối GitHub: {str(e)}"}

    def apply_update(self, download_url):
        if not download_url:
            return {"success": False, "error": "Đường dẫn tải bản cập nhật không hợp lệ!"}
        try:
            temp_zip = os.path.join(APP_DIR, "update_temp.zip")
            extract_dir = os.path.join(APP_DIR, "update_extracted")
            
            req = urllib.request.Request(download_url, headers={"User-Agent": "AI-3D-Studio-Updater"})
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_zip, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            subitems = os.listdir(extract_dir)
            src_root = os.path.join(extract_dir, subitems[0]) if len(subitems) == 1 and os.path.isdir(os.path.join(extract_dir, subitems[0])) else extract_dir

            for fname in os.listdir(src_root):
                if fname in [".venv", "output_app", "config.json"]:
                    continue
                src_path = os.path.join(src_root, fname)
                dst_path = os.path.join(APP_DIR, fname)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)

            try:
                os.remove(temp_zip)
                shutil.rmtree(extract_dir)
            except Exception:
                pass

            return {"success": True, "message": "Cập nhật thành công! Đang tự khởi động lại phần mềm..."}
        except Exception as e:
            return {"success": False, "error": f"Lỗi cập nhật: {str(e)}"}

    def restart_app(self):
        def _restart():
            time.sleep(1)
            python_exe = sys.executable
            app_script = os.path.join(APP_DIR, "app.py")
            os.system(f'start "" "{python_exe}" "{app_script}"')
            os._exit(0)
        threading.Thread(target=_restart).start()
        return True


HTML_CONTENT = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 3D Studio - Chuyển Ảnh Thành Model 3D Cho Game Unity</title>
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            user-select: none;
        }
        body {
            background-color: #0b0d13;
            color: #e2e8f0;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        header {
            height: 56px;
            background: #131722;
            border-bottom: 1px solid #232936;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 700;
            font-size: 17px;
            color: #60a5fa;
        }
        .logo-tag {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            padding: 4px 8px;
            border-radius: 6px;
            color: #fff;
            font-size: 11px;
            font-weight: 800;
        }
        .version-badge {
            font-size: 11px;
            color: #94a3b8;
            background: #1e293b;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid #334155;
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .gpu-badge {
            font-size: 12px;
            padding: 6px 14px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
        }
        .btn-header {
            background: #1e293b;
            border: 1px solid #334155;
            color: #cbd5e1;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .btn-header:hover {
            background: #334155;
            color: #fff;
            border-color: #60a5fa;
        }
        .main-container {
            flex: 1;
            display: flex;
            height: calc(100vh - 56px);
        }
        .sidebar {
            width: 380px;
            background: #11141d;
            border-right: 1px solid #232936;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            overflow-y: auto;
        }
        .dropzone {
            border: 2px dashed #334155;
            border-radius: 12px;
            padding: 18px 14px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: #161b26;
            position: relative;
            overflow: hidden;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .dropzone:hover {
            border-color: #3b82f6;
            background: #1c2331;
        }
        .dropzone img {
            max-width: 100%;
            max-height: 160px;
            object-fit: contain;
            border-radius: 8px;
            display: none;
        }
        .dropzone-prompt b {
            color: #60a5fa;
            display: block;
            font-size: 14px;
            margin-bottom: 2px;
        }
        .dropzone-prompt p {
            color: #64748b;
            font-size: 11px;
        }
        
        /* Mode Switcher */
        .mode-switcher {
            display: flex;
            background: #0f1219;
            padding: 4px;
            border-radius: 10px;
            border: 1px solid #232936;
        }
        .mode-btn {
            flex: 1;
            padding: 8px 6px;
            border: none;
            border-radius: 7px;
            background: transparent;
            color: #94a3b8;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }
        .mode-btn.active {
            background: #2563eb;
            color: #fff;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
        }
        .mode-btn.cloud.active {
            background: linear-gradient(135deg, #10b981, #3b82f6);
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.4);
        }

        .engine-card {
            background: #161b26;
            border: 1px solid #232936;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 12px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .engine-title {
            color: #f8fafc;
            font-weight: 600;
        }
        .engine-desc {
            color: #38bdf8;
            font-size: 11px;
            line-height: 1.4;
        }
        .engine-desc b {
            color: #34d399;
        }

        .setting-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        label {
            font-size: 12px;
            font-weight: 600;
            color: #94a3b8;
        }
        select, input[type="text"] {
            background: #161b26;
            border: 1px solid #2d3748;
            color: #e2e8f0;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 13px;
            outline: none;
        }
        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            cursor: pointer;
            color: #cbd5e1;
        }
        .btn-generate {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            color: white;
            border: none;
            padding: 13px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35);
        }
        .btn-generate.cloud-active {
            background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35);
        }
        .btn-generate:hover:not(:disabled) {
            transform: translateY(-1px);
        }
        .btn-generate:disabled {
            background: #2d3748;
            color: #718096;
            cursor: not-allowed;
            box-shadow: none;
        }
        .status-box {
            font-size: 12px;
            padding: 10px 12px;
            background: #161b26;
            border-radius: 8px;
            border: 1px solid #232936;
            color: #94a3b8;
            min-height: 44px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .spinner {
            width: 14px;
            height: 14px;
            border: 2px solid #2d3748;
            border-top-color: #38bdf8;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            display: none;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .viewport {
            flex: 1;
            background: radial-gradient(circle at center, #181d28 0%, #0a0c10 100%);
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        model-viewer {
            width: 100%;
            height: 100%;
            --poster-color: transparent;
        }
        .empty-placeholder {
            position: absolute;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            color: #475569;
            pointer-events: none;
        }
        .empty-placeholder svg {
            width: 64px;
            height: 64px;
            opacity: 0.5;
        }
        .action-bar {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            z-index: 10;
        }
        .btn-action {
            background: rgba(19, 23, 34, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #e2e8f0;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .btn-action:hover:not(:disabled) {
            background: #232936;
            border-color: #38bdf8;
        }
        .btn-action:disabled {
            opacity: 0.35;
            cursor: not-allowed;
        }
        .hint-controls {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(19, 23, 34, 0.85);
            backdrop-filter: blur(8px);
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 12px;
            color: #94a3b8;
            pointer-events: none;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Modal styling */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(6px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 100;
        }
        .modal {
            background: #161b26;
            border: 1px solid #2d3748;
            border-radius: 14px;
            width: 520px;
            max-width: 90vw;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-title {
            font-size: 17px;
            font-weight: 700;
            color: #60a5fa;
        }
        .modal-close {
            background: none;
            border: none;
            color: #a0aec0;
            font-size: 20px;
            cursor: pointer;
        }
        .notes-box {
            background: #0f1219;
            border: 1px solid #232936;
            border-radius: 8px;
            padding: 12px;
            font-size: 12px;
            color: #cbd5e1;
            max-height: 180px;
            overflow-y: auto;
            white-space: pre-wrap;
            line-height: 1.5;
        }
        .modal-footer {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: 6px;
        }
        .btn-modal {
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            border: none;
        }
        .btn-modal-primary {
            background: #2563eb;
            color: #fff;
        }
        .btn-modal-secondary {
            background: #2d3748;
            color: #cbd5e0;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-left">
            <div class="logo">
                <span class="logo-tag">3D AI</span> AI 3D Studio
            </div>
            <span class="version-badge" id="appVersion">v1.2.0</span>
        </div>
        <div class="header-right">
            <div class="gpu-badge" id="hwBadge">
                <div class="dot"></div>
                <span id="hwText">NVIDIA RTX 3050 (Ready)</span>
            </div>
            <button class="btn-header" onclick="openUpdateModal()">
                🔄 Cập nhật GitHub
            </button>
        </div>
    </header>

    <div class="main-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <!-- Mode Switcher -->
            <div class="mode-switcher">
                <button class="mode-btn active" id="btnModeLocal" onclick="setEngine('local')">
                    ⚡ GPU Offline
                </button>
                <button class="mode-btn cloud" id="btnModeCloud" onclick="setEngine('cloud_free')">
                    🌟 Cloud Free (Siêu Nét)
                </button>
            </div>

            <!-- Dropzone -->
            <div class="dropzone" id="dropzone" onclick="selectImage()">
                <img id="previewImg" alt="Preview">
                <div class="dropzone-prompt" id="dropPrompt">
                    <b>Chọn ảnh 2D</b>
                    <p>Click để nạp ảnh từ máy tính (PNG, JPG, WebP)</p>
                </div>
            </div>

            <!-- Engine Info Card -->
            <div class="engine-card">
                <span class="engine-title" id="engineTitle">⚡ Chế độ: GPU Cục bộ (Offline)</span>
                <span class="engine-desc" id="engineDesc">
                    Chạy trực tiếp trên RTX 3050. Miễn phí 100%, không cần mạng internet.
                </span>
            </div>

            <!-- Local Settings -->
            <div id="localSettings">
                <div class="setting-group" style="margin-bottom: 8px;">
                    <label>Độ phân giải lưới</label>
                    <select id="mcRes">
                        <option value="auto" selected>Tự động tối ưu theo máy (Khuyên dùng)</option>
                        <option value="128">128 (Siêu nhanh ~3s - Dành cho CPU)</option>
                        <option value="256">256 (Chuẩn chi tiết - Dành cho RTX)</option>
                        <option value="384">384 (Cực chi tiết ~15s)</option>
                    </select>
                </div>
                <div class="setting-group" style="margin-bottom: 8px;">
                    <label class="checkbox-label">
                        <input type="checkbox" id="smoothMesh" checked>
                        <span>✨ Làm mịn bề mặt Taubin (Khử lồi lõm sần sùi)</span>
                    </label>
                </div>
                <div class="setting-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="bakeTexture" checked>
                        <span>Bake Texture sắc nét (Trải UV 1024x1024)</span>
                    </label>
                </div>
            </div>

            <!-- Cloud Free Settings -->
            <div id="cloudSettings" style="display: none;">
                <div class="setting-group">
                    <p style="font-size: 12px; color: #94a3b8; line-height: 1.4;">
                        🚀 <b>Chế độ Miễn Phí 100%:</b> Kết nối trực tiếp vào cụm GPU Cloud mã nguồn mở (InstantMesh / Hunyuan3D).<br>
                        ✓ Không cần mua API Key.<br>
                        ✓ Không cần tài khoản Visa.<br>
                        ✓ Tạo mô hình chi tiết cao từ ảnh đơn.
                    </p>
                </div>
            </div>

            <button class="btn-generate" id="btnGen" onclick="startGeneration()" disabled>
                ⚡ BẮT ĐẦU TẠO 3D (OFFLINE)
            </button>

            <div class="status-box">
                <div class="spinner" id="spinner"></div>
                <span id="statusText">Vui lòng nạp 1 bức ảnh để bắt đầu.</span>
            </div>
        </div>

        <!-- 3D Viewport -->
        <div class="viewport">
            <div class="action-bar">
                <button class="btn-action" id="btnExportGlb" onclick="exportFile('glb')" disabled>
                    📦 Lưu .GLB (Cho Unity Game)
                </button>
                <button class="btn-action" id="btnExportObj" onclick="exportFile('obj')" disabled>
                    📦 Lưu .OBJ (Cho Blender)
                </button>
                <button class="btn-action" id="btnOpenFolder" onclick="openFolder()" disabled>
                    📂 Mở thư mục
                </button>
            </div>

            <div class="empty-placeholder" id="emptyPlaceholder">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                    <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
                <p>Mô hình 3D hoàn chỉnh sẽ hiển thị xoay 360° tại đây</p>
            </div>

            <model-viewer 
                id="viewer" 
                camera-controls 
                auto-rotate 
                auto-rotate-delay="1000" 
                rotation-per-second="25deg"
                shadow-intensity="1.5" 
                shadow-softness="0.8" 
                exposure="1.2"
                style="display: none;">
            </model-viewer>

            <div class="hint-controls" id="hintControls" style="display: none;">
                Chuột trái: Xoay | Con lăn: Zoom | Chuột phải: Di chuyển
            </div>
        </div>
    </div>

    <!-- UPDATE MODAL -->
    <div class="modal-overlay" id="updateModal">
        <div class="modal">
            <div class="modal-header">
                <span class="modal-title">🔄 Cập nhật phần mềm từ GitHub</span>
                <button class="modal-close" onclick="closeUpdateModal()">&times;</button>
            </div>
            <div id="updateStatusText" style="font-size: 13px; color: #a0aec0;">
                Nhấn "Kiểm tra bản mới" để đồng bộ trực tiếp với GitHub Releases.
            </div>
            <div class="notes-box" id="releaseNotesBox" style="display: none;"></div>
            <div class="modal-footer">
                <button class="btn-modal btn-modal-secondary" onclick="closeUpdateModal()">Đóng</button>
                <button class="btn-modal btn-modal-secondary" id="btnCheckUpdate" onclick="checkGitHubUpdates()">Kiểm tra bản mới</button>
                <button class="btn-modal btn-modal-primary" id="btnApplyUpdate" onclick="applyGitHubUpdate()" style="display: none;">🚀 Cập nhật ngay</button>
            </div>
        </div>
    </div>

    <script>
        let selectedImagePath = null;
        let lastFolder = null;
        let latestDownloadUrl = null;
        let currentEngine = "local";

        window.addEventListener('pywebviewready', async () => {
            const data = await window.pywebview.api.get_init_data();
            document.getElementById("appVersion").innerText = data.version;
            
            const hw = data.hardware;
            const badge = document.getElementById("hwBadge");
            const hwText = document.getElementById("hwText");
            
            if (hw.has_gpu) {
                badge.className = "gpu-badge";
                hwText.innerText = "GPU: " + hw.name + " (" + hw.vram_gb + " GB)";
            } else {
                hwText.innerText = hw.status_text;
                document.getElementById("mcRes").value = "128";
                document.getElementById("bakeTexture").checked = false;
            }
        });

        function setEngine(engine) {
            currentEngine = engine;
            const btnLocal = document.getElementById("btnModeLocal");
            const btnCloud = document.getElementById("btnModeCloud");
            const title = document.getElementById("engineTitle");
            const desc = document.getElementById("engineDesc");
            const btnGen = document.getElementById("btnGen");
            const localSettings = document.getElementById("localSettings");
            const cloudSettings = document.getElementById("cloudSettings");

            if (engine === "local") {
                btnLocal.className = "mode-btn active";
                btnCloud.className = "mode-btn cloud";
                title.innerText = "⚡ Chế độ: GPU Cục bộ (Offline)";
                desc.innerHTML = "Chạy trực tiếp trên RTX 3050. Miễn phí 100%, không cần mạng internet.";
                btnGen.className = "btn-generate";
                btnGen.innerText = "⚡ BẮT ĐẦU TẠO 3D (OFFLINE)";
                localSettings.style.display = "block";
                cloudSettings.style.display = "none";
            } else {
                btnLocal.className = "mode-btn";
                btnCloud.className = "mode-btn cloud active";
                title.innerText = "🌟 Chế độ: Cloud Miễn Phí (Hugging Face)";
                desc.innerHTML = "Dùng cụm GPU Cloud miễn phí 100%. <b>Không cần mua API Key</b>, không cần Visa!";
                btnGen.className = "btn-generate cloud-active";
                btnGen.innerText = "🌟 TẠO 3D BẰNG CLOUD MIỄN PHÍ";
                localSettings.style.display = "none";
                cloudSettings.style.display = "block";
            }
        }

        async function selectImage() {
            setStatus("Đang mở hộp thoại chọn ảnh...", false);
            try {
                const res = await window.pywebview.api.select_image();
                if (res && res.path) {
                    selectedImagePath = res.path;
                    document.getElementById("previewImg").src = res.dataUrl;
                    document.getElementById("previewImg").style.display = "block";
                    document.getElementById("dropPrompt").style.display = "none";
                    document.getElementById("btnGen").disabled = false;
                    setStatus("Đã nạp: " + res.name + ". Sẵn sàng tạo 3D!", false);
                } else {
                    setStatus("Chưa chọn ảnh nào.", false);
                }
            } catch (err) {
                setStatus("Lỗi: " + err, false);
            }
        }

        async function startGeneration() {
            if (!selectedImagePath) return;

            const resChoice = document.getElementById("mcRes").value;
            const mcRes = resChoice === "auto" ? "auto" : parseInt(resChoice);
            const bakeTex = document.getElementById("bakeTexture").checked;
            const smoothMesh = document.getElementById("smoothMesh").checked;

            document.getElementById("btnGen").disabled = true;
            document.getElementById("dropzone").style.pointerEvents = "none";
            
            if (currentEngine === "local") {
                setStatus("AI đang tính toán hình khối & làm mịn bề mặt (~5 - 15 giây)...", true);
            } else {
                setStatus("Đang kết nối Cloud Miễn Phí (Hugging Face) & tạo 3D (~30 - 60 giây)...", true);
            }

            try {
                const res = await window.pywebview.api.generate_3d(selectedImagePath, currentEngine, mcRes, bakeTex, smoothMesh);
                if (res.success) {
                    lastFolder = res.folder;
                    const viewer = document.getElementById("viewer");
                    viewer.src = res.glb_data;
                    viewer.style.display = "block";
                    document.getElementById("emptyPlaceholder").style.display = "none";
                    document.getElementById("hintControls").style.display = "block";

                    document.getElementById("btnExportGlb").disabled = false;
                    document.getElementById("btnExportObj").disabled = false;
                    document.getElementById("btnOpenFolder").disabled = false;

                    setStatus("✓ Tạo thành công model 3D (" + res.engine_used + ")! Chuột trái để xoay xem.", false);
                } else {
                    setStatus("❌ Thất bại: " + res.error, false);
                }
            } catch (err) {
                setStatus("❌ Lỗi ngoại lệ: " + err, false);
            } finally {
                document.getElementById("btnGen").disabled = false;
                document.getElementById("dropzone").style.pointerEvents = "auto";
            }
        }

        async function exportFile(type) {
            setStatus("Đang lưu file " + type.toUpperCase() + "...", false);
            const res = await window.pywebview.api.export_file(type);
            if (res.success) {
                setStatus("✓ Đã lưu file tại: " + res.saved_path, false);
            } else if (!res.canceled) {
                setStatus("Lỗi: " + res.error, false);
            }
        }

        async function openFolder() {
            await window.pywebview.api.open_folder(lastFolder);
        }

        function setStatus(text, loading) {
            document.getElementById("statusText").innerText = text;
            document.getElementById("spinner").style.display = loading ? "inline-block" : "none";
        }

        // ====== UPDATE SYSTEM ======
        function openUpdateModal() {
            document.getElementById("updateModal").style.display = "flex";
        }

        function closeUpdateModal() {
            document.getElementById("updateModal").style.display = "none";
        }

        async function checkGitHubUpdates() {
            const statusBox = document.getElementById("updateStatusText");
            const notesBox = document.getElementById("releaseNotesBox");
            const btnApply = document.getElementById("btnApplyUpdate");
            
            statusBox.innerHTML = "Đang kiểm tra bản phát hành trên GitHub...";
            btnApply.style.display = "none";
            notesBox.style.display = "none";

            const res = await window.pywebview.api.check_updates();
            if (!res.success) {
                statusBox.innerHTML = "<span style='color: #ef4444;'>❌ " + res.error + "</span>";
                return;
            }

            if (res.has_update) {
                statusBox.innerHTML = "<span style='color: #10b981; font-weight: bold;'>🎉 Đã có bản cập nhật mới: " + res.latest_version + " (Hiện tại: " + res.current_version + ")</span>";
                notesBox.innerText = res.release_notes || "Cập nhật tính năng và sửa lỗi.";
                notesBox.style.display = "block";
                latestDownloadUrl = res.download_url;
                btnApply.style.display = "inline-block";
            } else {
                statusBox.innerHTML = "<span style='color: #38bdf8;'>✓ Bạn đang sử dụng phiên bản mới nhất (" + res.current_version + ")!</span>";
                if (res.release_notes) {
                    notesBox.innerText = res.release_notes;
                    notesBox.style.display = "block";
                }
            }
        }

        async function applyGitHubUpdate() {
            if (!latestDownloadUrl) return;
            const statusBox = document.getElementById("updateStatusText");
            const btnApply = document.getElementById("btnApplyUpdate");
            btnApply.disabled = true;
            statusBox.innerHTML = "Đang tải bản cập nhật từ GitHub và cài đặt tự động... Xin vui lòng đợi...";

            const res = await window.pywebview.api.apply_update(latestDownloadUrl);
            if (res.success) {
                statusBox.innerHTML = "<span style='color: #10b981; font-weight: bold;'>✓ " + res.message + "</span>";
                setTimeout(async () => {
                    await window.pywebview.api.restart_app();
                }, 2000);
            } else {
                statusBox.innerHTML = "<span style='color: #ef4444;'>❌ " + res.error + "</span>";
                btnApply.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

def main():
    load_thread = threading.Thread(target=load_ai_model, daemon=True)
    load_thread.start()

    api = AppApi()
    window = webview.create_window(
        title=f"AI 3D Studio {APP_VERSION} - Chuyển Ảnh Thành Model 3D Cho Game Unity",
        html=HTML_CONTENT,
        js_api=api,
        width=1260,
        height=840,
        min_size=(960, 660)
    )
    api.set_window(window)
    webview.start(debug=False)

if __name__ == "__main__":
    main()
