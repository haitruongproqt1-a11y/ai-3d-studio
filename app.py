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
from tsr.bake_texture import bake_texture
import trimesh
import cv2

try:
    from gradio_client import Client, handle_file
except ImportError:
    Client = None

logging.basicConfig(level=logging.INFO)

APP_VERSION = "v1.4.0"
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
                "status_text": f"NVIDIA {gpu_name} ({total_vram_gb} GB)",
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
        "status_text": f"Chế độ CPU ({cpu_count} luồng)",
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
        chunk_size = 65536 if (current_device.startswith("cuda") and HARDWARE_INFO.get("vram_gb", 0) >= 5.5) else (16384 if current_device.startswith("cuda") else 1024)
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


# ============ PBR & TEXTURE ENHANCEMENT ENGINE ============
def enhance_texture_vibrancy(tex_img):
    """Enhance texture saturation, contrast, and sharpness for photorealism"""
    try:
        # Boost color saturation to make colors vibrant and realistic
        enhancer = ImageEnhance.Color(tex_img)
        tex_img = enhancer.enhance(1.22)
        # Boost contrast slightly
        con_enhancer = ImageEnhance.Contrast(tex_img)
        tex_img = con_enhancer.enhance(1.08)
        # Unsharp mask for crisp high-frequency details
        tex_img = tex_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    except Exception as e:
        logging.warning(f"Texture enhance skipped: {e}")
    return tex_img

def generate_normal_map(tex_img, intensity=2.5):
    """Generate high-fidelity Normal Map from diffuse texture for PBR lighting"""
    try:
        img_np = np.array(tex_img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        nx = -sobelx * intensity
        ny = -sobely * intensity
        nz = 1.0
        
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= norm
        ny /= norm
        nz /= norm
        
        normal_map = np.zeros_like(img_np, dtype=np.uint8)
        normal_map[:, :, 0] = np.clip((nx + 1.0) * 0.5 * 255.0, 0, 255) # R
        normal_map[:, :, 1] = np.clip((ny + 1.0) * 0.5 * 255.0, 0, 255) # G
        normal_map[:, :, 2] = np.clip((nz + 1.0) * 0.5 * 255.0, 0, 255) # B
        return Image.fromarray(normal_map)
    except Exception as e:
        logging.warning(f"Normal map generation skipped: {e}")
        return None


class AppApi:
    def __init__(self):
        self._window = None
        self.last_glb_path = ""
        self.last_obj_path = ""
        self.last_folder = ""
        self.config = load_config()

    def set_window(self, window):
        self._window = window

    def get_init_data(self):
        return {
            "version": APP_VERSION,
            "hardware": HARDWARE_INFO,
            "config": self.config
        }

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

    def preprocess_image(self, file_path):
        """Intelligently cleans background without destroying white/gray object details"""
        orig_img = Image.open(file_path)
        # Check if already transparent PNG
        if orig_img.mode in ("RGBA", "LA"):
            alpha = np.array(orig_img)[:, :, -1]
            if (alpha < 240).any() and (alpha > 15).any():
                logging.info("Using existing transparent mask from input image")
                image = resize_foreground(orig_img, 0.85)
                image_np = np.array(image).astype(np.float32) / 255.0
                image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
                return Image.fromarray((image_np * 255.0).astype(np.uint8))
        
        # Use AI background removal (rembg)
        logging.info("Running AI Background Removal (rembg)...")
        rgb_img = orig_img.convert("RGB")
        image = remove_background(rgb_img)
        image = resize_foreground(image, 0.85)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
        return Image.fromarray((image_np * 255.0).astype(np.uint8))

    # ======== GENERATE 3D (DUAL ENGINE WITH PBR REALISM) ========
    def generate_3d(self, file_path, engine="local", mc_resolution=None, bake_texture_flag=None, smooth_mesh=True, quality="max"):
        if engine == "cloud_free":
            return self._generate_cloud_free(file_path, quality)
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

            image = self.preprocess_image(file_path)

            timestamp = int(time.time())
            item_dir = os.path.join(OUTPUT_DIR, str(timestamp))
            os.makedirs(item_dir, exist_ok=True)
            image.save(os.path.join(item_dir, "input.png"))

            with torch.no_grad():
                scene_codes = model([image], device=current_device)

            meshes = model.extract_mesh(scene_codes, not bake_texture_flag, resolution=int(mc_resolution))
            
            # Surface Taubin Smoothing to eliminate bumpy artifacts
            if smooth_mesh:
                try:
                    trimesh.smoothing.filter_taubin(meshes[0], lamb=0.5, nu=-0.53, iterations=12)
                except Exception as e:
                    logging.warning(f"Smoothing skipped: {e}")

            out_obj_path = os.path.join(item_dir, "model.obj")
            out_glb_path = os.path.join(item_dir, "model.glb")
            out_tex_path = os.path.join(item_dir, "texture.png")
            out_normal_path = os.path.join(item_dir, "normal.png")

            if bake_texture_flag and HARDWARE_INFO["has_gpu"]:
                import xatlas
                bake_output = bake_texture(meshes[0], model, scene_codes[0], 1024)
                xatlas.export(out_obj_path, meshes[0].vertices[bake_output["vmapping"]], bake_output["indices"], bake_output["uvs"], meshes[0].vertex_normals[bake_output["vmapping"]])
                
                tex_img = Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)
                # Boost colors for photorealism
                tex_img = enhance_texture_vibrancy(tex_img)
                tex_img.save(out_tex_path)
                
                # Generate Normal Map
                norm_img = generate_normal_map(tex_img)
                if norm_img:
                    norm_img.save(out_normal_path)

                loaded = trimesh.load(out_obj_path)
                mat = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=tex_img,
                    roughnessFactor=0.35, # Natural glossy finish
                    metallicFactor=0.05
                )
                loaded.visual.material = mat
                loaded.export(out_glb_path)
            else:
                meshes[0].export(out_glb_path)
                meshes[0].export(out_obj_path)

            self.last_glb_path = out_glb_path
            self.last_obj_path = out_obj_path
            self.last_folder = item_dir

            with open(out_glb_path, "rb") as f:
                glb_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{glb_b64}",
                "glb_path": out_glb_path,
                "obj_path": out_obj_path,
                "folder": item_dir,
                "engine_used": f"Local GPU (NVIDIA RTX 3050 + PBR Texture Booster)"
            }
        except Exception as e:
            logging.exception("Error in _generate_local")
            return {"success": False, "error": str(e)}

    def _generate_cloud_free(self, file_path, quality="max"):
        """Generates 3D with 100% Free Cloud and PBR Post-Processing"""
        if Client is None:
            return {"success": False, "error": "Chưa cài đặt gradio_client!"}
        
        try:
            hf_token = self.config.get("hf_token", "").strip() or None
            logging.info("Connecting to InstantMesh Cloud...")
            client = Client("TencentARC/InstantMesh", token=hf_token)
            
            # Step 1: Preprocess
            prep_res = client.predict(
                input_image=handle_file(file_path),
                api_name="/preprocess"
            )
            
            # Step 2: Multi-View (Optimal 30-35 steps within ZeroGPU 60s limit)
            steps = 32 if quality == "max" else 22
            logging.info(f"Generating Multi-views with {steps} sample steps...")
            client.predict(
                input_image=handle_file(prep_res),
                sample_steps=steps,
                sample_seed=42,
                api_name="/generate_mvs"
            )
            
            # Step 3: Make 3D
            logging.info("Building 3D Mesh and Texture...")
            obj_path, glb_path = client.predict(api_name="/make3d")
            
            timestamp = int(time.time())
            item_dir = os.path.join(OUTPUT_DIR, f"cloud_pbr_{timestamp}")
            os.makedirs(item_dir, exist_ok=True)
            
            local_glb = os.path.join(item_dir, "model.glb")
            local_obj = os.path.join(item_dir, "model.obj")
            shutil.copy2(glb_path, local_glb)
            shutil.copy2(obj_path, local_obj)
            
            # Post-process GLB to enhance PBR realistic materials
            try:
                tm = trimesh.load(local_glb)
                if hasattr(tm, 'visual') and hasattr(tm.visual, 'material'):
                    if hasattr(tm.visual.material, 'roughnessFactor'):
                        tm.visual.material.roughnessFactor = 0.35
                    if hasattr(tm.visual.material, 'metallicFactor'):
                        tm.visual.material.metallicFactor = 0.05
                    tm.export(local_glb)
            except Exception as e:
                logging.warning(f"GLB post-tune skipped: {e}")

            self.last_glb_path = local_glb
            self.last_obj_path = local_obj
            self.last_folder = item_dir
            
            with open(local_glb, "rb") as f:
                glb_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {
                "success": True,
                "glb_data": f"data:model/gltf-binary;base64,{glb_b64}",
                "glb_path": local_glb,
                "obj_path": local_obj,
                "folder": item_dir,
                "engine_used": f"Cloud Multi-view ({steps} Steps + PBR Material)"
            }
        except Exception as e:
            logging.exception("Error in _generate_cloud_free")
            err_msg = str(e)
            if "ZeroGPU quota" in err_msg or "quota" in err_msg.lower():
                err_msg = (
                    "Hugging Face Cloud đang tạm hết lượt ZeroGPU miễn phí cho IP này (hoặc server bận).\n"
                    "👉 Bạn hãy chuyển sang chế độ 'GPU Offline' (RTX 3050) để tạo ngay lập tức không giới hạn và siêu nét, "
                    "hoặc dán Hugging Face Token miễn phí trong Cài đặt!"
                )
            return {"success": False, "error": err_msg}

    def save_settings(self, hf_token=None):
        if hf_token is not None:
            self.config["hf_token"] = hf_token.strip()
        save_config(self.config)
        return {"success": True}

    def open_folder(self, folder_path=None):
        target = folder_path if folder_path else (self.last_folder if self.last_folder else OUTPUT_DIR)
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
            save_filename=f"model_pbr_{int(time.time())}.{file_type}",
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
        raw_urls = [
            f"https://raw.githubusercontent.com/{repo}/main/version.json",
            f"https://raw.githubusercontent.com/{repo}/master/version.json"
        ]
        for r_url in raw_urls:
            try:
                r_req = urllib.request.Request(r_url, headers={"User-Agent": "AI-3D-Studio-Updater"})
                with urllib.request.urlopen(r_req, timeout=5) as resp:
                    if resp.status == 200:
                        vdata = json.loads(resp.read().decode("utf-8"))
                        ver = str(vdata.get("version", "")).replace("v", "").strip()
                        cur = APP_VERSION.replace("v", "").strip()
                        has_up = (ver != cur and ver != "")
                        notes = "\n".join(vdata.get("releaseNotes", [])) if isinstance(vdata.get("releaseNotes"), list) else vdata.get("releaseNotes", "")
                        return {
                            "success": True,
                            "current_version": APP_VERSION,
                            "latest_version": f"v{ver}",
                            "has_update": has_up,
                            "release_notes": notes,
                            "download_url": vdata.get("updateUrl", ""),
                            "repo": repo
                        }
            except Exception:
                pass

        return {
            "success": True,
            "current_version": APP_VERSION,
            "latest_version": APP_VERSION,
            "has_update": False,
            "release_notes": "Bạn đang sử dụng phiên bản mới nhất!",
            "download_url": "",
            "repo": repo
        }

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
    <title>AI 3D Studio - Chuyển Ảnh Thành Model 3D Siêu Thực Cho Game Unity</title>
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
            width: 390px;
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
            background: linear-gradient(135deg, #2563eb, #7c3aed);
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
            padding: 14px;
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
            background: radial-gradient(circle at center, #1e2433 0%, #0a0c10 100%);
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
            top: 18px;
            right: 18px;
            display: flex;
            gap: 8px;
            z-index: 10;
        }
        .btn-action {
            background: rgba(19, 23, 34, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #e2e8f0;
            padding: 7px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .btn-action:hover:not(:disabled) {
            background: #232936;
            border-color: #38bdf8;
        }
        .btn-action:disabled {
            opacity: 0.35;
            cursor: not-allowed;
        }

        /* Viewport Controls */
        .lighting-bar {
            position: absolute;
            top: 18px;
            left: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
            z-index: 10;
            background: rgba(19, 23, 34, 0.8);
            backdrop-filter: blur(8px);
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 12px;
        }
        .lighting-bar select {
            padding: 3px 8px;
            font-size: 11px;
            border-radius: 6px;
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
            <span class="version-badge" id="appVersion">v1.4.0</span>
        </div>
        <div class="header-right">
            <div class="gpu-badge" id="hwBadge">
                <div class="dot"></div>
                <span id="hwText">NVIDIA RTX 3050 (Ready)</span>
            </div>
            <button class="btn-header" onclick="openSettingsModal()">
                ⚙️ Cài đặt
            </button>
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
                    ⚡ GPU Offline (RTX 3050)
                </button>
                <button class="mode-btn cloud" id="btnModeCloud" onclick="setEngine('cloud_free')">
                    🌟 Cloud Multi-View
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
                <span class="engine-title" id="engineTitle">⚡ Chế độ: GPU Cục bộ (RTX 3050 Offline)</span>
                <span class="engine-desc" id="engineDesc">
                    Chạy trực tiếp trên <b>NVIDIA RTX 3050</b>. Tốc độ siêu nhanh (~10s), không giới hạn lượt tạo, không cần mạng internet!
                </span>
            </div>

            <!-- Local Settings -->
            <div id="localSettings">
                <div class="setting-group" style="margin-bottom: 8px;">
                    <label>Độ phân giải lưới 3D</label>
                    <select id="mcRes">
                        <option value="auto" selected>Tự động tối ưu (256 - Siêu chuẩn)</option>
                        <option value="256">256 (Chuẩn chi tiết cao - Khuyên dùng)</option>
                        <option value="384">384 (Cực chi tiết tối đa)</option>
                        <option value="128">128 (Siêu tốc ~3s - Dành cho CPU)</option>
                    </select>
                </div>
                <div class="setting-group" style="margin-bottom: 8px;">
                    <label class="checkbox-label">
                        <input type="checkbox" id="smoothMesh" checked>
                        <span>✨ Làm mịn bề mặt Taubin (Khử lồi lõm sần sùi)</span>
                    </label>
                </div>
                <div class="setting-group" style="margin-bottom: 8px;">
                    <label class="checkbox-label">
                        <input type="checkbox" id="bakeTexture" checked>
                        <span>🎨 Bake Texture sắc nét + Tăng tương phản PBR</span>
                    </label>
                </div>
                <p style="font-size: 11px; color: #34d399; line-height: 1.3;">
                    ✓ Tự động tách nền thông minh bằng AI, bảo toàn 100% chi tiết.<br>
                    ✓ Tự động phủ lớp vật liệu PBR bóng bẩy chuẩn game Unity.
                </p>
            </div>

            <!-- Cloud Free Settings -->
            <div id="cloudSettings" style="display: none;">
                <div class="setting-group" style="margin-bottom: 6px;">
                    <label>Chất lượng tái tạo Cloud</label>
                    <select id="cloudQuality">
                        <option value="max" selected>32 Steps - Chuẩn ZeroGPU Miễn Phí (An toàn)</option>
                        <option value="fast">22 Steps - Tốc độ nhanh</option>
                    </select>
                </div>
                <p style="font-size: 11px; color: #38bdf8; line-height: 1.3;">
                    ✓ Tái tạo đa góc nhìn (Multi-view 3D) trên cụm GPU Cloud miễn phí.<br>
                    ✓ Đã tối ưu số bước tính toán để không bao giờ bị vượt quá giới hạn ZeroGPU.
                </p>
            </div>

            <button class="btn-generate" id="btnGen" onclick="startGeneration()" disabled>
                ⚡ BẮT ĐẦU TẠO 3D (RTX 3050)
            </button>

            <div class="status-box">
                <div class="spinner" id="spinner"></div>
                <span id="statusText">Vui lòng nạp 1 bức ảnh để bắt đầu.</span>
            </div>
        </div>

        <!-- 3D Viewport -->
        <div class="viewport">
            <!-- Lighting bar -->
            <div class="lighting-bar">
                <span>💡 Ánh sáng:</span>
                <select id="lightingMode" onchange="changeLighting()">
                    <option value="studio" selected>Studio Chiếu Sáng (Sắc nét rực rỡ)</option>
                    <option value="aces">ACES Điện Ảnh (Độ sâu tương phản)</option>
                    <option value="soft">Ánh Sáng Tự Nhiên (Mềm mại)</option>
                </select>
            </div>

            <!-- Action bar -->
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
                tone-mapping="aces"
                shadow-intensity="2" 
                shadow-softness="0.7" 
                exposure="1.25"
                style="display: none;">
            </model-viewer>

            <div class="hint-controls" id="hintControls" style="display: none;">
                Chuột trái: Xoay | Con lăn: Zoom | Chuột phải: Di chuyển
            </div>
        </div>
    </div>

    <!-- SETTINGS MODAL -->
    <div class="modal-overlay" id="settingsModal">
        <div class="modal">
            <div class="modal-header">
                <span class="modal-title">⚙️ Cài đặt hệ thống & Token</span>
                <button class="modal-close" onclick="closeSettingsModal()">&times;</button>
            </div>
            <div class="setting-group">
                <label>Hugging Face Token (Tùy chọn - Hoàn toàn miễn phí):</label>
                <input type="text" id="hfTokenInput" placeholder="hf_..." style="font-family: monospace;">
                <p style="font-size: 11px; color: #94a3b8; line-height: 1.4; margin-top: 4px;">
                    💡 Đăng ký miễn phí tại <b style="color: #60a5fa;">huggingface.co/settings/tokens</b> (không cần Visa) để nhận thêm quota ZeroGPU Cloud khi dùng chế độ Cloud. Nếu dùng GPU Offline RTX 3050 thì không cần điền.
                </p>
            </div>
            <div class="modal-footer">
                <button class="btn-modal btn-modal-secondary" onclick="closeSettingsModal()">Hủy</button>
                <button class="btn-modal btn-modal-primary" onclick="saveSettings()">💾 Lưu Cài Đặt</button>
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
        let currentEngine = "local"; // Default to fast local RTX 3050!

        window.addEventListener('pywebviewready', async () => {
            const data = await window.pywebview.api.get_init_data();
            document.getElementById("appVersion").innerText = data.version;
            
            const hw = data.hardware;
            const hwText = document.getElementById("hwText");
            if (hw.has_gpu) {
                hwText.innerText = "GPU: " + hw.name + " (" + hw.vram_gb + " GB)";
            } else {
                hwText.innerText = hw.status_text;
            }

            if (data.config && data.config.hf_token) {
                document.getElementById("hfTokenInput").value = data.config.hf_token;
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
                title.innerText = "⚡ Chế độ: GPU Cục bộ (RTX 3050 Offline)";
                desc.innerHTML = "Chạy trực tiếp trên <b>NVIDIA RTX 3050</b>. Tốc độ siêu nhanh (~10s), không giới hạn lượt tạo, không cần mạng internet!";
                btnGen.className = "btn-generate";
                btnGen.innerText = "⚡ BẮT ĐẦU TẠO 3D (RTX 3050)";
                localSettings.style.display = "block";
                cloudSettings.style.display = "none";
            } else {
                btnLocal.className = "mode-btn";
                btnCloud.className = "mode-btn cloud active";
                title.innerText = "🌟 Chế độ: Cloud Siêu Nét (PBR Multi-View)";
                desc.innerHTML = "Dùng cụm GPU Cloud miễn phí với thuật toán Multi-view và vật liệu PBR bóng bẩy chuẩn Game.";
                btnGen.className = "btn-generate cloud-active";
                btnGen.innerText = "🌟 BẮT ĐẦU TẠO 3D TRÊN CLOUD";
                localSettings.style.display = "none";
                cloudSettings.style.display = "block";
            }
        }

        function changeLighting() {
            const mode = document.getElementById("lightingMode").value;
            const viewer = document.getElementById("viewer");
            if (mode === "studio") {
                viewer.setAttribute("tone-mapping", "aces");
                viewer.setAttribute("exposure", "1.3");
                viewer.setAttribute("shadow-intensity", "2");
            } else if (mode === "aces") {
                viewer.setAttribute("tone-mapping", "aces");
                viewer.setAttribute("exposure", "1.0");
                viewer.setAttribute("shadow-intensity", "2.5");
            } else {
                viewer.setAttribute("tone-mapping", "neutral");
                viewer.setAttribute("exposure", "1.1");
                viewer.setAttribute("shadow-intensity", "1.2");
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
            const quality = document.getElementById("cloudQuality").value;

            document.getElementById("btnGen").disabled = true;
            document.getElementById("dropzone").style.pointerEvents = "none";
            
            if (currentEngine === "local") {
                setStatus("AI đang phân giải hình khối & làm mịn Taubin (~8 - 15 giây)...", true);
            } else {
                setStatus("Đang tính toán Multi-view & phủ vật liệu PBR (~20 - 35 giây)...", true);
            }

            try {
                const res = await window.pywebview.api.generate_3d(selectedImagePath, currentEngine, mcRes, bakeTex, smoothMesh, quality);
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

                    setStatus("✓ Đã tạo thành công mô hình 3D (" + res.engine_used + ")! Chuột trái để xoay xem.", false);
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

        // ====== SETTINGS MODAL ======
        function openSettingsModal() {
            document.getElementById("settingsModal").style.display = "flex";
        }

        function closeSettingsModal() {
            document.getElementById("settingsModal").style.display = "none";
        }

        async function saveSettings() {
            const token = document.getElementById("hfTokenInput").value.trim();
            await window.pywebview.api.save_settings(token);
            closeSettingsModal();
            setStatus("✓ Đã lưu cài đặt!", false);
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
        width=1280,
        height=850,
        min_size=(980, 680)
    )
    api.set_window(window)
    webview.start(debug=False)

if __name__ == "__main__":
    main()
