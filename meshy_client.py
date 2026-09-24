"""
meshy_client.py - Meshy.ai Cloud REST API Client for AI 3D Studio
Integrates Meshy.ai multi-view diffusion & PBR material generation into AI 3D Studio.
"""
import os
import time
import json
import base64
import logging
import urllib.request
import urllib.error
import io
from PIL import Image

MESHY_API_BASE = "https://api.meshy.ai/openapi/v1"

class MeshyClient:
    def __init__(self, api_key: str = ""):
        self.api_key = (api_key or "").strip()

    def set_api_key(self, api_key: str):
        self.api_key = (api_key or "").strip()

    def get_api_key(self) -> str:
        return self.api_key

    def has_valid_key(self) -> bool:
        return bool(self.api_key and len(self.api_key) >= 10)

    def get_masked_key(self) -> str:
        if not self.has_valid_key():
            return "Chưa cấu hình (Bấm để nhập key)"
        if len(self.api_key) <= 8:
            return "****"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def image_to_data_uri(self, image_path: str) -> str:
        """
        Convert local image to optimized base64 data URI.
        Resizes dimensions to maximum 1024 to minimize upload time and prevent HTTP limits.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

        with Image.open(image_path) as im:
            has_alpha = ("A" in im.getbands())
            max_dim = 1024
            w, h = im.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            if has_alpha:
                im.save(buf, format="PNG", optimize=True)
                mime = "image/png"
            else:
                im = im.convert("RGB")
                im.save(buf, format="JPEG", quality=92, optimize=True)
                mime = "image/jpeg"

            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:{mime};base64,{b64}"

    def create_image_to_3d_task(self, image_path: str, enable_pbr: bool = True, ai_model: str = "latest") -> str:
        """
        Submit image-to-3d request to Meshy.ai.
        Returns the task_id.
        """
        if not self.has_valid_key():
            raise ValueError("Chưa thiết lập Meshy API Key! Vui lòng bấm vào '🔑 Đổi Key' để nhập API Key từ meshy.ai.")

        data_uri = self.image_to_data_uri(image_path)
        url = f"{MESHY_API_BASE}/image-to-3d"
        payload = {
            "image_url": data_uri,
            "enable_pbr": bool(enable_pbr),
            "ai_model": ai_model or "latest"
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AI-3D-Studio/2.0.5"
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                task_id = data.get("result")
                if not task_id:
                    raise RuntimeError(f"Meshy không phản hồi task_id hợp lệ: {data}")
                return task_id
        except urllib.error.HTTPError as e:
            err_msg = ""
            try:
                err_body = json.loads(e.read().decode("utf-8"))
                err_msg = err_body.get("message") or err_body.get("error") or str(err_body)
            except Exception:
                err_msg = f"HTTP {e.code}: {e.reason}"

            if e.code == 401:
                raise ValueError("Meshy API Key không đúng hoặc đã hết hạn! Vui lòng kiểm tra lại key trên meshy.ai.")
            elif e.code == 402:
                raise ValueError("Tài khoản Meshy.ai đã hết điểm tín dụng (Credits). Vui lòng nạp thêm hoặc tạo tài khoản mới!")
            elif e.code == 429:
                raise ValueError("Bị giới hạn tần suất yêu cầu (Rate Limit). Vui lòng đợi 1 phút và thử lại!")
            raise RuntimeError(f"Lỗi máy chủ Meshy ({e.code}): {err_msg}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Không thể kết nối tới máy chủ Meshy.ai ({e.reason}). Vui lòng kiểm tra kết nối mạng Internet!")

    def poll_task(self, task_id: str, progress_callback=None, poll_interval: float = 3.0, timeout: float = 600.0) -> dict:
        """
        Poll Meshy task until finished or failed.
        """
        start_time = time.time()
        url = f"{MESHY_API_BASE}/image-to-3d/{task_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "AI-3D-Studio/2.0.5"
        }

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Quá thời gian chờ xử lý từ Meshy.ai (10 phút).")

            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_text = ""
                try:
                    err_text = json.loads(e.read().decode("utf-8")).get("message", "")
                except Exception:
                    err_text = str(e)
                raise RuntimeError(f"Lỗi kiểm tra tiến độ Meshy ({e.code}): {err_text}")
            except urllib.error.URLError as e:
                logging.warning(f"Tạm thời mất kết nối tới Meshy: {e.reason}")
                time.sleep(poll_interval)
                continue

            status = data.get("status", "")
            progress = data.get("progress", 0)

            if status == "SUCCEEDED":
                if progress_callback:
                    progress_callback("✅ Meshy AI đã hoàn thành mô hình 3D! Đang tải tài nguyên...", 92)
                return data
            elif status == "FAILED":
                task_err = data.get("task_error", {})
                msg = task_err.get("message") if isinstance(task_err, dict) else str(task_err)
                raise RuntimeError(f"Meshy xử lý thất bại: {msg or 'Lỗi tạo mô hình'}")
            elif status == "EXPIRED":
                raise RuntimeError("Tác vụ Meshy đã hết hạn trên máy chủ.")
            elif status == "IN_PROGRESS":
                pct = max(20, min(90, int(progress)))
                if progress_callback:
                    progress_callback(f"✨ Meshy AI đang tái tạo 3D & nướng vật liệu PBR ({pct}%)…", pct)
            else: # PENDING
                if progress_callback:
                    progress_callback("⏳ Meshy AI đang xếp hàng trên cụm máy chủ GPU A100…", 15)

            time.sleep(poll_interval)

    def download_file(self, url: str, target_path: str):
        """Download remote asset to local disk."""
        req = urllib.request.Request(url, headers={"User-Agent": "AI-3D-Studio/2.0.5"})
        with urllib.request.urlopen(req, timeout=90) as resp, open(target_path, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)

    def download_model_assets(self, task_data: dict, target_dir: str, input_image_path: str = None) -> dict:
        """
        Download all model assets (.glb, .obj, thumb) to target directory.
        """
        os.makedirs(target_dir, exist_ok=True)
        model_urls = task_data.get("model_urls", {})
        glb_url = model_urls.get("glb")
        obj_url = model_urls.get("obj")
        thumb_url = task_data.get("thumbnail_url")

        glb_path = os.path.join(target_dir, "model.glb")
        obj_path = os.path.join(target_dir, "model.obj")
        thumb_path = os.path.join(target_dir, "thumb.jpg")
        input_target = os.path.join(target_dir, "input.png")

        # 1. Download GLB (Primary model for WebViewer and standard 3D export)
        if glb_url:
            self.download_file(glb_url, glb_path)
        else:
            raise RuntimeError("Meshy không trả về đường dẫn tải GLB!")

        # 2. Download OBJ if available
        if obj_url:
            try:
                self.download_file(obj_url, obj_path)
            except Exception as e:
                logging.warning(f"Could not download Meshy OBJ: {e}")

        # 3. Save / Download thumbnail
        if thumb_url:
            try:
                self.download_file(thumb_url, thumb_path)
            except Exception as e:
                logging.warning(f"Could not download Meshy thumbnail: {e}")

        # 4. Copy input image
        if input_image_path and os.path.exists(input_image_path):
            try:
                import shutil
                shutil.copy2(input_image_path, input_target)
                if not os.path.exists(thumb_path):
                    with Image.open(input_image_path) as im:
                        im = im.convert("RGB")
                        im.thumbnail((180, 180))
                        im.save(thumb_path, "JPEG", quality=75)
            except Exception as e:
                logging.warning(f"Could not save input/thumb: {e}")

        return {
            "glb_path": glb_path,
            "obj_path": obj_path if os.path.exists(obj_path) else "",
            "thumb_path": thumb_path if os.path.exists(thumb_path) else "",
            "input_path": input_target if os.path.exists(input_target) else ""
        }
