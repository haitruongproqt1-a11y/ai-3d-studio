"""
hf_free_client.py - 100% Free Hugging Face Spaces Cloud 3D Engine for AI 3D Studio
-----------------------------------------------------------------------------------
Generates high-definition 3D models and textures with ZERO paid credit deduction (0 VNĐ).
Uses Hugging Face Free ZeroGPU Spaces (Tencent Hunyuan3D-2.1 / TRELLIS).
Supports:
- Multi-View inputs (Front, Back, Left, Right)
- Pure Plaster/Clay Sculpture Mode (no color/texture, perfect for 3D print & Blender)
- Full PBR Realistic Color Mode (calibrated Dual-View 1:1 camera projection)
Optionally accepts a free Hugging Face User Access Token (https://huggingface.co/settings/tokens)
for extended personal quota and queue priority.
"""

import os
import time
import shutil
import logging
import httpx
from PIL import Image

class HuggingFaceFreeClient:
    def __init__(self, hf_token: str = ""):
        self.hf_token = (hf_token or "").strip()

    def set_token(self, token: str):
        self.hf_token = (token or "").strip()

    def get_token(self) -> str:
        return self.hf_token

    def has_token(self) -> bool:
        return bool(self.hf_token and len(self.hf_token) >= 10)

    def get_masked_token(self) -> str:
        if not self.has_token():
            return "Chưa cấu hình (Bấm để nhập Token miễn phí)"
        if len(self.hf_token) <= 8:
            return "****"
        return f"{self.hf_token[:4]}...{self.hf_token[-4:]}"

    def _download_mesh_from_result(self, raw_result, target_path: str, token: str = None, progress_cb=None) -> bool:
        """
        Robustly extracts GLB model reference from gradio_client output (dict, nested dict, url, filepath),
        and downloads or copies it to target_path.
        """
        if not raw_result:
            return False

        def _extract_url_or_path(obj):
            if not obj:
                return None
            if isinstance(obj, str):
                return obj.strip()
            if isinstance(obj, dict):
                val = obj.get("value")
                if isinstance(val, dict):
                    return val.get("url") or val.get("path")
                elif isinstance(val, str):
                    return val.strip()
                return obj.get("url") or obj.get("path")
            return None

        candidates = []
        if isinstance(raw_result, (list, tuple)):
            for item in raw_result:
                cand = _extract_url_or_path(item)
                if cand:
                    candidates.append(cand)
        else:
            cand = _extract_url_or_path(raw_result)
            if cand:
                candidates.append(cand)

        chosen = None
        for c in candidates:
            if isinstance(c, str):
                c_low = c.lower()
                if ".glb" in c_low or "white_mesh" in c_low or "file=" in c_low:
                    chosen = c
                    break
        if not chosen and candidates:
            chosen = candidates[0]

        if not chosen:
            logging.error(f"No valid file candidate found in HF result: {raw_result}")
            return False

        logging.info(f"Hugging Face downloading model from: {chosen}")

        if chosen.startswith("http://") or chosen.startswith("https://"):
            if progress_cb:
                progress_cb("📥 Đang tải mô hình 3D nguyên bản từ Hugging Face Cloud…", 78)
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            try:
                with httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0), follow_redirects=True) as client:
                    resp = client.get(chosen, headers=headers)
                    if (resp.status_code in (401, 403)) and headers:
                        # Retry without auth header for public spaces
                        resp = client.get(chosen)
                    resp.raise_for_status()
                    with open(target_path, "wb") as f_out:
                        f_out.write(resp.content)
            except Exception as e_dl:
                logging.exception(f"Direct stream download failed: {e_dl}")
                return False
        else:
            if os.path.exists(chosen):
                shutil.copy2(chosen, target_path)
            else:
                logging.error(f"Local file path not found: {chosen}")
                return False

        return os.path.exists(target_path) and os.path.getsize(target_path) > 1024

    def _execute_space_job(self, c, image_path, back_image_path, left_image_path, right_image_path,
                           steps, octree_res, progress_cb, start_pct=35):
        from gradio_client import handle_file

        submit_kwargs = {
            "image": handle_file(image_path),
            "steps": int(steps),
            "guidance_scale": 5.0,
            "octree_resolution": int(octree_res),
            "check_box_rembg": False,
            "num_chunks": 8000,
            "randomize_seed": True,
            "api_name": "/shape_generation"
        }

        if back_image_path and os.path.exists(back_image_path):
            submit_kwargs["mv_image_front"] = handle_file(image_path)
            submit_kwargs["mv_image_back"] = handle_file(back_image_path)
            if left_image_path and os.path.exists(left_image_path):
                submit_kwargs["mv_image_left"] = handle_file(left_image_path)
            if right_image_path and os.path.exists(right_image_path):
                submit_kwargs["mv_image_right"] = handle_file(right_image_path)

        job = c.submit(**submit_kwargs)

        poll_count = 0
        while not job.done():
            time.sleep(1.5)
            poll_count += 1
            try:
                st = job.status()
                code = str(getattr(st, 'code', ''))
                rank = getattr(st, 'rank', None)
                if 'STARTING' in code:
                    if rank is not None and rank > 0:
                        msg = f"⏳ Đang xếp hàng Cloud ZeroGPU: Vị trí #{rank} (0đ Miễn phí)…"
                    else:
                        msg = "⏳ Đang kết nối phân bổ nhân ZeroGPU Cloud (0đ Miễn phí)…"
                    if progress_cb:
                        progress_cb(msg, min(start_pct + 10, start_pct + poll_count))
                elif 'PROCESSING' in code:
                    pct = min(75, start_pct + 10 + poll_count * 2)
                    if progress_cb:
                        progress_cb(f"⚡ ZeroGPU đang suy luận hình khối 3D Flow Matching ({pct}%)…", pct)
            except Exception:
                pass

        return job

    def generate_3d_free(self, image_path: str, progress_cb=None, item_dir=None, steps=15, octree_res=256,
                          back_image_path=None, left_image_path=None, right_image_path=None,
                          color_mode="color") -> dict:
        """
        Executes free 3D generation via Hugging Face Spaces.
        Features dual-strategy resilience:
        1. Try with user token (if configured).
        2. If user token exceeds ZeroGPU quota, automatically retry anonymously.
        3. Supports single-view and multi-view.
        4. Supports Plaster/Clay sculpture vs Full Meshy-grade PBR texture.
        """
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Không tìm thấy ảnh: {image_path}"}

        try:
            from gradio_client import Client
        except ImportError:
            return {"success": False, "error": "Chưa cài đặt gradio_client. Vui lòng cập nhật môi trường!"}

        if progress_cb:
            progress_cb("🌐 Đang kết nối tới máy chủ Hugging Face Cloud Free (ZeroGPU 0đ)…", 15)

        # Prepare target directory
        if not item_dir:
            ts = int(time.time())
            item_dir = os.path.join(r"H:\AI_3D_Studio\output_app", f"hf_{ts}")
        os.makedirs(item_dir, exist_ok=True)
        raw_glb = os.path.join(item_dir, "raw_model.glb")

        job = None
        raw_res = None
        used_token = self.hf_token if self.has_token() else None

        # --- ATTEMPT 1: With User Token (if present) or Anonymous ---
        try:
            if progress_cb:
                progress_cb("🐉 Đang nạp mô hình Tencent Hunyuan3D-2.1 trên Cloud GPU…", 25)

            c = Client(
                "tencent/Hunyuan3D-2.1",
                token=used_token,
                httpx_kwargs={"timeout": httpx.Timeout(300.0, connect=60.0)},
                download_files=False,
                verbose=False
            )

            if progress_cb:
                progress_cb("⏳ Đang gửi yêu cầu vào hàng đợi Cloud ZeroGPU (0đ Miễn phí)…", 35)

            job = self._execute_space_job(
                c, image_path, back_image_path, left_image_path, right_image_path,
                steps, octree_res, progress_cb, start_pct=35
            )

            raw_res = job.result()

        except Exception as e_first:
            err_str = str(e_first)
            logging.warning(f"HF Free Attempt 1 warning: {err_str}")

            is_quota_err = any(k in err_str.lower() for k in ("quota", "zerogpu", "rate limit", "exceeded", "429"))

            # If user token had quota exceeded, retry immediately with anonymous pool!
            if used_token and is_quota_err:
                try:
                    if progress_cb:
                        progress_cb("💡 Token đã chạm hạn mức ngày. Tự động chuyển sang cụm ZeroGPU công cộng (0đ)…", 30)
                    used_token = None
                    c_anon = Client(
                        "tencent/Hunyuan3D-2.1",
                        token=None,
                        httpx_kwargs={"timeout": httpx.Timeout(300.0, connect=60.0)},
                        download_files=False,
                        verbose=False
                    )
                    job = self._execute_space_job(
                        c_anon, image_path, back_image_path, left_image_path, right_image_path,
                        steps, octree_res, progress_cb, start_pct=35
                    )
                    raw_res = job.result()
                except Exception as e_anon:
                    logging.exception(f"HF Free Anonymous Attempt failed: {e_anon}")
                    return {
                        "success": False,
                        "error": f"Hạn mức Cloud Free ZeroGPU tạm hết: {e_anon}",
                        "quota_exceeded": True
                    }
            else:
                return {
                    "success": False,
                    "error": f"Lỗi Hugging Face Cloud Free: {err_str}",
                    "quota_exceeded": is_quota_err
                }

        # Download raw GLB from result
        dl_ok = self._download_mesh_from_result(raw_res, raw_glb, token=used_token, progress_cb=progress_cb)
        if not dl_ok and job is not None:
            try:
                outs = job.outputs()
                if outs and len(outs) > 0:
                    dl_ok = self._download_mesh_from_result(outs[-1], raw_glb, token=used_token, progress_cb=progress_cb)
            except Exception:
                pass

        if not dl_ok or not os.path.exists(raw_glb):
            return {
                "success": False,
                "error": "Không thể trích xuất mô hình 3D từ Không gian Hugging Face Cloud Free."
            }

        import trimesh
        mesh = trimesh.load(raw_glb, force="mesh")

        if color_mode == "clay":
            if progress_cb:
                progress_cb("🏛️ Đang tạo Tượng Thạch Cao Clay đơn sắc mịn màng…", 80)
            from texture_engine import create_clay_sculpture_mesh
            baked_mesh = create_clay_sculpture_mesh(mesh)
        else:
            if progress_cb:
                progress_cb("🎨 AI đang nướng bản đồ vân PBR 360° 6 Hướng (Chất lượng Meshy)…", 80)
            from texture_engine import bake_meshy_pbr_mesh
            baked_mesh, _ = bake_meshy_pbr_mesh(
                mesh, image_path,
                back_image_source=back_image_path,
                left_image_source=left_image_path,
                right_image_source=right_image_path,
                color_mode=color_mode
            )

        if progress_cb:
            progress_cb("💾 Đang xuất tệp mô hình GLB và OBJ sắc nét…", 95)

        glb_path = os.path.join(item_dir, "model.glb")
        obj_path = os.path.join(item_dir, "model.obj")
        baked_mesh.export(glb_path)
        baked_mesh.export(obj_path)

        engine_name = "🌐 Hugging Face Free Cloud (0đ - Thạch Cao Clay)" if color_mode == "clay" else "🌐 Hugging Face Free Cloud (0đ)"

        return {
            "success": True,
            "glb_path": glb_path,
            "obj_path": obj_path,
            "item_dir": item_dir,
            "engine_used": engine_name
        }
