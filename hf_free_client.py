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

    def generate_3d_free(self, image_path: str, progress_cb=None, item_dir=None, steps=15, octree_res=256,
                         back_image_path=None, left_image_path=None, right_image_path=None,
                         color_mode="color") -> dict:
        """
        Executes free 3D generation via Hugging Face Spaces.
        Supports single view or multi-view (front, back, left, right),
        and output style: 'color' (full PBR) or 'clay' (untextured sculpture).
        """
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Không tìm thấy ảnh: {image_path}"}

        try:
            from gradio_client import Client, handle_file
        except ImportError:
            return {"success": False, "error": "Chưa cài đặt gradio_client. Vui lòng cập nhật môi trường!"}

        if progress_cb:
            progress_cb("🌐 Đang kết nối tới máy chủ Hugging Face Cloud Free (ZeroGPU 0đ)…", 15)

        token_arg = self.hf_token if self.has_token() else None

        temp_glb_path = None
        try:
            if progress_cb:
                progress_cb("🐉 Đang nạp mô hình Tencent Hunyuan3D-2.1 trên Cloud GPU…", 25)
            c = Client("tencent/Hunyuan3D-2.1", token=token_arg, verbose=False)

            if progress_cb:
                progress_cb("⏳ Đang gửi yêu cầu vào hàng đợi Cloud ZeroGPU (0đ Miễn phí)…", 35)

            submit_kwargs = {
                "image": handle_file(image_path),
                "api_name": "/shape_generation"
            }

            # If multi-view back image provided, supply multi-view ports
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
                    if 'STARTING' in code:
                        if progress_cb:
                            progress_cb("⏳ Đang chuẩn bị nhân tính toán ZeroGPU Cloud…", 40)
                    elif 'PROCESSING' in code:
                        pct = min(75, 45 + poll_count * 2)
                        if progress_cb:
                            progress_cb(f"⚡ ZeroGPU đang suy luận hình khối 3D ({pct}%)…", pct)
                except Exception:
                    pass

            outputs = job.outputs()
            if outputs and len(outputs) > 0:
                first_out = outputs[0]
                if isinstance(first_out, (list, tuple)) and len(first_out) > 0:
                    item = first_out[0]
                    if isinstance(item, dict) and "value" in item:
                        temp_glb_path = item["value"]
                    elif isinstance(item, str):
                        temp_glb_path = item

            if not temp_glb_path:
                try:
                    res = job.result()
                    if isinstance(res, (list, tuple)) and len(res) > 0:
                        item = res[0]
                        if isinstance(item, dict) and "value" in item:
                            temp_glb_path = item["value"]
                        elif isinstance(item, str):
                            temp_glb_path = item
                except Exception as e_res:
                    logging.warning(f"job.result fallback notice: {e_res}")

        except Exception as e_hy:
            logging.exception(f"Hunyuan3D-2.1 free space error: {e_hy}")
            err_msg = str(e_hy)
            if "quota" in err_msg.lower() or "zerogpu" in err_msg.lower():
                return {
                    "success": False,
                    "error": (
                        "Bạn đã dùng hết hạn mức ZeroGPU miễn phí cho địa chỉ IP này.\n\n"
                        "💡 Cách mở khóa dùng tiếp 100% Miễn Phí:\n"
                        "1. Vào https://huggingface.co/settings/tokens tạo một Token miễn phí (30 giây).\n"
                        "2. Nhập Token vào ô 'Cấu hình Hugging Face Free Token' bên dưới.\n"
                        "3. Hoặc chuyển sang chế độ '⚡ RTX Hunyuan3D Turbo' để tạo 100% Offline trên GPU máy tính của bạn!"
                    )
                }

        if not temp_glb_path or not os.path.exists(temp_glb_path):
            return {
                "success": False,
                "error": "Không thể trích xuất mô hình 3D từ Không gian Hugging Face. Vui lòng thử lại sau giây lát hoặc sử dụng chế độ RTX Offline!"
            }

        # Save downloaded mesh to target directory
        if not item_dir:
            ts = int(time.time())
            item_dir = os.path.join(r"H:\AI_3D_Studio\output_app", f"hf_{ts}")
        os.makedirs(item_dir, exist_ok=True)

        raw_glb = os.path.join(item_dir, "raw_model.glb")
        shutil.copy2(temp_glb_path, raw_glb)

        import trimesh
        mesh = trimesh.load(raw_glb, force="mesh")

        if color_mode == "clay":
            if progress_cb:
                progress_cb("🏛️ Đang tạo Tượng Thạch Cao Clay đơn sắc mịn màng…", 80)
            from texture_engine import create_clay_sculpture_mesh
            baked_mesh = create_clay_sculpture_mesh(mesh)
        else:
            if progress_cb:
                progress_cb("🎨 AI đang nướng vân bề mặt PBR Dual-View HD (Chất lượng Meshy)…", 80)
            from texture_engine import bake_meshy_pbr_mesh
            baked_mesh, _ = bake_meshy_pbr_mesh(
                mesh, image_path,
                back_image_source=back_image_path,
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
