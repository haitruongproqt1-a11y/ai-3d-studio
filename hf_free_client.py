"""
hf_free_client.py - 100% Free Hugging Face Spaces Cloud 3D Engine for AI 3D Studio
-----------------------------------------------------------------------------------
Generates high-definition 3D models and textures with ZERO paid credit deduction (0 VNĐ).
Uses Hugging Face Free ZeroGPU Spaces (Tencent Hunyuan3D-2.1 / TRELLIS).
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

    def generate_3d_free(self, image_path: str, progress_cb=None, item_dir=None, steps=15, octree_res=256) -> dict:
        """
        Executes free 3D generation via Hugging Face Spaces.
        1. Queries Tencent Hunyuan3D-2.1 or TRELLIS Free Space.
        2. Retrieves the 3D mesh.
        3. Applies our local Meshy-grade photographic texture engine.
        Returns dict with success status and file paths.
        """
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Không tìm thấy ảnh: {image_path}"}

        try:
            from gradio_client import Client, handle_file
        except ImportError:
            return {"success": False, "error": "Chưa cài đặt gradio_client. Vui lòng cập nhật môi trường!"}

        if progress_cb:
            progress_cb("🌐 Đang kết nối tới Hugging Face Cloud Free (ZeroGPU 0đ)…", 15)

        client_kwargs = {"verbose": False}
        if self.has_token():
            client_kwargs["hf_token"] = self.hf_token

        # Try Tencent Hunyuan3D-2.1 First (Fastest & Most Stable Shape Gen)
        temp_glb_path = None
        try:
            if progress_cb:
                progress_cb("🐉 Đang nạp mô hình Tencent Hunyuan3D-2.1 trên Cloud GPU…", 25)
            c = Client("tencent/Hunyuan3D-2.1", **client_kwargs)

            if progress_cb:
                progress_cb("⚡ ZeroGPU đang suy luận hình học 3D (Không tốn Credit)…", 45)

            res = c.predict(
                image=handle_file(image_path),
                mv_image_front=None,
                mv_image_back=None,
                mv_image_left=None,
                mv_image_right=None,
                steps=int(steps),
                guidance_scale=5.0,
                seed=1234,
                octree_resolution=int(octree_res),
                check_box_rembg=True,
                num_chunks=8000,
                randomize_seed=True,
                api_name="/shape_generation"
            )

            # res is (file, output, mesh_stats, seed)
            # res[0] is dict with {'value': filepath} or string filepath
            if isinstance(res, (list, tuple)) and len(res) > 0:
                out_item = res[0]
                if isinstance(out_item, dict) and "value" in out_item:
                    temp_glb_path = out_item["value"]
                elif isinstance(out_item, str):
                    temp_glb_path = out_item
        except Exception as e_hy:
            logging.warning(f"Hunyuan3D-2.1 free space error: {e_hy}")
            # Check if quota exceeded
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

        # Apply our Meshy-grade Universal Camera-Adaptive PBR Texture Engine
        if progress_cb:
            progress_cb("🎨 AI đang nướng vân bề mặt PBR Dual-View HD (Chất lượng Meshy)…", 75)

        import trimesh
        from texture_engine import bake_meshy_pbr_mesh

        mesh = trimesh.load(raw_glb, force="mesh")
        baked_mesh, _ = bake_meshy_pbr_mesh(mesh, image_path)

        if progress_cb:
            progress_cb("💾 Đang xuất tệp mô hình GLB và OBJ sắc nét…", 90)

        glb_path = os.path.join(item_dir, "model.glb")
        obj_path = os.path.join(item_dir, "model.obj")
        baked_mesh.export(glb_path)
        baked_mesh.export(obj_path)

        return {
            "success": True,
            "glb_path": glb_path,
            "obj_path": obj_path,
            "item_dir": item_dir,
            "engine_used": "🌐 Hugging Face Free Cloud (0đ)"
        }
