"""
idol_engine.py - Tích hợp Engine Speridlabs IDOL 3D (3D Gaussian Splatting Cloud Free 0đ)
-----------------------------------------------------------------------------------------
Hỗ trợ tạo mô hình 3D Gaussian Splatting (.ply) quang học siêu thực từ 1 ảnh duy nhất,
đồng thời tự động chuyển đổi sang định dạng .glb Mesh tương thích 100% với trình xem
3D trong ứng dụng, Blender, Unity, Unreal Engine và Windows 3D Viewer.
"""

import os
import time
import shutil
import logging
import numpy as np
import trimesh
from PIL import Image

logger = logging.getLogger("IdolEngine")


def convert_splat_ply_to_glb(ply_path: str, out_glb_path: str, out_obj_path: str = None) -> bool:
    """
    Chuyển đổi file 3D Gaussian Splat (.ply) thành file .glb mesh với màu sắc RGB
    chính xác và các mẩu splat có hướng, hiển thị mượt mà trên mọi trình xem glTF / model-viewer.
    """
    try:
        if not os.path.exists(ply_path) or os.path.getsize(ply_path) < 1024:
            logger.error(f"Invalid PLY file: {ply_path}")
            return False

        with open(ply_path, "rb") as f:
            while True:
                line = f.readline().decode("ascii", errors="ignore")
                if "end_header" in line:
                    break
            dt = np.dtype([
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
                ("opacity", "f4"),
                ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
                ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")
            ])
            data = np.fromfile(f, dtype=dt)

        if len(data) == 0:
            return False

        # Sigmoid opacity
        raw_op = data["opacity"]
        op = 1.0 / (1.0 + np.exp(-np.clip(raw_op, -20.0, 20.0)))
        valid = op > 0.15
        if np.sum(valid) < 50:
            valid = op > 0.05
        if np.sum(valid) == 0:
            valid = np.ones(len(data), dtype=bool)

        data = data[valid]
        op = op[valid]
        N = len(data)

        # Trích xuất màu sắc RGB từ bậc 0 hàm sóng cầu Spherical Harmonics
        SH_C0 = 0.28209479177387814
        r = np.clip((SH_C0 * data["f_dc_0"] + 0.5) * 255.0, 0, 255).astype(np.uint8)
        g = np.clip((SH_C0 * data["f_dc_1"] + 0.5) * 255.0, 0, 255).astype(np.uint8)
        b = np.clip((SH_C0 * data["f_dc_2"] + 0.5) * 255.0, 0, 255).astype(np.uint8)
        a = np.clip(op * 255.0, 0, 255).astype(np.uint8)
        colors = np.column_stack([r, g, b, a])

        # Kích thước trục elip của mỗi Gaussian Splat
        s0 = np.clip(np.exp(data["scale_0"]), 1e-4, 0.5)
        s1 = np.clip(np.exp(data["scale_1"]), 1e-4, 0.5)

        # Quaternion to rotation
        q0, q1, q2, q3 = data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]
        q_norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2) + 1e-8
        q0, q1, q2, q3 = q0 / q_norm, q1 / q_norm, q2 / q_norm, q3 / q_norm

        ux = 2.0 * (q1 * q2 + q0 * q3)
        uy = 1.0 - 2.0 * (q1**2 + q3**2)
        uz = 2.0 * (q2 * q3 - q0 * q1)

        vx = 1.0 - 2.0 * (q2**2 + q3**2)
        vy = 2.0 * (q1 * q2 - q0 * q3)
        vz = 2.0 * (q1 * q3 + q0 * q2)

        u_vec = np.column_stack([vx, vy, vz]) * (s0[:, None] * 1.4)
        v_vec = np.column_stack([ux, uy, uz]) * (s1[:, None] * 1.4)

        centers = np.column_stack([data["x"], data["y"], data["z"]])

        v0 = centers - u_vec - v_vec
        v1 = centers + u_vec - v_vec
        v2 = centers + u_vec + v_vec
        v3 = centers - u_vec + v_vec

        verts = np.empty((N * 4, 3), dtype=np.float32)
        verts[0::4] = v0
        verts[1::4] = v1
        verts[2::4] = v2
        verts[3::4] = v3

        vert_colors = np.empty((N * 4, 4), dtype=np.uint8)
        vert_colors[0::4] = colors
        vert_colors[1::4] = colors
        vert_colors[2::4] = colors
        vert_colors[3::4] = colors

        idx = np.arange(N, dtype=np.int32) * 4
        faces = np.empty((N * 2, 3), dtype=np.int32)
        faces[0::2] = np.column_stack([idx, idx + 1, idx + 2])
        faces[1::2] = np.column_stack([idx, idx + 2, idx + 3])

        # Tạo mesh chuẩn glTF
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vert_colors, process=False)
        mesh.export(out_glb_path)

        if out_obj_path:
            try:
                mesh.export(out_obj_path)
            except Exception:
                pass

        return os.path.exists(out_glb_path) and os.path.getsize(out_glb_path) > 1024
    except Exception as e:
        logger.exception(f"convert_splat_ply_to_glb error: {e}")
        return False


def generate_idol_3d(image_path: str, hf_token: str = None, progress_cb=None,
                     item_dir: str = None, steps: int = 30,
                     gaussians: str = "131,072", mode: str = "Real photo") -> dict:
    """
    Gửi ảnh lên Hugging Face Space 'speridlabs/idol' để tạo 3D Gaussian Splatting hoàn toàn miễn phí 0đ.
    """
    if not os.path.exists(image_path):
        return {"success": False, "error": f"Không tìm thấy ảnh: {image_path}"}

    try:
        import gradio_client
        from gradio_client import handle_file
    except ImportError:
        return {"success": False, "error": "Chưa cài đặt gradio_client. Vui lòng cập nhật môi trường!"}

    if not item_dir:
        ts = int(time.time())
        item_dir = os.path.join(r"H:\FULL APP PC\3D MODEL\output_app", f"idol_{ts}")
    os.makedirs(item_dir, exist_ok=True)

    if progress_cb:
        progress_cb("🔮 Đang kết nối tới siêu mô hình Speridlabs IDOL 3D trên ZeroGPU Cloud (0đ)…", 15)

    used_token = (hf_token or "").strip() or None

    try:
        client = gradio_client.Client(
            "speridlabs/idol",
            token=used_token,
            download_files=True,
            verbose=False
        )
    except Exception as e_init:
        logger.warning(f"IDOL Client init with token warning: {e_init}. Retrying anonymous...")
        try:
            client = gradio_client.Client("speridlabs/idol", token=None, download_files=True, verbose=False)
        except Exception as e_anon:
            return {"success": False, "error": f"Không thể kết nối tới máy chủ IDOL: {e_anon}"}

    # Chọn số bước phù hợp (20 - 50)
    step_val = int(np.clip(steps, 20, 50))
    valid_gaussians = ["65,536", "131,072", "262,144", "524,288"]
    g_val = gaussians if gaussians in valid_gaussians else "131,072"
    mode_val = "Real photo" if mode.lower().startswith("real") else "Synthetic"

    if progress_cb:
        progress_cb(f"📤 Đang gửi ảnh vào hàng đợi AI DiT ({g_val} Gaussians, {step_val} bước)…", 25)

    try:
        job = client.submit(
            image=handle_file(image_path),
            mode=mode_val,
            steps=step_val,
            cfg_scale=3.0,
            rescale_t=3.0,
            seed=42,
            randomize=True,
            gaussians=g_val,
            api_name="/run"
        )
    except Exception as e_sub:
        return {"success": False, "error": f"Lỗi gửi tác vụ IDOL: {e_sub}"}

    t0 = time.time()
    poll = 0
    while not job.done():
        time.sleep(1.5)
        poll += 1
        elapsed = int(time.time() - t0)
        if elapsed > 180:
            try:
                job.cancel()
            except Exception:
                pass
            return {"success": False, "error": "Tác vụ IDOL phản hồi quá 180s. Vui lòng thử lại!"}

        try:
            st = job.status()
            code = str(getattr(st, "code", "")).upper()
            rank = getattr(st, "rank", None)
            if "PROCESSING" in code:
                pct = min(85, 30 + poll * 3)
                if progress_cb:
                    progress_cb(f"🔮 AI Diffusion Transformer đang suy luận cấu trúc 3D Gaussian Splats ({pct}%)…", pct)
            elif "STARTING" in code or "IN_QUEUE" in code:
                if progress_cb:
                    r_str = f"#{rank} " if rank else ""
                progress_cb(f"⏳ Đang xếp hàng nhận GPU ZeroGPU A100 {r_str}({elapsed}s)…", min(30, 20 + poll))
            else:
                if progress_cb:
                    progress_cb(f"⚡ Đang giải mã trường xạ quang học 3DGS ({min(88, 30 + poll * 2)}%)…", min(88, 30 + poll * 2))
        except Exception:
            pass

    try:
        res = job.result()
    except Exception as e_res:
        return {"success": False, "error": f"Lỗi tính toán mô hình IDOL: {e_res}"}

    # res is (ply_file_path, seed, meta_html)
    raw_ply = None
    if isinstance(res, (list, tuple)) and len(res) > 0:
        raw_ply = res[0]
    elif isinstance(res, str):
        raw_ply = res

    if not raw_ply or not os.path.exists(raw_ply):
        return {"success": False, "error": f"Không tìm thấy file kết quả 3DGS PLY từ IDOL: {res}"}

    if progress_cb:
        progress_cb("📥 Đang lưu trữ file 3D Gaussian Splatting (.ply) và dựng lưới Mesh (.glb)…", 90)

    dest_ply = os.path.join(item_dir, "model.ply")
    shutil.copy2(raw_ply, dest_ply)

    dest_glb = os.path.join(item_dir, "model.glb")
    dest_obj = os.path.join(item_dir, "model.obj")

    ok_conv = convert_splat_ply_to_glb(dest_ply, dest_glb, dest_obj)
    if not ok_conv or not os.path.exists(dest_glb):
        return {"success": False, "error": "Lỗi chuyển đổi 3D Gaussian Splatting sang GLB."}

    if progress_cb:
        progress_cb("✅ Hoàn tất! Mô hình 3DGS Speridlabs IDOL sẵn sàng (100%).", 100)

    return {
        "success": True,
        "glb_path": dest_glb,
        "ply_path": dest_ply,
        "obj_path": dest_obj,
        "item_dir": item_dir,
        "engine_used": "🔮 Speridlabs IDOL 3D (3D Gaussian Splatting 0đ)"
    }
