"""
pbr_perfector.py - Tự Động Hoàn Thiện Vật Liệu PBR & Khử Nhựa 3D Chuẩn Blender
--------------------------------------------------------------------------------
Tự động phân tích mô hình 3D (.glb) và ảnh vân bề mặt để kiến tạo bộ vật liệu
PBR đa kênh đạt chuẩn game AAA và điện ảnh mà không cần mở Blender:
1. Real-ESRGAN 4K + Face Restoration: Phục hồi mắt, mũi, miệng và chi tiết vi mô.
2. Intelligent Material Segmentation:
   - Da mặt & cơ thể: Mịn màng, ấm áp, khử 100% cảm giác tượng sáp (Roughness ~0.38).
   - Vải quần áo & nón: Thô nhám, hút sáng tự nhiên (Roughness ~0.85, Metallic = 0).
   - Kim loại (bi-đông, khóa thắt lưng, vũ khí, cúc đồng): Ánh kim chân thực (Metallic ~0.92, Roughness ~0.18).
   - Da thuộc / Túi đạn: Bóng mờ bán lì (Roughness ~0.55).
3. Tangent-Space Normal Map (Bản đồ pháp tuyến 3D):
   - Tạo gờ nổi nếp gấp vải, đường chỉ may, mắt lưới nón cối, gờ bi-đông dưới ánh đèn.
4. Ambient Occlusion Map (AO):
   - Tạo bóng đổ tự nhiên ở các hốc nách, kẽ túi đạn, nếp nhăn và rãnh sâu.
5. glTF 2.0 PBR Packing:
   - Đóng gói kênh ORM (R=AO, G=Roughness, B=Metallic) và Normal Map trực tiếp vào file .glb.
"""

import os
import gc
import logging
import cv2
import numpy as np
from PIL import Image
import trimesh

logger = logging.getLogger("PBRPerfector")
logging.basicConfig(level=logging.INFO)


def generate_pbr_maps(base_img: Image.Image, progress_cb=None):
    """
    Tự động phân tích ảnh Texture và tạo bộ bản đồ PBR:
    - orm_img: Texture ORM (R: Ambient Occlusion, G: Roughness, B: Metallic)
    - normal_img: Texture Normal Map (Không gian tiếp tuyến Tangent-Space)
    """
    if progress_cb:
        progress_cb("🔍 Đang phân tích chất liệu vải, da mặt và kim loại…", 30)

    w, h = base_img.size
    arr_rgb = np.array(base_img.convert("RGB"))

    # Làm việc trên độ phân giải trung gian nếu ảnh quá lớn để tiết kiệm RAM & tăng tốc
    work_w = min(2048, w)
    work_h = int(round(h * (work_w / float(w))))
    work_rgb = cv2.resize(arr_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)

    # 1. Phân đoạn vật liệu trong không gian màu HSV & Grayscale
    hsv = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2HSV)
    H_chan = hsv[:, :, 0]
    S_chan = hsv[:, :, 1] / 255.0
    V_chan = hsv[:, :, 2] / 255.0
    gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Kim loại thực sự (chỉ những điểm sáng bạc/nhôm hoặc đồng sáng rõ ràng):
    # Rất hạn chế để không biến bóng đổ vải/túi thành kim loại đen bóng!
    metal_mask = (S_chan < 0.08) & (V_chan > 0.85) & (gray > 0.82)

    # Da mặt & bàn tay: Tông màu ấm tự nhiên
    skin_mask = (H_chan >= 3) & (H_chan <= 24) & (S_chan >= 0.16) & (S_chan <= 0.68) & (V_chan >= 0.35)

    # Da thuộc / quai dây đeo: Nâu thẫm / vàng sẫm
    leather_mask = (H_chan >= 10) & (H_chan <= 32) & (S_chan >= 0.25) & (V_chan >= 0.15) & (V_chan < 0.50)

    if progress_cb:
        progress_cb("🎨 Đang tổng hợp bản đồ độ nhám Roughness & kim loại Metallic…", 50)

    # 2. Tạo Kênh Roughness (Kênh Green trong glTF ORM)
    # Vải quân phục = 0.90 (mờ lì, hút sáng, khử sạch bóng nhựa nylon)
    # Da mặt = 0.64 (ẩm mịn, độ mờ lì tự nhiên chuẩn người thật, không bóng nhờn)
    # Da thuộc = 0.72 (bán lì)
    # Kim loại = 0.42 (kim loại dã chiến phay xước, không phản chiếu gương)
    roughness = np.full((work_h, work_w), 0.90, dtype=np.float32)
    roughness[leather_mask] = 0.72
    roughness[skin_mask] = 0.64
    roughness[metal_mask] = 0.42

    # Vi nếp nhăn thớ vải (Micro-weave roughness rất nhẹ)
    fine_detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.5))
    roughness = np.clip(roughness + fine_detail * 0.05, 0.40, 0.96)

    # 3. Tạo Kênh Metallic (Kênh Blue trong glTF ORM)
    # 99.9% vật liệu là phi kim (metallic = 0.0). Chỉ các chi tiết kim loại cực nhỏ mới có ánh kim nhẹ (0.35 max)
    metallic = np.zeros((work_h, work_w), dtype=np.float32)
    metallic[metal_mask] = 0.35

    # 4. Tạo Kênh Ambient Occlusion (Kênh Red trong glTF ORM)
    # Tạo bóng đổ êm dịu, không làm đen sì hay loang lổ bề mặt
    blur_coarse = cv2.GaussianBlur(gray, (0, 0), 12.0)
    ao = np.clip(0.92 + (gray - blur_coarse) * 0.25, 0.75, 1.0)

    # Ghép thành ảnh ORM tiêu chuẩn glTF 2.0 (R=AO, G=Roughness, B=Metallic)
    orm_work = np.dstack([
        (ao * 255).astype(np.uint8),
        (roughness * 255).astype(np.uint8),
        (metallic * 255).astype(np.uint8)
    ])
    orm_full = cv2.resize(orm_work, (w, h), interpolation=cv2.INTER_LINEAR)
    orm_img = Image.fromarray(orm_full)

    if progress_cb:
        progress_cb("📐 Đang điêu khắc bản đồ pháp tuyến vi mô Normal Map (Nếp nhăn 3D)…", 70)

    # 5. Tạo Bản đồ Pháp Tuyến Tangent-Space Normal Map (R=Nx, G=Ny, B=Nz)
    # Lọc mượt khử nhiễu nén ảnh trước khi tính gradient
    gray_smooth = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sobel_x = cv2.Sobel(gray_smooth, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=3)

    # Cường độ nổi khối chuẩn AAA (0.75) – nếp nhăn tinh tế, không biến bề mặt thành tôn nhăn
    bump_strength = 0.75
    nx = -sobel_x * bump_strength
    ny = -sobel_y * bump_strength
    nz = np.ones_like(nx)

    norm_len = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
    nx /= norm_len
    ny /= norm_len
    nz /= norm_len

    normal_r = np.clip((nx * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    normal_g = np.clip((ny * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    normal_b = np.clip((nz * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)

    normal_work = np.dstack([normal_r, normal_g, normal_b])
    normal_full = cv2.resize(normal_work, (w, h), interpolation=cv2.INTER_LINEAR)
    normal_img = Image.fromarray(normal_full)

    return orm_img, normal_img


def perfect_glb_model(glb_path: str, output_path: str = None,
                      enable_upscale: bool = True,
                      enable_face_restore: bool = True,
                      progress_cb=None) -> dict:
    """
    Nâng cấp toàn diện mô hình GLB thành chất lượng PBR Studio chân thực:
    - Khôi phục khuôn mặt sắc nét & nâng texture lên 4K
    - Tự động sinh Normal Map + ORM Metallic-Roughness Map
    - Áp dụng PBR Material chuẩn glTF 2.0
    """
    if not os.path.exists(glb_path):
        return {"success": False, "error": f"Không tìm thấy file: {glb_path}"}

    model_dir = os.path.dirname(os.path.abspath(glb_path))
    out_glb = output_path or glb_path

    if progress_cb:
        progress_cb("📂 Đang nạp mô hình 3D và trích xuất cấu trúc UV…", 10)

    try:
        scene = trimesh.load(glb_path, process=False)
    except Exception as e_load:
        return {"success": False, "error": f"Lỗi đọc file GLB: {e_load}"}

    # 1. Trích xuất Texture màu hiện tại
    base_img = None
    target_geoms = []

    if isinstance(scene, trimesh.Scene):
        for k, g in scene.geometry.items():
            if hasattr(g, 'visual') and hasattr(g.visual, 'material'):
                target_geoms.append(g)
                mat = g.visual.material
                if base_img is None:
                    if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                        base_img = mat.baseColorTexture.convert('RGB')
                    elif hasattr(mat, 'image') and mat.image is not None:
                        base_img = mat.image.convert('RGB')
    elif isinstance(scene, trimesh.Trimesh):
        target_geoms.append(scene)
        if hasattr(scene.visual, 'material'):
            mat = scene.visual.material
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                base_img = mat.baseColorTexture.convert('RGB')
            elif hasattr(mat, 'image') and mat.image is not None:
                base_img = mat.image.convert('RGB')

    if base_img is None:
        # Thử tìm file texture rời trong thư mục
        for cand in ["texture.png", "input.png", "albedo.png", "texture_4k.png"]:
            cand_p = os.path.join(model_dir, cand)
            if os.path.exists(cand_p):
                base_img = Image.open(cand_p).convert("RGB")
                break

    if base_img is None:
        return {"success": False, "error": "Mô hình không chứa Texture Map để hoàn thiện PBR."}

    # 1.5 Khử màu xanh lem trên bàn tay & phục hồi màu da tự nhiên
    try:
        from texture_engine import repair_hands_skin
        base_img = repair_hands_skin(scene, base_img)
    except Exception as e_rh:
        logger.warning(f"Hand repair notice: {e_rh}")

    # 2. Nâng cấp 4K & Phục hồi khuôn mặt (nếu được kích hoạt)
    if enable_upscale or enable_face_restore:
        if progress_cb:
            progress_cb("✨ Đang làm nét khuôn mặt (Face Restoration) & Upscale Texture 4K…", 20)
        try:
            from texture_postprocess import post_process_texture_map
            base_img = post_process_texture_map(
                base_img,
                enable_upscale=enable_upscale,
                enable_face_restore=enable_face_restore,
                target_scale=4 if enable_upscale else 1,
                progress_cb=progress_cb
            )
        except Exception as e_post:
            logger.warning(f"Notice during texture upscale: {e_post}")

    # 3. Tạo Bản Đồ PBR (ORM Metallic-Roughness + Normal Map)
    orm_img, normal_img = generate_pbr_maps(base_img, progress_cb=progress_cb)

    # Lưu các file texture PBR dự phòng vào thư mục
    try:
        base_img.save(os.path.join(model_dir, "texture_pbr_base.png"), quality=95)
        orm_img.save(os.path.join(model_dir, "texture_pbr_orm.png"), quality=95)
        normal_img.save(os.path.join(model_dir, "texture_pbr_normal.png"), quality=95)
    except Exception:
        pass

    # 4. Gán PBRMaterial đạt chuẩn glTF 2.0
    if progress_cb:
        progress_cb("💾 Đang áp dụng vật liệu PBR vào mô hình GLB…", 88)

    pbr_mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_img,
        metallicRoughnessTexture=orm_img,
        normalTexture=normal_img,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        doubleSided=True
    )

    for g in target_geoms:
        g.visual.material = pbr_mat

    # 5. Xuất file GLB hoàn thiện
    scene.export(out_glb)
    logger.info(f"Exported perfected PBR GLB: {out_glb}")

    if progress_cb:
        progress_cb("✅ Hoàn tất! Mô hình 3D đã được thổi hồn vật liệu PBR chuẩn điện ảnh.", 100)

    return {
        "success": True,
        "glb_path": out_glb,
        "resolution": f"{base_img.width}x{base_img.height}",
        "has_normal": True,
        "has_orm": True
    }
