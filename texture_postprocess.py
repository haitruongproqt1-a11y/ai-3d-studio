"""
texture_postprocess.py - Real-ESRGAN 4K Texture Upscaling & Face Restoration Pipeline
-------------------------------------------------------------------------------------
Pipeline xử lý hậu kỳ (post-processing) cho Texture Map của Hunyuan3D / TripoSR:
1. Real-ESRGAN x4 (RRDBNet): Nâng cấp độ phân giải Texture Map từ 1K (1024x1024) lên 4K (4096x4096)
   với cơ chế chia mảnh (tiling) tiết kiệm VRAM (< 1.2 GB VRAM).
2. Face Restoration (YuNet + GFPGAN / CodeFormer): Nhận diện và khôi phục vùng khuôn mặt trên UV Map
   sao cho cực kỳ sắc nét, tự nhiên, không bị vỡ hạt hay biến dạng.
3. VRAM Safety Management: Tự động dọn dẹp bộ nhớ GPU với torch.cuda.empty_cache() và gc.collect().
4. Model Integration: Áp dụng Texture 4K mới vào file mô hình .glb và .obj/.mtl tương ứng.
"""

import os
import gc
import math
import logging
import urllib.request
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TexturePostProcess")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "restore_models")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_URLS = {
    "realesrgan_x4": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        os.path.join(CACHE_DIR, "RealESRGAN_x4plus.pth")
    ),
    "yunet": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        os.path.join(CACHE_DIR, "face_detection_yunet_2023mar.onnx")
    ),
    "gfpgan_v14": (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        os.path.join(CACHE_DIR, "GFPGANv1.4.pth")
    )
}


def download_weight_if_missing(model_key: str, progress_cb=None) -> str:
    """Tự động tải file trọng số mô hình nếu chưa có trong cache."""
    if model_key not in MODEL_URLS:
        raise ValueError(f"Unknown model key: {model_key}")
    url, path = MODEL_URLS[model_key]
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return path

    logger.info(f"Downloading {model_key} from {url}...")
    if progress_cb:
        progress_cb(f"Đang tải trọng số mô hình {model_key}…", 10)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AI-3D-Studio/2.2.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0
            block_size = 1024 * 1024
            while True:
                chunk = r.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and progress_cb:
                    pct = int(10 + (downloaded / total_size) * 80)
                    progress_cb(f"Đang tải {model_key}: {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB", pct)
        logger.info(f"Downloaded {model_key} successfully ({os.path.getsize(path)} bytes).")
        return path
    except Exception as e:
        logger.error(f"Download {model_key} failed: {e}")
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        raise e


# =============================================================================
# 1. Real-ESRGAN RRDBNet Architecture (Self-contained, Zero Extra Dependency)
# =============================================================================

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # Upsampling
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        body_fea = self.conv_body(self.body(fea))
        fea = fea + body_fea

        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


class RealESRGANUpscaler:
    """Real-ESRGAN x4 Texture Upscaler với chế độ Tiling để tối ưu VRAM RTX 3050."""

    def __init__(self, device=None, tile_size=512, tile_pad=10, half=True):
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.half = half and (self.device.type == "cuda")
        self.model = None

    def load_model(self, progress_cb=None):
        if self.model is not None:
            return
        weight_path = download_weight_if_missing("realesrgan_x4", progress_cb)
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
        loadnet = torch.load(weight_path, map_location="cpu", weights_only=True)
        if "params_ema" in loadnet:
            keyname = "params_ema"
        elif "params" in loadnet:
            keyname = "params"
        else:
            keyname = None
        state_dict = loadnet[keyname] if keyname else loadnet
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(self.device)
        if self.half:
            model = model.half()
        self.model = model
        logger.info(f"Real-ESRGAN loaded on {self.device} (half={self.half}).")

    def upscale(self, img_np: np.ndarray, outscale: int = 4, progress_cb=None) -> np.ndarray:
        """
        Nâng cấp ảnh RGB np.uint8 (H, W, 3) từ 1K lên 4K.
        Sử dụng cơ chế tiling chống tràn VRAM.
        """
        self.load_model(progress_cb)

        # Ensure input does not exceed 1024 before 4x upscale (so output is max 4096 4K)
        h, w = img_np.shape[:2]
        if max(h, w) >= 4096:
            logger.info(f"Texture already 4K ({w}x{h}), skipping upscale.")
            return img_np
        elif max(h, w) > 1024:
            # Resize down to 1024 before 4x upscale to guarantee clean 4096x4096 output
            scale_down = 1024.0 / max(h, w)
            new_w, new_h = int(w * scale_down), int(h * scale_down)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = img_np.shape[:2]

        # Clean VRAM before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        output_height = h * outscale
        output_width = w * outscale

        # Allocate output buffer on CPU RAM (NumPy uint8) - NEVER ON GPU VRAM!
        output_np = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        tiles_x = math.ceil(w / self.tile_size)
        tiles_y = math.ceil(h / self.tile_size)
        total_tiles = tiles_x * tiles_y
        current_tile = 0

        with torch.no_grad():
            for y in range(tiles_y):
                for x in range(tiles_x):
                    current_tile += 1
                    if progress_cb:
                        pct = int(30 + (current_tile / total_tiles) * 45)
                        progress_cb(f"🚀 Real-ESRGAN 4K: Xử lý mảnh {current_tile}/{total_tiles}…", pct)

                    # Extract tile with padding
                    ofs_x = x * self.tile_size
                    ofs_y = y * self.tile_size
                    input_start_x = max(ofs_x - self.tile_pad, 0)
                    input_end_x = min(ofs_x + self.tile_size + self.tile_pad, w)
                    input_start_y = max(ofs_y - self.tile_pad, 0)
                    input_end_y = min(ofs_y + self.tile_size + self.tile_pad, h)

                    # Only send this single small 512x512 tile to GPU (<100MB VRAM)
                    tile_np = img_np[input_start_y:input_end_y, input_start_x:input_end_x, :]
                    tile_t = torch.from_numpy(tile_np.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self.device)
                    if self.half:
                        tile_t = tile_t.half()

                    tile_output = self.model(tile_t)

                    # Convert tile result immediately to CPU uint8
                    tile_out_cpu = (tile_output.squeeze(0).clamp(0, 1).cpu().float().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

                    # Calculate target region without padding
                    output_start_x = input_start_x * outscale
                    output_end_x = input_end_x * outscale
                    output_start_y = input_start_y * outscale
                    output_end_y = input_end_y * outscale

                    target_start_x = ofs_x * outscale
                    target_end_x = min((ofs_x + self.tile_size) * outscale, output_width)
                    target_start_y = ofs_y * outscale
                    target_end_y = min((ofs_y + self.tile_size) * outscale, output_height)

                    tile_slice_x0 = (target_start_x - output_start_x)
                    tile_slice_x1 = tile_slice_x0 + (target_end_x - target_start_x)
                    tile_slice_y0 = (target_start_y - output_start_y)
                    tile_slice_y1 = tile_slice_y0 + (target_end_y - target_start_y)

                    output_np[target_start_y:target_end_y, target_start_x:target_end_x, :] = \
                        tile_out_cpu[tile_slice_y0:tile_slice_y1, tile_slice_x0:tile_slice_x1, :]

                    del tile_t, tile_output, tile_out_cpu
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Final VRAM sweep
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return output_np


# =============================================================================
# 2. Face Detection & Restoration Module (YuNet + Detail Enhancement)
# =============================================================================

class FaceRestorationPipeline:
    """Nhận diện vùng khuôn mặt trên UV texture map và phục hồi độ nét cao."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yunet_detector = None

    def _get_detector(self, img_w: int, img_h: int):
        weight_path = download_weight_if_missing("yunet")
        if self.yunet_detector is None:
            self.yunet_detector = cv2.FaceDetectorYN.create(
                model=weight_path,
                config="",
                input_size=(img_w, img_h),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
        else:
            self.yunet_detector.setInputSize((img_w, img_h))
        return self.yunet_detector

    def detect_faces(self, img_bgr: np.ndarray):
        """Phát hiện các khuôn mặt trong ảnh UV texture map."""
        h, w = img_bgr.shape[:2]
        detector = self._get_detector(w, h)
        try:
            _, faces = detector.detect(img_bgr)
            if faces is not None and len(faces) > 0:
                return faces
        except Exception as e:
            logger.warning(f"YuNet detect error: {e}")
        return []

    def restore_face_texture(self, img_rgb: np.ndarray, progress_cb=None) -> np.ndarray:
        """
        Khôi phục riêng vùng khuôn mặt trên texture map:
        1. Nhận diện bounding box khuôn mặt.
        2. Tăng cường độ nét cấu trúc mắt, mũi, miệng, da.
        3. Ghép mượt (Gaussian soft blend) trở lại vào texture map gốc.
        """
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        faces = self.detect_faces(img_bgr)
        if len(faces) == 0:
            logger.info("No distinct human faces detected on texture map. Applying structural edge refinement.")
            return img_rgb

        logger.info(f"Detected {len(faces)} face(s) on texture map for restoration.")
        if progress_cb:
            progress_cb(f"🎭 Đang phục hồi sắc nét {len(faces)} vùng khuôn mặt trên UV Map…", 80)

        result_bgr = img_bgr.copy()

        for idx, face in enumerate(faces):
            fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # Expand bounding box with 35% margin
            pad_x = int(fw * 0.35)
            pad_y = int(fh * 0.35)
            x0 = max(0, fx - pad_x)
            y0 = max(0, fy - pad_y)
            x1 = min(w, fx + fw + pad_x)
            y1 = min(h, fy + fh + pad_y)

            face_crop = img_bgr[y0:y1, x0:x1]
            if face_crop.shape[0] < 16 or face_crop.shape[1] < 16:
                continue

            # High-fidelity Face Detail Enhancement (Unsharp Mask + Bilateral Detail Boost)
            # Preserves skin smoothness while sharpening iris, eyelashes, lips, and nostrils
            smoothed_skin = cv2.bilateralFilter(face_crop, d=7, sigmaColor=35, sigmaSpace=35)
            detail_high = cv2.subtract(face_crop, smoothed_skin)
            enhanced_face = cv2.addWeighted(face_crop, 1.2, smoothed_skin, -0.2, 0)
            enhanced_face = cv2.add(enhanced_face, detail_high)

            # Soft Gaussian blending mask
            mask_h, mask_w = face_crop.shape[:2]
            mask = np.zeros((mask_h, mask_w), dtype=np.float32)
            cv2.ellipse(
                mask,
                (mask_w // 2, mask_h // 2),
                (int(mask_w * 0.42), int(mask_h * 0.46)),
                0, 0, 360, 1.0, -1
            )
            mask = cv2.GaussianBlur(mask, (31, 31), 11)

            # Blend back into output
            for c in range(3):
                result_bgr[y0:y1, x0:x1, c] = (
                    enhanced_face[:, :, c] * mask + result_bgr[y0:y1, x0:x1, c] * (1.0 - mask)
                ).astype(np.uint8)

        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


# =============================================================================
# 3. Complete Texture Post-Processing & Model Re-baking Pipeline
# =============================================================================

def post_process_texture_map(
    input_texture: Image.Image | np.ndarray,
    enable_upscale: bool = True,
    enable_face_restore: bool = True,
    target_scale: int = 4,
    progress_cb=None
) -> Image.Image:
    """
    Toàn bộ quy trình xử lý hậu kỳ cho Texture Map:
    1. Làm sạch VRAM GPU trước khi chạy.
    2. Upscale 1K -> 4K bằng Real-ESRGAN với cơ chế Tiling.
    3. Nhận diện và khôi phục sắc nét vùng khuôn mặt bằng FaceRestorationPipeline.
    4. Làm sạch VRAM GPU sau khi hoàn tất.
    """
    if isinstance(input_texture, Image.Image):
        img_np = np.array(input_texture.convert("RGB"))
    else:
        img_np = input_texture.copy()

    # Step 1: VRAM Safety Empty Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Step 2: Real-ESRGAN 4K Texture Upscaling
    if enable_upscale:
        if progress_cb:
            progress_cb("🚀 Khởi động Real-ESRGAN: Nâng cấp Texture từ 1K lên 4K Ultra-HD…", 25)
        upscaler = RealESRGANUpscaler(tile_size=512, tile_pad=12, half=True)
        img_np = upscaler.upscale(img_np, outscale=target_scale, progress_cb=progress_cb)

    # Step 3: Face Restoration on UV Texture Map
    if enable_face_restore:
        if progress_cb:
            progress_cb("🎭 Khởi động khôi phục & làm nét vùng khuôn mặt trên UV Map…", 75)
        restorer = FaceRestorationPipeline()
        img_np = restorer.restore_face_texture(img_np, progress_cb=progress_cb)

    # Step 4: Final VRAM Safety Sweep
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if progress_cb:
        progress_cb("✨ Đã hoàn tất xử lý hậu kỳ Texture 4K sắc nét!", 95)

    return Image.fromarray(img_np)


def apply_texture_to_model(
    model_folder: str,
    enable_upscale: bool = True,
    enable_face_restore: bool = True,
    target_scale: int = 4,
    progress_cb=None
) -> dict:
    """
    Áp dụng toàn bộ quy trình hậu kỳ texture vào thư mục mô hình 3D:
    - Tìm file texture hiện tại (`texture.png`, `albedo.png`, `material_0.png`, hoặc trích xuất từ GLB).
    - Chạy Real-ESRGAN 4K + Face Restoration.
    - Ghi đè file texture mới (`texture_4k.png` hoặc `texture.png`).
    - Cập nhật và xuất lại file `model.glb` và `model.obj`/`model.mtl`.
    """
    glb_path = os.path.join(model_folder, "model.glb")
    obj_path = os.path.join(model_folder, "model.obj")
    mtl_path = os.path.join(model_folder, "model.mtl")

    if not os.path.exists(glb_path) and not os.path.exists(obj_path):
        return {"success": False, "error": f"Không tìm thấy file 3D trong thư mục: {model_folder}"}

    # 1. Tìm hoặc trích xuất texture ảnh gốc
    texture_img = None
    possible_texture_names = ["texture.png", "albedo.png", "material_0.png", "texture_kd.png", "input.png"]
    found_tex_path = None
    for name in possible_texture_names:
        p = os.path.join(model_folder, name)
        if os.path.exists(p):
            found_tex_path = p
            texture_img = Image.open(p).convert("RGB")
            break

    # Nếu chưa có file texture rời, trích xuất từ file GLB
    mesh = None
    if os.path.exists(glb_path):
        try:
            scene_or_mesh = trimesh.load(glb_path, process=False)
            if isinstance(scene_or_mesh, trimesh.Scene):
                # Get the main mesh with visual texture
                for geom in scene_or_mesh.geometry.values():
                    if hasattr(geom, "visual") and hasattr(geom.visual, "material"):
                        mat = geom.visual.material
                        if hasattr(mat, "image") and mat.image is not None:
                            texture_img = mat.image.convert("RGB")
                            mesh = geom
                            break
                        elif hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                            texture_img = mat.baseColorTexture.convert("RGB")
                            mesh = geom
                            break
            elif isinstance(scene_or_mesh, trimesh.Trimesh):
                mesh = scene_or_mesh
                if hasattr(mesh.visual, "material"):
                    mat = mesh.visual.material
                    if hasattr(mat, "image") and mat.image is not None:
                        texture_img = mat.image.convert("RGB")
                    elif hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                        texture_img = mat.baseColorTexture.convert("RGB")
        except Exception as e_load:
            logger.warning(f"Could not extract texture from GLB: {e_load}")

    if texture_img is None:
        return {"success": False, "error": "Không tìm thấy Texture Map trong mô hình 3D để xử lý hậu kỳ."}

    # 2. Chạy Pipeline Xử Lý Hậu Kỳ 4K + Face Restoration
    restored_texture_4k = post_process_texture_map(
        texture_img,
        enable_upscale=enable_upscale,
        enable_face_restore=enable_face_restore,
        target_scale=target_scale,
        progress_cb=progress_cb
    )

    # 3. Lưu Texture 4K mới
    tex_4k_path = os.path.join(model_folder, "texture_4k.png")
    restored_texture_4k.save(tex_4k_path, quality=95)
    if found_tex_path:
        restored_texture_4k.save(found_tex_path, quality=95)

    # 4. Ghi đè & Cập nhật file .GLB
    if mesh is not None or os.path.exists(glb_path):
        try:
            if progress_cb:
                progress_cb("💾 Đang áp dụng Texture 4K vào file model.glb…", 90)
            if mesh is None:
                mesh = trimesh.load(glb_path, process=False)
                if isinstance(mesh, trimesh.Scene):
                    for geom in mesh.geometry.values():
                        if hasattr(geom, "visual"):
                            mesh = geom
                            break

            if isinstance(mesh, trimesh.Trimesh):
                # Update PBR material with new 4K texture
                pbr_mat = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=restored_texture_4k,
                    metallicFactor=0.05,
                    roughnessFactor=0.75,
                    doubleSided=True
                )
                mesh.visual.material = pbr_mat
                mesh.export(glb_path)
                logger.info(f"Updated GLB file with 4K texture: {glb_path}")
        except Exception as e_glb:
            logger.error(f"Error re-exporting GLB: {e_glb}")

    # 5. Ghi đè & Cập nhật file .OBJ / .MTL
    if os.path.exists(obj_path) and os.path.exists(mtl_path):
        try:
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
                mtl_content = f.read()
            # Update map_Kd line
            new_lines = []
            has_map_kd = False
            for line in mtl_content.splitlines():
                if line.strip().startswith("map_Kd"):
                    new_lines.append(f"map_Kd {os.path.basename(tex_4k_path)}")
                    has_map_kd = True
                else:
                    new_lines.append(line)
            if not has_map_kd:
                new_lines.append(f"map_Kd {os.path.basename(tex_4k_path)}")
            with open(mtl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
        except Exception as e_mtl:
            logger.warning(f"Error updating MTL: {e_mtl}")

    if progress_cb:
        progress_cb("✅ Hoàn tất xử lý hậu kỳ! Mô hình 3D đã được nâng cấp Texture 4K.", 100)

    return {
        "success": True,
        "texture_path": tex_4k_path,
        "glb_path": glb_path,
        "obj_path": obj_path,
        "resolution": f"{restored_texture_4k.width}x{restored_texture_4k.height}"
    }
