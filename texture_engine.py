"""
texture_engine.py - Meshy-Grade 2K 3D-Silhouette-Calibrated Relief & Orthographic PBR Engine (v5.0)
---------------------------------------------------------------------------------------------------
Features:
- Symmetrical Collar & Interior Hole Restoration (`repair_symmetrical_collar`):
  * Automatically un-premultiplies interior semi-transparent holes created by background removal (`rembg`).
  * Restores missing collar/lapel cutouts on humanoid portraits using anatomical neck-to-shoulder symmetry.
- 3D Mesh (X, Y) Anatomical Silhouette Calibration:
  * Directly calibrates the 3D mesh's (X, Y) vertices to match the 2D photo's true anatomical silhouette (`geom_mask`),
    centering the 3D nose ridge onto the 2D head midpoint and aligning shoulder top contours.
- High-Poly Subdivision + Multi-Scale Photometric DoG 3D Relief Sculpting:
  * Removes Marching Cubes voxel staircases via pre-subdivision Taubin smoothing.
  * Subdivides mesh up to ~170k-315k faces and embosses fine 2D photometric details (eyes, nose, lips, badges,
    buttons, pocket flaps, collar insignia, seams) directly into 3D vertex geometry for both Color and Clay modes.
- Pure 1:1 Orthographic 2K UV Projection + Clean Tailored Synthetic Back View:
  * Maps front-facing geometry with zero row-by-row distortion and 100% interior EDT safety guarantee.
"""

import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import trimesh
import logging
from PIL import Image
from trimesh.smoothing import filter_taubin
from trimesh.visual.material import PBRMaterial


def voronoi_pad(img_rgb, mask):
    """
    Extends pure interior foreground colors infinitely into background using Voronoi nearest-neighbor propagation.
    Guarantees zero white/grey/black borders or halo seams.
    """
    if np.all(mask) or not np.any(mask):
        return img_rgb.copy()
    _, indices = ndi.distance_transform_edt(~mask, return_indices=True)
    return img_rgb[indices[0], indices[1]]


def _is_humanoid_portrait(img_rgb, mask):
    """
    Detects whether the foreground subject is a standing humanoid character/portrait.
    """
    coords = np.nonzero(mask)
    if len(coords[0]) < 100:
        return False
    y0, y1 = int(coords[0].min()), int(coords[0].max())
    x0, x1 = int(coords[1].min()), int(coords[1].max())
    h_fg = float(max(1, y1 - y0))
    w_fg = float(max(1, x1 - x0))
    if (h_fg / w_fg) < 1.35:
        return False

    y_cheek0, y_cheek1 = int(y0 + 0.045 * h_fg), int(y0 + 0.115 * h_fg)
    cheek_region = img_rgb[y_cheek0:y_cheek1, :]
    cheek_mask = mask[y_cheek0:y_cheek1, :]
    if not np.any(cheek_mask):
        return False
    cheek_pixels = cheek_region[cheek_mask]
    skin_candidates = cheek_pixels[(cheek_pixels[:, 0] > cheek_pixels[:, 1]) & (cheek_pixels[:, 0] > 85)]
    return len(skin_candidates) >= 0.12 * len(cheek_pixels)


def repair_symmetrical_collar(rgb_orig, orig_alpha):
    """
    Repairs two common `rembg` U2Net matting defects before 3D calibration and texture baking:
    1. Un-premultiplies interior semi-transparent holes where `rembg` dimmed valid foreground pixels.
    2. On humanoid portraits, repairs asymmetric collar/lapel notches where `rembg` bit into one side of the neck/collar.
    """
    H, W = rgb_orig.shape[:2]
    repaired_lapel_mask = np.zeros((H, W), dtype=bool)
    if orig_alpha is None or not np.any(orig_alpha > 100):
        return rgb_orig, orig_alpha, repaired_lapel_mask

    rgb = rgb_orig.astype(np.float32).copy()
    alpha = orig_alpha.astype(np.float32).copy()
    alpha_raw = alpha.copy()

    # 1. Un-premultiply strictly enclosed interior holes anywhere in the foreground
    solid = alpha_raw > 215
    filled = ndi.binary_fill_holes(solid)
    interior_hole = filled & (alpha_raw > 25) & (alpha_raw < 248)
    if np.any(interior_hole):
        rgb[interior_hole] = np.clip(rgb[interior_hole] * (255.0 / alpha_raw[interior_hole, None]), 0, 255)
        alpha[interior_hole] = 255.0

    # 2. Humanoid neck/collar restoration
    if not _is_humanoid_portrait(rgb_orig, alpha_raw > 100):
        return np.clip(rgb, 0, 255).astype(np.uint8), np.clip(alpha, 0, 255).astype(np.uint8), repaired_lapel_mask

    ys, xs = np.where(alpha_raw > 100)
    y0, y1 = int(ys.min()), int(ys.max())
    h_2d = max(1, y1 - y0)

    ny0 = int(y0 + 0.115 * h_2d)
    ny1 = int(y0 + 0.145 * h_2d)
    widths = []
    for y in range(ny0, ny1):
        rxs = np.where(alpha_raw[y] > 230)[0]
        if len(rxs) > 10:
            widths.append((int(rxs[-1] - rxs[0]), y, int(rxs[0]), int(rxs[-1])))
    if not widths:
        return np.clip(rgb, 0, 255).astype(np.uint8), np.clip(alpha, 0, 255).astype(np.uint8), repaired_lapel_mask

    _, y_neck, xL_neck, xR_neck = min(widths, key=lambda t: t[0])

    # Step 2a: Un-premultiply semi-transparent collar cutouts in [y_neck, y0 + 0.195*h_2d]
    ny_bot = min(H, int(y0 + 0.195 * h_2d))
    for y in range(y_neck, ny_bot):
        rxs_hi = np.where(alpha_raw[y] > 215)[0]
        rxs_lo = np.where(alpha_raw[y] > 60)[0]
        if len(rxs_hi) >= 2 and len(rxs_lo) >= 2:
            x_min = int(rxs_hi[0]) + 3
            x_max = int(rxs_lo[-1]) - 1
            if x_max > x_min:
                span = np.arange(x_min, x_max + 1)
                semi = span[(alpha_raw[y, span] > 40) & (alpha_raw[y, span] < 248)]
                for x in semi:
                    u = np.clip(rgb[y, x] * (255.0 / alpha_raw[y, x]), 0, 255)
                    if (x > rxs_hi[0] + 2 and x < rxs_hi[-1] - 1) or (u[0] > 180 and u[1] > 180):
                        rgb[y, x] = u
                        alpha[y, x] = 255.0

    # Step 2b: Restore missing outer jacket lapel if one side has a concave cutout between neck and shoulder
    y_lapel_end = min(H - 1, y_neck + int(0.027 * h_2d))
    rxs_end = np.where(alpha_raw[y_lapel_end] > 230)[0]
    if len(rxs_end) >= 10:
        flare_L_end = max(0, xL_neck - int(rxs_end[0]))
        flare_R_end = max(0, int(rxs_end[-1]) - xR_neck)

        y_ref = int(0.5 * (y_neck + y_lapel_end))
        rxs_ref = np.where(alpha[y_ref] > 235)[0]
        if len(rxs_ref) >= 10:
            if flare_L_end - flare_R_end > 10:
                lapel_samples = rgb[y_ref:y_lapel_end, rxs_ref[0] + 3:min(W, rxs_ref[0] + 18)]
                dark_mask = lapel_samples.mean(axis=2) < 95
                lapel_base_rgb = np.median(lapel_samples[dark_mask], axis=0) if np.any(dark_mask) else np.median(lapel_samples.reshape(-1, 3), axis=0)

                y_range = np.arange(y_neck, min(H, y_lapel_end + 10))
                cur_L_arr = np.zeros(len(y_range), dtype=np.float32)
                cur_R_arr = np.zeros(len(y_range), dtype=np.float32)
                for idx, y in enumerate(y_range):
                    rxs = np.where(alpha[y] > 230)[0]
                    cur_L_arr[idx] = rxs[0] if len(rxs) else xL_neck
                    cur_R_arr[idx] = rxs[-1] if len(rxs) else xR_neck
                cur_R_smooth = ndi.gaussian_filter1d(cur_R_arr, sigma=5.0)

                for idx, y in enumerate(y_range):
                    cur_L = int(round(cur_L_arr[idx]))
                    cur_R = int(round(cur_R_smooth[idx]))
                    t = (y - y_neck) / max(1.0, float(y_lapel_end - y_neck))
                    target_R = int(round(xR_neck + t * flare_L_end))
                    if target_R > cur_R:
                        lapel_w = target_R - cur_R
                        for k in range(-4, lapel_w + 1):
                            dst_x = cur_R + k
                            src_x = np.clip(cur_L + max(3, lapel_w - max(0, k)), cur_L + 3, min(W - 1, cur_L + 26))
                            if 0 <= dst_x < W:
                                if k >= -1 or rgb[y, dst_x].mean() < 185:
                                    cand = rgb[y, src_x]
                                    rgb[y, dst_x] = cand if cand.mean() < 95 else lapel_base_rgb
                                    alpha[y, dst_x] = 255.0
                                    repaired_lapel_mask[y, dst_x] = True
                        alpha[y, cur_L + 3:min(W, target_R + 1)] = 255.0
                        repaired_lapel_mask[y, max(0, cur_R - 6):min(W, target_R + 1)] = True

            elif flare_R_end - flare_L_end > 10:
                lapel_samples = rgb[y_ref:y_lapel_end, max(0, rxs_ref[-1] - 18):max(1, rxs_ref[-1] - 3)]
                dark_mask = lapel_samples.mean(axis=2) < 95
                lapel_base_rgb = np.median(lapel_samples[dark_mask], axis=0) if np.any(dark_mask) else np.median(lapel_samples.reshape(-1, 3), axis=0)

                y_range = np.arange(y_neck, min(H, y_lapel_end + 10))
                cur_L_arr = np.zeros(len(y_range), dtype=np.float32)
                cur_R_arr = np.zeros(len(y_range), dtype=np.float32)
                for idx, y in enumerate(y_range):
                    rxs = np.where(alpha[y] > 230)[0]
                    cur_L_arr[idx] = rxs[0] if len(rxs) else xL_neck
                    cur_R_arr[idx] = rxs[-1] if len(rxs) else xR_neck
                cur_L_smooth = ndi.gaussian_filter1d(cur_L_arr, sigma=5.0)

                for idx, y in enumerate(y_range):
                    cur_L = int(round(cur_L_smooth[idx]))
                    cur_R = int(round(cur_R_arr[idx]))
                    t = (y - y_neck) / max(1.0, float(y_lapel_end - y_neck))
                    target_L = int(round(xL_neck - t * flare_R_end))
                    if target_L < cur_L:
                        lapel_w = cur_L - target_L
                        for k in range(-4, lapel_w + 1):
                            dst_x = cur_L - k
                            src_x = np.clip(cur_R - max(3, lapel_w - max(0, k)), max(0, cur_R - 26), cur_R - 3)
                            if 0 <= dst_x < W:
                                if k >= -1 or rgb[y, dst_x].mean() < 185:
                                    cand = rgb[y, src_x]
                                    rgb[y, dst_x] = cand if cand.mean() < 95 else lapel_base_rgb
                                    alpha[y, dst_x] = 255.0
                                    repaired_lapel_mask[y, dst_x] = True
                        alpha[y, max(0, target_L):min(W, cur_R - 3)] = 255.0
                        repaired_lapel_mask[y, max(0, target_L):min(W, cur_L + 6)] = True

    return np.clip(rgb, 0, 255).astype(np.uint8), np.clip(alpha, 0, 255).astype(np.uint8), repaired_lapel_mask


def extract_dual_masks(img_rgb, pil_img, orig_alpha=None):
    """
    Extracts two specialized masks at original resolution (after collar & interior hole restoration):
    1. geom_mask (alpha > 128, un-eroded): true geometric silhouette preserving hair top, ears, collar & shoe tips.
    2. color_mask (alpha > 235, eroded ~0.35%): pure interior color mask completely free of rembg grey background fringe.
    """
    h, w = img_rgb.shape[:2]
    erode_iters = max(3, int(round(max(h, w) * 0.0035)))
    alpha = None

    if orig_alpha is not None and np.min(orig_alpha) < 200 and np.any(orig_alpha > 50):
        alpha = orig_alpha
    else:
        try:
            if isinstance(pil_img, Image.Image) and pil_img.mode == "RGBA":
                arr_a = np.array(pil_img)[:, :, 3]
                if np.min(arr_a) < 200 and np.any(arr_a > 50):
                    alpha = arr_a
            if alpha is None:
                import rembg
                clean_rgba = rembg.remove(pil_img)
                arr = np.array(clean_rgba)
                if arr.shape[2] == 4:
                    alpha = arr[:, :, 3]
        except Exception as e:
            logging.warning(f"Foreground alpha extraction warning: {e}")

    if alpha is not None:
        img_rgb_repaired, alpha_repaired, repaired_lapel_mask = repair_symmetrical_collar(img_rgb, alpha)
        img_rgb[:] = img_rgb_repaired
        geom_mask = (alpha_repaired > 128)
        raw_color_mask = (alpha_repaired > 235) if np.any(alpha_repaired > 235) else geom_mask
        color_mask = ndi.binary_erosion(raw_color_mask, iterations=erode_iters) | repaired_lapel_mask
        if not np.any(color_mask):
            color_mask = geom_mask
        return geom_mask, color_mask

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    corners = [gray[0:10, 0:10], gray[0:10, -10:], gray[-10:, 0:10], gray[-10:, -10:]]
    bg_val = np.median([np.median(c) for c in corners])
    diff = np.abs(gray.astype(np.int16) - bg_val)
    geom_mask = (diff > 18)
    color_mask = ndi.binary_erosion(geom_mask, iterations=erode_iters)
    if not np.any(color_mask):
        color_mask = geom_mask
    return geom_mask, color_mask


def clean_and_pad_view(rgb_orig, geom_mask_orig, color_mask_orig, target_w, target_h):
    """
    Strips background halo at original resolution via Voronoi extrapolation from color_mask_orig,
    resizes to (target_w, target_h), and applies horizontal scanline padding.
    """
    rgb_voronoi = voronoi_pad(rgb_orig, color_mask_orig)
    view_rgb = cv2.resize(rgb_voronoi, (target_w, target_h), interpolation=cv2.INTER_AREA)
    geom_mask = cv2.resize(geom_mask_orig.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
    color_mask = cv2.resize(color_mask_orig.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0

    padded = view_rgb.copy()
    for y in range(target_h):
        xs = np.nonzero(color_mask[y])[0]
        if len(xs) >= 4:
            xl, xr = xs[0], xs[-1]
            c_left = np.median(view_rgb[y, xl:min(target_w, xl + 6)], axis=0).astype(np.uint8)
            c_right = np.median(view_rgb[y, max(0, xr - 5):xr + 1], axis=0).astype(np.uint8)
            padded[y, :xl] = c_left
            padded[y, xr + 1:] = c_right

    padded = cv2.GaussianBlur(padded, (5, 5), 0)
    padded[color_mask] = view_rgb[color_mask]
    return padded, geom_mask, color_mask


def synthesize_clean_back_view(front_padded, geom_mask_f):
    """
    Synthesizes a clean, realistic back view from the front view when no back photo is provided.
    """
    H, W = front_padded.shape[:2]
    back_padded = cv2.flip(front_padded, 1)
    mask_b = cv2.flip(geom_mask_f.astype(np.uint8), 1) > 0

    if not _is_humanoid_portrait(back_padded, mask_b):
        return back_padded, mask_b

    coords = np.nonzero(mask_b)
    y0, y1 = int(coords[0].min()), int(coords[0].max())
    h_fg = float(max(1, y1 - y0))

    y_cheek0, y_cheek1 = int(y0 + 0.055 * h_fg), int(y0 + 0.110 * h_fg)
    cheek_pixels = back_padded[y_cheek0:y_cheek1, :][mask_b[y_cheek0:y_cheek1, :]]
    skin_candidates = cheek_pixels[(cheek_pixels[:, 0] > cheek_pixels[:, 1]) & (cheek_pixels[:, 0] > 85)]
    skin_color = np.median(skin_candidates, axis=0) if len(skin_candidates) > 0 else np.array([195, 155, 140])

    # 1. Back of head & nape of neck [y0 + 0.015*h_fg .. y0 + 0.130*h_fg]
    y_crown0, y_crown1 = int(y0 + 0.005 * h_fg), int(y0 + 0.035 * h_fg)
    crown_pixels = back_padded[y_crown0:y_crown1, :][mask_b[y_crown0:y_crown1, :]]
    hair_color = np.percentile(crown_pixels, 20, axis=0) if len(crown_pixels) > 0 else np.array([25, 25, 28])

    y_face0, y_neck1 = int(y0 + 0.015 * h_fg), int(y0 + 0.130 * h_fg)
    head_overlay = back_padded.copy().astype(np.float32)
    for y in range(y_face0, y_neck1):
        xs = np.nonzero(mask_b[y])[0]
        xl, xr = (xs[0], xs[-1]) if len(xs) >= 2 else (int(W * 0.35), int(W * 0.65))
        w_row = max(10.0, float(xr - xl))
        x_norm = (np.arange(W) - (xl + xr) / 2.0) / (w_row * 0.5)
        shade = 1.02 - 0.15 * np.clip(x_norm ** 2, 0.0, 1.2)
        t_neck = np.clip((y - (y0 + 0.065 * h_fg)) / (0.022 * h_fg), 0.0, 1.0)
        base_c = (1.0 - t_neck) * hair_color + t_neck * skin_color
        head_overlay[y, :] = np.clip(base_c[None, :] * shade[:, None], 0, 255)

    head_alpha = np.zeros((H, W), dtype=np.float32)
    head_alpha[y_face0:y_neck1, :] = 1.0
    k_head = max(3, int(round(h_fg * 0.018)) | 1)
    head_alpha = cv2.GaussianBlur(head_alpha, (k_head, k_head), 0)[:, :, None]
    back_padded = (head_alpha * head_overlay + (1.0 - head_alpha) * back_padded).astype(np.uint8)

    # 2. Back of collar & jacket torso [y0 + 0.122*h_fg .. y0 + 0.510*h_fg]
    y_tor0, y_tor1 = int(y0 + 0.122 * h_fg), int(y0 + 0.510 * h_fg)
    y_s0, y_s1 = int(y0 + 0.22 * h_fg), int(y0 + 0.29 * h_fg)
    tor_sample = back_padded[y_s0:y_s1, :][mask_b[y_s0:y_s1, :]]
    garment_color = np.percentile(tor_sample, 35, axis=0) if len(tor_sample) > 0 else np.array([20, 55, 48])

    tor_region = back_padded[y_tor0:y_tor1, :].astype(np.float32)
    coat_synth = np.zeros_like(tor_region)
    blend_mask = np.zeros((y_tor1 - y_tor0, W), dtype=np.float32)

    for ry in range(y_tor1 - y_tor0):
        y = y_tor0 + ry
        xs = np.nonzero(mask_b[y])[0]
        xl, xr = (xs[0], xs[-1]) if len(xs) >= 2 else (int(W * 0.15), int(W * 0.85))
        w_row = max(20.0, float(xr - xl))
        xm = (xl + xr) / 2.0
        x_norm = (np.arange(W) - xm) / (w_row * 0.5)
        cyl_shade = 1.04 - 0.16 * np.clip(x_norm ** 2, 0.0, 1.3) - 0.05 * np.exp(-((np.arange(W) - xm) / max(3.0, w_row * 0.015)) ** 2)
        coat_synth[ry, :] = np.clip(garment_color[None, :] * cyl_shade[:, None], 0, 255)

        if y < y0 + 0.168 * h_fg:
            blend_mask[ry, :] = 1.0
        elif y < y0 + 0.185 * h_fg:
            c_l, c_r = int(xm - 0.34 * w_row), int(xm + 0.34 * w_row)
            blend_mask[ry, max(0, c_l):min(W, c_r)] = 1.0
            c_dist = np.linalg.norm(tor_region[ry] - garment_color[None, :], axis=1)
            mid_zone = (np.arange(W) > (xm - 0.38 * w_row)) & (np.arange(W) < (xm + 0.38 * w_row))
            blend_mask[ry, mid_zone & (c_dist > 18.0)] = 1.0
        else:
            c_l, c_r = int(xm - 0.38 * w_row), int(xm + 0.38 * w_row)
            blend_mask[ry, max(0, c_l):min(W, c_r)] = 1.0
            c_dist = np.linalg.norm(tor_region[ry] - garment_color[None, :], axis=1)
            blend_mask[ry, c_dist > 18.0] = 1.0

    k_tor = max(3, int(round(h_fg * 0.022)) | 1)
    blend_mask = cv2.GaussianBlur(blend_mask, (k_tor, k_tor), 0)[:, :, None]
    back_padded[y_tor0:y_tor1, :] = (blend_mask * coat_synth + (1.0 - blend_mask) * tor_region).astype(np.uint8)
    return back_padded, mask_b


def create_clay_sculpture_mesh(mesh, normal_map_pil=None):
    """
    Transforms 3D mesh into a pristine, untextured plaster/clay sculpture.
    Optimal for 3D printing and manual painting in Blender / ZBrush.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])

    clay = mesh.copy()
    clay.visual = trimesh.visual.ColorVisuals(mesh=clay)
    clay.visual.vertex_colors = np.full((len(clay.vertices), 4), [235, 232, 226, 255], dtype=np.uint8)
    return clay


def _load_image_rgb_and_alpha(source):
    orig_alpha = None
    if isinstance(source, str):
        raw_pil = Image.open(source)
        if raw_pil.mode == "RGBA":
            orig_alpha = np.array(raw_pil)[:, :, 3]
        pil_img = raw_pil.convert("RGB")
        rgb = np.array(pil_img)
    elif isinstance(source, Image.Image):
        if source.mode == "RGBA":
            orig_alpha = np.array(source)[:, :, 3]
        pil_img = source.convert("RGB")
        rgb = np.array(pil_img)
    elif isinstance(source, np.ndarray):
        if source.shape[-1] == 4:
            orig_alpha = source[:, :, 3]
            rgb = source[:, :, :3].copy()
        else:
            rgb = source.copy()
        pil_img = Image.fromarray(rgb)
    else:
        raise ValueError("Unsupported image source type")
    return rgb, pil_img, orig_alpha


def _calibrate_mesh_and_sculpt_relief(mesh, front_padded, geom_mask_f):
    """
    Meshy-Grade 3D Geometry Enhancement Pipeline:
    1. Pre-subdivision Taubin smoothing removes Marching Cubes voxel staircases.
    2. 3D (X, Y) Silhouette Calibration aligns the 3D mesh silhouette & nose ridge 1:1 with the 2D photo.
    3. High-Poly Subdivision + Multi-Scale Photometric DoG 3D Relief Sculpting embosses fine 2D details
       (facial features, badges, buttons, pockets, seams) directly into the 3D mesh surface.
    Returns:
       (mesh_sculpted, ortho_params)
    """
    H, W = geom_mask_f.shape[:2]
    coords = np.nonzero(geom_mask_f)
    if len(coords[0]) < 50:
        return mesh, None

    y0_2d, y1_2d = float(coords[0].min()), float(coords[0].max())
    x0_2d, x1_2d = float(coords[1].min()), float(coords[1].max())
    h_2d = max(1.0, y1_2d - y0_2d)
    x_mid_2d = 0.5 * (x0_2d + x1_2d)

    # 1. Remove Marching Cubes voxel staircases prior to subdivision
    try:
        filter_taubin(mesh, lamb=0.45, nu=-0.48, iterations=3)
    except Exception as e:
        logging.warning(f"Taubin pre-smoothing notice: {e}")

    v = mesh.vertices.copy()
    y0_3d = float(np.percentile(v[:, 1], 0.05))
    y1_3d = float(np.percentile(v[:, 1], 99.95))
    h_3d = max(1e-4, y1_3d - y0_3d)
    px_per_unit = h_2d / h_3d
    x_mid_3d = 0.5 * (float(np.percentile(v[:, 0], 0.5)) + float(np.percentile(v[:, 0], 99.5)))

    is_humanoid = _is_humanoid_portrait(front_padded, geom_mask_f)

    # 2. Calibrate 3D (X, Y) Silhouette to match 2D Photo Silhouette
    s_3d = np.clip((y1_3d - v[:, 1]) / h_3d, 0.0, 1.0)

    if is_humanoid:
        # 2a. Center 3D Head Nose-Ridge onto 2D Head Midpoint
        head_front_mask = (s_3d > 0.03) & (s_3d < 0.13)
        if np.any(head_front_mask):
            z_head_cut = np.percentile(v[head_front_mask, 2], 70.0)
            nose_pts = v[head_front_mask & (v[:, 2] >= z_head_cut)]
            nose_x_3d = float(np.median(nose_pts[:, 0])) if len(nose_pts) > 10 else x_mid_3d
        else:
            nose_x_3d = x_mid_3d

        hy0, hy1 = int(y0_2d + 0.02 * h_2d), int(y0_2d + 0.12 * h_2d)
        head_xs = np.nonzero(geom_mask_f[hy0:hy1, :])[1]
        head_mid_2d = 0.5 * (float(np.percentile(head_xs, 1.0)) + float(np.percentile(head_xs, 99.0))) if len(head_xs) > 10 else x_mid_2d
        target_nose_3d = x_mid_3d + (head_mid_2d - x_mid_2d) / px_per_unit

        head_weight = np.clip((0.15 - s_3d) / 0.05, 0.0, 1.0)
        v[:, 0] = v[:, 0] - head_weight * (nose_x_3d - target_nose_3d)

    # 2b. Scanline Horizontal Envelope Alignment (N = 256 slices)
    N = 256
    s_grid = np.linspace(0.0, 1.0, N)
    x2_min_3d = np.zeros(N)
    x2_max_3d = np.zeros(N)
    x3_min_s = np.zeros(N)
    x3_max_s = np.zeros(N)

    for i in range(N):
        frac = s_grid[i]
        y2 = int(round(y0_2d + frac * h_2d))
        r0, r1 = max(0, y2 - 3), min(H, y2 + 4)
        xs2 = np.nonzero(geom_mask_f[r0:r1, :])[1]
        if len(xs2) > 0:
            x2_min_3d[i] = x_mid_3d + (np.percentile(xs2, 0.5) - x_mid_2d) / px_per_unit
            x2_max_3d[i] = x_mid_3d + (np.percentile(xs2, 99.5) - x_mid_2d) / px_per_unit
        elif i > 0:
            x2_min_3d[i], x2_max_3d[i] = x2_min_3d[i - 1], x2_max_3d[i - 1]

        y3 = y1_3d - frac * h_3d
        sl3 = v[np.abs(v[:, 1] - y3) < h_3d * 0.012]
        if len(sl3) > 0:
            x3_min_s[i] = np.percentile(sl3[:, 0], 0.5)
            x3_max_s[i] = np.percentile(sl3[:, 0], 99.5)
        elif i > 0:
            x3_min_s[i], x3_max_s[i] = x3_min_s[i - 1], x3_max_s[i - 1]

    sigma_env = 6.0 if is_humanoid else 10.0
    x2_min_3d = ndi.gaussian_filter1d(x2_min_3d, sigma=sigma_env)
    x2_max_3d = ndi.gaussian_filter1d(x2_max_3d, sigma=sigma_env)
    x3_min_s = ndi.gaussian_filter1d(x3_min_s, sigma=sigma_env)
    x3_max_s = ndi.gaussian_filter1d(x3_max_s, sigma=sigma_env)

    x3_mid_s = 0.5 * (x3_min_s + x3_max_s)
    x2_mid_3d = 0.5 * (x2_min_3d + x2_max_3d)

    if is_humanoid:
        for i in range(int(0.14 * N)):
            w_h = np.clip((0.14 - s_grid[i]) / 0.04, 0.0, 1.0)
            x3_mid_s[i] = w_h * target_nose_3d + (1.0 - w_h) * x3_mid_s[i]
            x2_mid_3d[i] = w_h * target_nose_3d + (1.0 - w_h) * x2_mid_3d[i]

    v_x3_l = np.interp(s_3d, s_grid, x3_min_s)
    v_x3_m = np.interp(s_3d, s_grid, x3_mid_s)
    v_x3_r = np.interp(s_3d, s_grid, x3_max_s)
    v_x2_l = np.interp(s_3d, s_grid, x2_min_3d)
    v_x2_m = np.interp(s_3d, s_grid, x2_mid_3d)
    v_x2_r = np.interp(s_3d, s_grid, x2_max_3d)

    is_left = v[:, 0] <= v_x3_m
    tl = np.clip((v[:, 0] - v_x3_l) / np.maximum(1e-4, v_x3_m - v_x3_l), 0.0, 1.15)
    tr = np.clip((v[:, 0] - v_x3_m) / np.maximum(1e-4, v_x3_r - v_x3_m), 0.0, 1.15)
    v[:, 0] = np.where(is_left, v_x2_l + tl * (v_x2_m - v_x2_l), v_x2_m + tr * (v_x2_r - v_x2_m))

    if is_humanoid:
        # 2c. Column-Wise Top Silhouette Alignment for Shoulders/Collar (s_3d in [0.11, 0.35])
        M_col = 180
        xg = np.linspace(float(v[:, 0].min()), float(v[:, 0].max()), M_col)
        y2_top_3d = np.full(M_col, y1_3d)
        y3_top_3d = np.full(M_col, y1_3d)

        upper_v = v[s_3d < 0.26]
        for j in range(M_col):
            px = int(round(x_mid_2d + (xg[j] - x_mid_3d) * px_per_unit))
            c0, c1 = max(0, px - 2), min(W, px + 3)
            ys2 = np.nonzero(geom_mask_f[:int(y0_2d + 0.26 * h_2d), c0:c1])[0]
            if len(ys2) > 0:
                y2_top_3d[j] = y1_3d - (float(ys2.min()) - y0_2d) / px_per_unit
            elif j > 0:
                y2_top_3d[j] = y2_top_3d[j - 1]

            col_v = upper_v[np.abs(upper_v[:, 0] - xg[j]) < 0.015]
            if len(col_v) > 0:
                y3_top_3d[j] = np.percentile(col_v[:, 1], 99.0)
            elif j > 0:
                y3_top_3d[j] = y3_top_3d[j - 1]

        dy_top = ndi.gaussian_filter1d(y2_top_3d - y3_top_3d, sigma=5.0)
        head_col_mask = np.abs(xg - target_nose_3d) < (0.075 * h_3d)
        dy_top[head_col_mask] = 0.0
        dy_top = ndi.gaussian_filter1d(dy_top, sigma=4.0)

        v_dy_top = np.interp(v[:, 0], xg, dy_top)
        sh_y_weight = np.clip((s_3d - 0.10) / 0.04, 0.0, 1.0) * np.clip((0.36 - s_3d) / 0.12, 0.0, 1.0)
        v[:, 1] = v[:, 1] + sh_y_weight * v_dy_top

        # Post-Y horizontal silhouette bound clamp so no shoulder/collar vertex protrudes past geom_mask_f at its new Y
        s_3d_new = np.clip((y1_3d - v[:, 1]) / h_3d, 0.0, 1.0)
        v[:, 0] = np.clip(
            v[:, 0],
            np.interp(s_3d_new, s_grid, x2_min_3d),
            np.interp(s_3d_new, s_grid, x2_max_3d)
        )

    mesh.vertices = v

    # 3. High-Poly Mesh Subdivision (~160k - 315k faces like Meshy Flagship)
    if len(mesh.faces) < 120000:
        try:
            mesh = mesh.subdivide()
        except Exception as e:
            logging.warning(f"Mesh subdivision notice: {e}")

    # 4. Multi-Scale Photometric Difference-of-Gaussians (DoG) 3D Relief Sculpting
    v_sub = mesh.vertices.copy()
    vn_sub = mesh.vertex_normals.copy()

    gray_f = cv2.cvtColor(front_padded, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g_fine = ndi.gaussian_filter(gray_f, sigma=0.8) - ndi.gaussian_filter(gray_f, sigma=3.5)
    g_mid = ndi.gaussian_filter(gray_f, sigma=2.0) - ndi.gaussian_filter(gray_f, sigma=10.0)
    contrast = np.abs(gray_f - ndi.gaussian_filter(gray_f, sigma=5.0))
    bilateral_w = 1.0 / (1.0 + (contrast / 0.18) ** 2)
    relief_2d = np.clip((0.65 * g_fine + 0.35 * g_mid) * bilateral_w, -0.16, 0.16)

    # Fade relief smoothly to 0 within 6px of silhouette edge so outer silhouette never ripples
    dist_in = ndi.distance_transform_edt(geom_mask_f)
    edge_taper = np.clip((dist_in - 2.0) / 6.0, 0.0, 1.0)
    relief_2d = relief_2d * edge_taper

    fx_sub = np.clip(x_mid_2d + (v_sub[:, 0] - x_mid_3d) * px_per_unit, 0.0, W - 1.0)
    fy_sub = np.clip(y0_2d + (y1_3d - v_sub[:, 1]) * px_per_unit, 0.0, H - 1.0)

    sampled_relief = ndi.map_coordinates(relief_2d, [fy_sub, fx_sub], order=1, mode="nearest")

    # Apply relief strictly to front-facing surface vertices
    z_min_sub = float(np.percentile(v_sub[:, 2], 2.0))
    z_max_sub = float(np.percentile(v_sub[:, 2], 98.0))
    z_rel_sub = np.clip((v_sub[:, 2] - z_min_sub) / max(0.05, z_max_sub - z_min_sub), 0.0, 1.0)
    front_w = np.clip((vn_sub[:, 2] - 0.15) / 0.45, 0.0, 1.0) * np.clip((z_rel_sub - 0.35) / 0.25, 0.0, 1.0)

    disp_amp = 0.0082 * h_3d
    disp = (sampled_relief * front_w * disp_amp)[:, None] * vn_sub
    disp[:, 2] *= 1.35
    v_sub = v_sub + disp
    mesh.vertices = v_sub

    try:
        filter_taubin(mesh, lamb=0.32, nu=-0.34, iterations=1)
    except Exception:
        pass

    ortho_params = (x_mid_2d, y0_2d, y1_2d, x_mid_3d, y0_3d, y1_3d, px_per_unit)
    return mesh, ortho_params


def _compute_orthographic_uv_mapping(verts, geom_mask, ortho_params, is_back=False):
    """
    Pure 1:1 Orthographic UV Projection for calibrated 3D meshes, paired with a 2px EDT interior
    safety pull-in so 100.0% of vertices sample strictly inside `geom_mask` with zero background bleed.
    """
    H, W = geom_mask.shape[:2]
    x_mid_2d, y0_2d, _, x_mid_3d, _, y1_3d, px_per_unit = ortho_params

    vx = -verts[:, 0] if is_back else verts[:, 0]
    fx = x_mid_2d + (vx - x_mid_3d) * px_per_unit
    fy = y0_2d + (y1_3d - verts[:, 1]) * px_per_unit

    # Smooth EDT boundary pull-in guarantees 100% of vertices land strictly inside geom_mask
    _, (near_y, near_x) = ndi.distance_transform_edt(~geom_mask, return_indices=True)
    dx_edt = ndi.gaussian_filter((near_x - np.arange(W)[None, :]).astype(np.float32), sigma=1.5)
    dy_edt = ndi.gaussian_filter((near_y - np.arange(H)[:, None]).astype(np.float32), sigma=1.5)

    cy = np.clip(fy, 0, H - 1)
    cx = np.clip(fx, 0, W - 1)
    fx = fx + ndi.map_coordinates(dx_edt, [cy, cx], order=1, mode="nearest")
    fy = fy + ndi.map_coordinates(dy_edt, [cy, cx], order=1, mode="nearest")

    iy = np.clip(np.round(fy).astype(int), 0, H - 1)
    ix = np.clip(np.round(fx).astype(int), 0, W - 1)
    out = ~geom_mask[iy, ix]
    if np.any(out):
        fx[out] = near_x[iy[out], ix[out]]
        fy[out] = near_y[iy[out], ix[out]]

    return fx, fy


def bake_meshy_pbr_mesh(mesh, image_source=None, back_image_source=None,
                        left_image_source=None, right_image_source=None,
                        color_mode="color"):
    """
    Applies Meshy-Grade 3D Silhouette Calibration, High-Poly Relief Subdivision, and 1:1 Orthographic 2K PBR Texture.
    - Works for BOTH `color_mode == 'color'` and `color_mode == 'clay'` when `image_source` is provided.
    - Guarantees 100% undistorted alignment of facial features, collar, badges, and shoulders.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    else:
        mesh = mesh.copy()

    try:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
    except Exception:
        pass

    if image_source is None:
        return create_clay_sculpture_mesh(mesh), None

    # 1. Load Front Image, Restore Collar/Holes & Extract Dual Masks
    rgb_orig, pil_img, orig_alpha = _load_image_rgb_and_alpha(image_source)
    geom_mask_orig, color_mask_orig = extract_dual_masks(rgb_orig, pil_img, orig_alpha=orig_alpha)

    h_raw, w_raw = rgb_orig.shape[:2]
    max_dim = 2048.0
    scale_down = (max_dim / max(h_raw, w_raw)) if max(h_raw, w_raw) > max_dim else 1.0
    W = max(64, int(round(w_raw * scale_down)))
    H = max(64, int(round(h_raw * scale_down)))

    front_padded, geom_mask_f, _ = clean_and_pad_view(rgb_orig, geom_mask_orig, color_mask_orig, W, H)

    # 2. Calibrate 3D Mesh Silhouette & Sculpt High-Poly 3D Relief
    mesh, ortho_params = _calibrate_mesh_and_sculpt_relief(mesh, front_padded, geom_mask_f)

    if color_mode == "clay" or ortho_params is None:
        return create_clay_sculpture_mesh(mesh), None

    # 3. Load Real Back Image or Synthesize Clean Tailored Back View
    has_real_back = False
    back_padded = None
    geom_mask_b = None

    if back_image_source is not None:
        try:
            if not (isinstance(back_image_source, str) and not os.path.exists(back_image_source)):
                rgb_b, pil_b, alpha_b = _load_image_rgb_and_alpha(back_image_source)
                if rgb_b is not None and rgb_b.size > 0:
                    gm_b_orig, cm_b_orig = extract_dual_masks(rgb_b, pil_b, orig_alpha=alpha_b)
                    back_padded, geom_mask_b, _ = clean_and_pad_view(rgb_b, gm_b_orig, cm_b_orig, W, H)
                    has_real_back = True
        except Exception as e_back:
            logging.warning(f"Notice loading real back image: {e_back}")
            has_real_back = False

    if not has_real_back or back_padded is None or geom_mask_b is None:
        back_padded, geom_mask_b = synthesize_clean_back_view(front_padded, geom_mask_f)

    # 4. Pack 2K Ultra-HD Texture Atlas (Left: Front, Right: Back)
    atlas_arr = np.hstack([front_padded, back_padded])
    atlas_pil = Image.fromarray(atlas_arr)

    # 5. Adaptive Local Coronal Depth + Normal Segmentation on Calibrated/Subdivided Mesh
    v = mesh.vertices
    fn = mesh.face_normals
    f = mesh.faces

    _, _, _, _, y0_3d, y1_3d, _ = ortho_params
    h_3d = max(1e-4, y1_3d - y0_3d)

    N = 128
    s_grid = np.linspace(0.0, 1.0, N)
    z3_min_s = np.zeros(N)
    z3_max_s = np.zeros(N)
    for i in range(N):
        y3 = y1_3d - s_grid[i] * h_3d
        sl3 = v[np.abs(v[:, 1] - y3) < h_3d * 0.018]
        if len(sl3) > 0:
            z3_min_s[i] = np.percentile(sl3[:, 2], 2.0)
            z3_max_s[i] = np.percentile(sl3[:, 2], 98.0)
        elif i > 0:
            z3_min_s[i], z3_max_s[i] = z3_min_s[i - 1], z3_max_s[i - 1]

    z3_min_s = ndi.gaussian_filter1d(z3_min_s, sigma=3)
    z3_max_s = ndi.gaussian_filter1d(z3_max_s, sigma=3)

    f_centers = v[f].mean(axis=1)
    f_s = np.clip((y1_3d - f_centers[:, 1]) / h_3d, 0.0, 1.0)
    f_zmin = np.interp(f_s, s_grid, z3_min_s)
    f_zmax = np.interp(f_s, s_grid, z3_max_s)
    f_z_rel = (f_centers[:, 2] - f_zmin) / np.maximum(0.05, f_zmax - f_zmin)

    front_face_mask = (fn[:, 2] >= -0.15) & (f_z_rel >= 0.25)
    back_face_mask = ~front_face_mask

    w_tex = float(W * 2.0)
    h_tex = float(H)

    all_verts = []
    all_faces = []
    all_uvs = []
    v_offset = 0

    if np.any(front_face_mask):
        front_sub = mesh.submesh([front_face_mask], append=True)
        fv = front_sub.vertices
        fx_px, fy_px = _compute_orthographic_uv_mapping(fv, geom_mask_f, ortho_params, is_back=False)
        front_u = np.clip(fx_px / w_tex, 0.001, 0.499)
        front_v = np.clip(1.0 - (fy_px / h_tex), 0.001, 0.999)
        all_verts.append(fv)
        all_faces.append(front_sub.faces + v_offset)
        all_uvs.append(np.column_stack([front_u, front_v]))
        v_offset += len(fv)

    if np.any(back_face_mask):
        back_sub = mesh.submesh([back_face_mask], append=True)
        bv = back_sub.vertices
        bx_px, by_px = _compute_orthographic_uv_mapping(bv, geom_mask_b, ortho_params, is_back=True)
        back_u = np.clip(0.5 + (bx_px / w_tex), 0.501, 0.999)
        back_v = np.clip(1.0 - (by_px / h_tex), 0.001, 0.999)
        all_verts.append(bv)
        all_faces.append(back_sub.faces + v_offset)
        all_uvs.append(np.column_stack([back_u, back_v]))
        v_offset += len(bv)

    pbr_mat = PBRMaterial(
        baseColorTexture=atlas_pil,
        roughnessFactor=0.62,
        metallicFactor=0.04,
        doubleSided=True
    )

    combo = trimesh.Trimesh(
        vertices=np.vstack(all_verts),
        faces=np.vstack(all_faces),
        process=False
    )
    combo.visual = trimesh.visual.TextureVisuals(
        uv=np.vstack(all_uvs),
        material=pbr_mat
    )
    return combo, atlas_pil
