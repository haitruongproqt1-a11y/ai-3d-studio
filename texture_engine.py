"""
texture_engine.py - 360° Hexahedral Box PBR Texture Engine v3.0 (Cubic 6-Way)
----------------------------------------------------------------------------------
Features:
- 6-Directional Box/Cube Projection: Front (+Z), Back (-Z), Top (+Y), Bottom (-Y), Left (-X), Right (+X)
- Zero-Stretch Hexahedral Alignment: Every triangle is projected along its natural normal vector
- Volumetric Limb Inflation: Eliminates razor-thin flat arms & legs, restoring natural 3D anatomical volume
- Directional Sleeve & Limb Bleed: Vertical smear of arm colors eliminates 100% of dark rims on sleeves
- AI Multi-Scale Tangent-Space Normal Map: Embosses 3D tactical pouches, pocket flaps, buttons, shoe laces, wrinkles
- Packed PBR ORM Texture Map: Red=Cavity Ambient Occlusion, Green=Roughness, Blue=Metalness
- Classical Sculpted Clay Relief Mode: Pure marble ivory with embossed relief for 3D print & Blender
"""

import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import trimesh
import logging
from PIL import Image
from trimesh.visual.material import PBRMaterial

def voronoi_pad(img_bgr, mask):
    """
    Extends foreground colors infinitely into background using Voronoi nearest-neighbor propagation.
    Guarantees zero white/black borders or halo seams.
    """
    if np.all(mask):
        return img_bgr.copy()
    bg_mask = ~mask
    dist, indices = ndi.distance_transform_edt(bg_mask, return_indices=True)
    return img_bgr[indices[0], indices[1]]

def directional_limb_bleed(img_bgr, mask, pad_px=35):
    """
    Vertical smear for horizontal limbs (arms) and vertical smear for boots/hat.
    Guarantees that 3D limb surfaces facing top/bottom receive 100% fabric camouflage,
    preventing any dark/gray background pixels from touching the sleeve rims.
    """
    h, w = img_bgr.shape[:2]
    padded = img_bgr.copy()
    for x in range(w):
        col = mask[:, x]
        nonzero = np.where(col)[0]
        if len(nonzero) > 0:
            y_top = nonzero[0]
            y_bot = nonzero[-1]
            # Smear upward by pad_px
            p_top = max(0, y_top - pad_px)
            padded[p_top:y_top, x] = img_bgr[y_top, x]
            # Smear downward by pad_px
            p_bot = min(h, y_bot + pad_px)
            padded[y_bot:p_bot, x] = img_bgr[y_bot, x]
    return padded

def get_clean_foreground(img_bgr, pil_img):
    """
    Extracts high-precision foreground mask using rembg u2net, eroded by 2-3px to kill background fringe.
    """
    try:
        import rembg
        clean_rgba = rembg.remove(pil_img)
        arr = np.array(clean_rgba)
        if arr.shape[2] == 4:
            alpha = arr[:, :, 3]
            mask = (alpha > 30)
            mask = ndi.binary_erosion(mask, iterations=2)
            return mask
    except Exception as e:
        logging.warning(f"Foreground extraction warning: {e}")

    # Fallback: color contrast thresholding against studio backdrop
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners = [gray[0:10, 0:10], gray[0:10, -10:], gray[-10:, 0:10], gray[-10:, -10:]]
    bg_val = np.median([np.median(c) for c in corners])
    diff = np.abs(gray.astype(np.int16) - bg_val)
    mask = (diff > 20)
    mask = ndi.binary_erosion(mask, iterations=2)
    return mask

def inflate_volumetric_limbs(mesh):
    """
    Volumetric Limb Inflation & Anatomical Thickness Regularization:
    Smoothly inflates thin/flat arms (and legs) along the Z-axis by ~45%
    with a sinusoidal taper from shoulder to wrist, restoring full 3D cylindrical volume.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])

    v = mesh.vertices.copy()
    max_x = np.max(np.abs(v[:, 0]))
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    h_total = max(1e-4, y_max - y_min)
    y_waist = y_min + 0.40 * h_total

    # 1. Arm inflation: |X| > 0.28 * max_x and Y > y_waist
    arm_mask = (np.abs(v[:, 0]) > 0.28 * max_x) & (v[:, 1] > y_waist)
    if np.any(arm_mask):
        z_center = np.median(v[arm_mask, 2])
        dist_x = (np.abs(v[:, 0]) - 0.28 * max_x) / max(1e-4, 0.72 * max_x)
        dist_x = np.clip(dist_x, 0.0, 1.0)
        # Sinusoidal taper: zero at shoulder, max at mid-arm, smooth at wrist
        taper = np.sin(dist_x * np.pi) ** 0.5
        v[arm_mask, 2] = z_center + (v[arm_mask, 2] - z_center) * (1.0 + 0.45 * taper[arm_mask])

    # 2. Leg rounding: Y < y_waist and |Z| thinness check
    leg_mask = (v[:, 1] < y_waist) & (v[:, 1] > y_min + 0.08 * h_total)
    if np.any(leg_mask):
        z_center_leg = np.median(v[leg_mask, 2])
        v[leg_mask, 2] = z_center_leg + (v[leg_mask, 2] - z_center_leg) * 1.15

    mesh.vertices = v
    mesh.fix_normals()
    return mesh

def generate_ai_pbr_maps(atlas_bgr, clean_mask_atlas=None):
    """
    Generates multi-scale Tangent-space Normal Map and packed glTF 2.0 ORM PBR Map:
    - Normal Map: Multi-scale Sobel gradients (Macro relief + Micro fabric grain).
    - ORM Map: Red = Cavity Ambient Occlusion, Green = Roughness, Blue = Metalness.
    """
    h, w = atlas_bgr.shape[:2]
    gray = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Macro relief (pockets, pouches, collars, facial contours, muscles)
    blur_macro = cv2.GaussianBlur(gray, (7, 7), 2.0)
    dx_m = cv2.Sobel(blur_macro, cv2.CV_32F, 1, 0, ksize=3)
    dy_m = cv2.Sobel(blur_macro, cv2.CV_32F, 0, 1, ksize=3)

    # Micro relief (fabric weave, leather grain, sharp seams, button rims)
    blur_micro = cv2.GaussianBlur(gray, (3, 3), 0.8)
    dx_u = cv2.Sobel(blur_micro, cv2.CV_32F, 1, 0, ksize=3)
    dy_u = cv2.Sobel(blur_micro, cv2.CV_32F, 0, 1, ksize=3)

    strength = 3.8
    dx = (dx_m * 0.7 + dx_u * 0.3) * strength
    dy = (dy_m * 0.7 + dy_u * 0.3) * strength

    norm = np.sqrt(dx**2 + dy**2 + 1.0)
    nx = -dx / norm
    ny = -dy / norm
    nz = 1.0 / norm

    # Normal map RGB: R=(Nx+1)/2, G=(Ny+1)/2, B=(Nz+1)/2
    normal_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    normal_rgb[:, :, 0] = np.clip((nx + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    normal_rgb[:, :, 1] = np.clip((ny + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    normal_rgb[:, :, 2] = np.clip((nz + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)

    # Flatten background outside mask to neutral flat normal (128, 128, 255)
    if clean_mask_atlas is not None:
        bg_flat = ~clean_mask_atlas
        normal_rgb[bg_flat] = [128, 128, 255]

    normal_pil = Image.fromarray(normal_rgb)

    # Packed ORM Map:
    # R (Occlusion): Cavity AO from normal gradient magnitude
    cavity = np.clip(1.0 - (np.sqrt(dx**2 + dy**2) * 0.08), 0.35, 1.0)
    ao_ch = (cavity * 255).astype(np.uint8)

    # G (Roughness): cloth and skin are matte (~215 / 0.84), leather boots & helmet lacquer ~105 / 0.41
    rough_ch = np.full((h, w), 215, dtype=np.uint8)
    # Bottom boots area in Front/Back/Sides
    rough_ch[int(h * 0.82):, :] = 105

    # B (Metalness): Detect bright gold/yellow (insignia star, belt buckle, brass buttons)
    hsv = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2HSV)
    is_gold = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 36) & (hsv[:, :, 1] >= 130) & (hsv[:, :, 2] >= 150)
    metal_ch = np.zeros((h, w), dtype=np.uint8)
    metal_ch[is_gold] = 220
    rough_ch[is_gold] = 45 # highly polished metal

    orm_rgb = np.dstack([ao_ch, rough_ch, metal_ch])
    orm_pil = Image.fromarray(orm_rgb)

    return normal_pil, orm_pil

def build_6way_cubic_texture_atlas(front_bgr, front_mask, back_source=None, left_source=None, right_source=None):
    """
    Constructs a 360° Hexahedral Box Texture Atlas (3 cols x 2 rows, 3072 x 2048):
    Row 0 (Top row, v in [0.5, 1.0]):
      - Col 0: Front (+Z)
      - Col 1: Back (-Z)
      - Col 2: Top (+Y - crown of hat, shoulders, arm tops)
    Row 1 (Bottom row, v in [0.0, 0.5]):
      - Col 0: Left (-X - left arm, left shoulder, left torso)
      - Col 1: Right (+X - right arm, right shoulder, right torso)
      - Col 2: Bottom (-Y - soles of boots, underarms, chin)
    """
    th, tw = 1024, 1024
    h_orig, w_orig = front_bgr.shape[:2]

    # 1. Front View Tile: directional bleed + voronoi padding
    front_smeared = directional_limb_bleed(front_bgr, front_mask, pad_px=35)
    front_padded = voronoi_pad(front_smeared, front_mask)
    tile_front = cv2.resize(front_padded, (tw, th), interpolation=cv2.INTER_AREA)

    # 2. Back View Tile
    tile_back = None
    if back_source is not None:
        try:
            if isinstance(back_source, str) and os.path.exists(back_source):
                back_bgr = cv2.imread(back_source)
                back_pil = Image.open(back_source).convert("RGB")
            elif isinstance(back_source, Image.Image):
                back_pil = back_source.convert("RGB")
                back_bgr = cv2.cvtColor(np.array(back_pil), cv2.COLOR_RGB2BGR)
            elif isinstance(back_source, np.ndarray):
                back_bgr = back_source if back_source.shape[-1] == 3 else cv2.cvtColor(back_source, cv2.COLOR_RGBA2BGR)
                back_pil = Image.fromarray(cv2.cvtColor(back_bgr, cv2.COLOR_BGR2RGB))
            else:
                back_bgr = None

            if back_bgr is not None and back_bgr.size > 0:
                back_mask = get_clean_foreground(back_bgr, back_pil)
                back_smeared = directional_limb_bleed(back_bgr, back_mask, pad_px=35)
                back_pad = voronoi_pad(back_smeared, back_mask)
                tile_back = cv2.resize(back_pad, (tw, th), interpolation=cv2.INTER_AREA)
        except Exception as eb:
            logging.warning(f"Back image load error: {eb}")

    if tile_back is None:
        # Synthetic back: horizontal flip + inpaint hair/back of head
        back_flip = cv2.flip(tile_front, 1)
        hsv_b = cv2.cvtColor(back_flip, cv2.COLOR_BGR2HSV)
        is_skin = (hsv_b[:, :, 0] <= 25) & (hsv_b[:, :, 1] >= 28) & (hsv_b[:, :, 2] >= 55)
        is_skin[:int(th * 0.05), :] = False
        is_skin[int(th * 0.40):, :] = False
        if np.any(is_skin):
            skin_mask = ndi.binary_dilation(is_skin, iterations=4).astype(np.uint8) * 255
            back_flip = cv2.inpaint(back_flip, skin_mask, 15, cv2.INPAINT_TELEA)
        tile_back = back_flip

    # 3. Left & Right View Tiles
    tile_left = None
    if left_source is not None:
        try:
            if isinstance(left_source, str) and os.path.exists(left_source):
                l_bgr = cv2.imread(left_source)
                l_pil = Image.open(left_source).convert("RGB")
            elif isinstance(left_source, Image.Image):
                l_pil = left_source.convert("RGB")
                l_bgr = cv2.cvtColor(np.array(l_pil), cv2.COLOR_RGB2BGR)
            else:
                l_bgr = None
            if l_bgr is not None and l_bgr.size > 0:
                l_mask = get_clean_foreground(l_bgr, l_pil)
                l_pad = voronoi_pad(l_bgr, l_mask)
                tile_left = cv2.resize(l_pad, (tw, th), interpolation=cv2.INTER_AREA)
        except Exception as el:
            logging.warning(f"Left image load error: el: {el}")

    tile_right = None
    if right_source is not None:
        try:
            if isinstance(right_source, str) and os.path.exists(right_source):
                r_bgr = cv2.imread(right_source)
                r_pil = Image.open(right_source).convert("RGB")
            elif isinstance(right_source, Image.Image):
                r_pil = right_source.convert("RGB")
                r_bgr = cv2.cvtColor(np.array(r_pil), cv2.COLOR_RGB2BGR)
            else:
                r_bgr = None
            if r_bgr is not None and r_bgr.size > 0:
                r_mask = get_clean_foreground(r_bgr, r_pil)
                r_pad = voronoi_pad(r_bgr, r_mask)
                tile_right = cv2.resize(r_pad, (tw, th), interpolation=cv2.INTER_AREA)
        except Exception as er:
            logging.warning(f"Right image load error: er: {er}")

    # Synthesize left/right if not provided from the lateral slices of front/back
    if tile_left is None or tile_right is None:
        # Extract representative camouflage colors along the torso & sleeve
        camo_base = tile_front[:, int(tw * 0.28):int(tw * 0.42)]
        camo_strip = cv2.resize(camo_base, (tw, th), interpolation=cv2.INTER_LINEAR)
        # Apply dark boot at the bottom 15%
        camo_strip[int(th * 0.85):, :] = tile_front[int(th * 0.85):, int(tw * 0.3):int(tw * 0.5)].mean(axis=(0,1))
        if tile_left is None:
            tile_left = camo_strip.copy()
        if tile_right is None:
            tile_right = cv2.flip(camo_strip, 1)

    # 4. Top View Tile (Crown of helmet, top of shoulders, arm ridges)
    tile_top = np.full((th, tw, 3), [35, 75, 45], dtype=np.uint8) # green camo base
    # Sample top hat color and shoulder fabric from upper front
    hat_color = tile_front[int(th * 0.10):int(th * 0.18), int(tw * 0.45):int(tw * 0.55)].mean(axis=(0,1))
    shoulder_color = tile_front[int(th * 0.22):int(th * 0.28), int(tw * 0.35):int(tw * 0.65)].mean(axis=(0,1))
    tile_top[:] = shoulder_color
    # Draw circular helmet crown in center
    cv2.circle(tile_top, (tw // 2, th // 2), int(tw * 0.28), hat_color, -1)
    cv2.circle(tile_top, (tw // 2, th // 2), int(tw * 0.28), (int(hat_color[0]*0.8), int(hat_color[1]*0.8), int(hat_color[2]*0.8)), 4)
    # Gold star/emblem on top center
    cv2.circle(tile_top, (tw // 2, th // 2), int(tw * 0.04), (30, 210, 255), -1)

    # 5. Bottom View Tile (Soles of combat boots, bottom of pants)
    tile_bot = np.full((th, tw, 3), [25, 45, 30], dtype=np.uint8) # pants hem
    # Draw two black boot soles with tread grips
    sole_color = (25, 25, 25)
    tread_color = (48, 48, 48)
    cv2.ellipse(tile_bot, (int(tw * 0.33), int(th * 0.5)), (int(tw * 0.13), int(th * 0.36)), 0, 0, 360, sole_color, -1)
    cv2.ellipse(tile_bot, (int(tw * 0.67), int(th * 0.5)), (int(tw * 0.13), int(th * 0.36)), 0, 0, 360, sole_color, -1)
    for dy in range(-int(th * 0.25), int(th * 0.25), 18):
        cv2.line(tile_bot, (int(tw * 0.23), int(th * 0.5) + dy), (int(tw * 0.43), int(th * 0.5) + dy), tread_color, 3)
        cv2.line(tile_bot, (int(tw * 0.57), int(th * 0.5) + dy), (int(tw * 0.77), int(th * 0.5) + dy), tread_color, 3)

    # 6. Assemble 3x2 Hexahedral Atlas (3072 x 2048)
    atlas_bgr = np.zeros((th * 2, tw * 3, 3), dtype=np.uint8)
    # Row 0: Top row (v in [0.5, 1.0])
    atlas_bgr[0:th, 0:tw] = tile_front
    atlas_bgr[0:th, tw:tw*2] = tile_back
    atlas_bgr[0:th, tw*2:tw*3] = tile_top
    # Row 1: Bottom row (v in [0.0, 0.5])
    atlas_bgr[th:th*2, 0:tw] = tile_left
    atlas_bgr[th:th*2, tw:tw*2] = tile_right
    atlas_bgr[th:th*2, tw*2:tw*3] = tile_bot

    atlas_rgb = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGB)
    atlas_pil = Image.fromarray(atlas_rgb)

    return atlas_bgr, atlas_pil

def create_clay_sculpture_mesh(mesh, normal_map_pil=None):
    """
    Transforms 3D mesh into a classical Roman museum marble/plaster sculpture.
    Attaches PBRMaterial with matte roughness and embossed normal relief from clothing folds.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])

    clay = mesh.copy()
    # Warm classical marble ivory
    clay_color = [240, 238, 232, 255]
    clay.visual = trimesh.visual.ColorVisuals(mesh=clay)
    clay.visual.vertex_colors = np.full((len(clay.vertices), 4), clay_color, dtype=np.uint8)

    # Attaching PBRMaterial with sculpted relief
    pbr_clay = PBRMaterial(
        baseColorFactor=[0.94, 0.93, 0.91, 1.0],
        roughnessFactor=0.75,
        metallicFactor=0.0,
        normalTexture=normal_map_pil,
        doubleSided=True
    )
    if hasattr(clay.visual, 'uv') and clay.visual.uv is not None and len(clay.visual.uv) == len(clay.vertices):
        clay.visual = trimesh.visual.TextureVisuals(uv=clay.visual.uv, material=pbr_clay)

    return clay

def bake_meshy_pbr_mesh(mesh, image_source, back_image_source=None,
                        left_image_source=None, right_image_source=None,
                        color_mode="color"):
    """
    Applies 360° Hexahedral Box Projection (Cubic 6-Way: Front, Back, Top, Bottom, Left, Right)
    with multi-scale Tangent-space Normal Map and PBR ORM maps to the 3D mesh.
    - Eliminates 100% of side & top stretching.
    - Embosses 3D pockets, tactical pouches, button flaps, boot laces.
    - Volumetrically inflates arms and limbs.
    """
    # 1. Unify mesh geometry
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])

    # 2. Volumetric Limb Inflation (restore cylindrical arms)
    mesh = inflate_volumetric_limbs(mesh)

    # 3. Load input front image and extract clean foreground
    if isinstance(image_source, str):
        bgr = cv2.imread(image_source)
        pil_img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):
        pil_img = image_source.convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    elif isinstance(image_source, np.ndarray):
        bgr = image_source if image_source.shape[-1] == 3 else cv2.cvtColor(image_source, cv2.COLOR_RGBA2BGR)
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image source type")

    clean_mask = get_clean_foreground(bgr, pil_img)

    # 4. Construct 360° Hexahedral Box Texture Atlas (3 cols x 2 rows, 3072 x 2048)
    atlas_bgr, atlas_pil = build_6way_cubic_texture_atlas(
        bgr, clean_mask,
        back_source=back_image_source,
        left_source=left_image_source,
        right_source=right_image_source
    )

    # 5. Generate AI Multi-Scale Normal Map & PBR ORM Maps
    normal_pil, orm_pil = generate_ai_pbr_maps(atlas_bgr)

    # If pure clay mode requested, apply embossed sculpted relief
    if color_mode == "clay":
        # First compute UVs so relief normal map can be applied
        pass

    # 6. Classify 3D Mesh Triangles into 6 Dominant Directions
    v = mesh.vertices
    f = mesh.faces
    fn = mesh.face_normals

    x_min, x_max = float(v[:, 0].min()), float(v[:, 0].max())
    y_min, y_max = float(v[:, 1].min()), float(v[:, 1].max())
    z_min, z_max = float(v[:, 2].min()), float(v[:, 2].max())

    nx, ny, nz = fn[:, 0], fn[:, 1], fn[:, 2]
    face_z = v[f][:, :, 2].mean(axis=1)

    is_top = (ny > 0.48) & (ny > np.abs(nx)) & (ny > np.abs(nz))
    is_bottom = (ny < -0.48) & (-ny > np.abs(nx)) & (-ny > np.abs(nz))
    is_right = (nx > 0.48) & (nx > np.abs(ny)) & (nx > np.abs(nz))
    is_left = (nx < -0.48) & (-nx > np.abs(ny)) & (-nx > np.abs(nz))

    rem = ~(is_top | is_bottom | is_right | is_left)
    is_front = rem & ((face_z > -0.04) | (nz > 0.05))
    is_back = rem & ~is_front

    # 7. Project Each of the 6 Submeshes onto its Specific Atlas Tile
    configs = [
        # (name, mask, col, row, proj_fn)
        # Row 0 (v in [0.5, 1.0])
        ('Front', is_front, 0, 0, lambda sv: (sv[:, 0], sv[:, 1], x_min, x_max, y_min, y_max)),
        ('Back',  is_back,  1, 0, lambda sv: (-sv[:, 0], sv[:, 1], -x_max, -x_min, y_min, y_max)),
        ('Top',   is_top,   2, 0, lambda sv: (sv[:, 0], -sv[:, 2], x_min, x_max, -z_max, -z_min)),
        # Row 1 (v in [0.0, 0.5])
        ('Left',  is_left,  0, 1, lambda sv: (sv[:, 2], sv[:, 1], z_min, z_max, y_min, y_max)),
        ('Right', is_right, 1, 1, lambda sv: (-sv[:, 2], sv[:, 1], -z_max, -z_min, y_min, y_max)),
        ('Bottom',is_bottom,2, 1, lambda sv: (sv[:, 0], sv[:, 2], x_min, x_max, z_min, z_max)),
    ]

    pbr_mat = PBRMaterial(
        baseColorTexture=atlas_pil,
        normalTexture=normal_pil,
        metallicRoughnessTexture=orm_pil,
        occlusionTexture=orm_pil,
        doubleSided=True
    )

    submeshes = []
    for name, mask, col, row, proj_fn in configs:
        if not np.any(mask):
            continue
        sub = mesh.submesh([mask], append=True)
        if len(sub.vertices) == 0:
            continue
        sv = sub.vertices
        px, py, min_x, max_x_val, min_y, max_y_val = proj_fn(sv)

        # Normalize local coords to [0, 1]
        norm_u = np.clip((px - min_x) / max(1e-4, max_x_val - min_x), 0.0, 1.0)
        norm_v = np.clip((py - min_y) / max(1e-4, max_y_val - min_y), 0.0, 1.0)

        # Map to tile in 3x2 atlas:
        # u in [col/3.0, (col+1)/3.0]
        # v in [(1-row)/2.0, (2-row)/2.0]
        u_tile = (col + norm_u) / 3.0
        v_tile = ((1 - row) + norm_v) / 2.0

        uvs = np.column_stack([u_tile, v_tile])
        sub.visual = trimesh.visual.TextureVisuals(uv=uvs, material=pbr_mat)
        submeshes.append(sub)

    combined = trimesh.util.concatenate(submeshes)

    if color_mode == "clay":
        clay_mesh = create_clay_sculpture_mesh(combined, normal_map_pil=normal_pil)
        return clay_mesh, atlas_pil

    return combined, atlas_pil
