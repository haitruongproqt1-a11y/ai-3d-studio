"""
hf_free_client.py - 100% Free Hugging Face Spaces Cloud 3D Engine for AI 3D Studio
-----------------------------------------------------------------------------------
Generates high-definition 3D models and textures with ZERO paid credit deduction (0 VNĐ).
Uses Hugging Face Free ZeroGPU Spaces (Tencent Hunyuan3D-2.1 / TRELLIS).
Supports:
- Multi-View inputs (Front, Back, Left, Right)
- Pure Plaster/Clay Sculpture Mode (no color/texture, perfect for 3D print & Blender)
- Full PBR Realistic Color Mode (calibrated 360° 6-Way Cubic camera projection)
Optionally accepts a free Hugging Face User Access Token (https://huggingface.co/settings/tokens)
for extended personal quota and queue priority.
"""

import os
import time
import shutil
import logging
import urllib.parse
import httpx
from PIL import Image


def _safe_close_gradio_client(c):
    """Closes Gradio Client background heartbeat & SSE stream threads to prevent socket hangs."""
    if c is None:
        return
    try:
        c.close()
    except Exception:
        pass
    for exec_attr in ("stream_executor", "helper_executor", "executor"):
        ex = getattr(c, exec_attr, None)
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass


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

    def _download_mesh_from_result(self, raw_result, target_path: str, client_obj=None,
                                   token: str = None, progress_cb=None) -> bool:
        """
        Robustly extracts GLB model reference from gradio_client output (local filepath, dict, nested dict, url),
        and copies or streams it to target_path with strict non-blocking timeouts and replica session cookies.
        """
        if not raw_result:
            return False

        def _extract_candidates(obj):
            found = []
            if not obj:
                return found
            if isinstance(obj, str):
                s = obj.strip()
                if s and not s.startswith("<"):
                    found.append(s)
            elif isinstance(obj, dict):
                val = obj.get("value")
                if isinstance(val, dict):
                    # Check local downloaded path first, then URL, then server path
                    p = val.get("path")
                    u = val.get("url")
                    if p and isinstance(p, str) and os.path.exists(p):
                        found.append(p.strip())
                    if u and isinstance(u, str):
                        found.append(u.strip())
                    if p and isinstance(p, str):
                        found.append(p.strip())
                elif isinstance(val, str) and val.strip() and not val.strip().startswith("<"):
                    found.append(val.strip())
                else:
                    p = obj.get("path")
                    u = obj.get("url")
                    if p and isinstance(p, str) and os.path.exists(p):
                        found.append(p.strip())
                    if u and isinstance(u, str):
                        found.append(u.strip())
                    if p and isinstance(p, str):
                        found.append(p.strip())
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    found.extend(_extract_candidates(item))
            return found

        candidates = _extract_candidates(raw_result)
        if not candidates:
            logging.error(f"No valid file candidate found in HF result: {raw_result}")
            return False

        # 1. First priority: If gradio_client already downloaded a local file on disk, copy it immediately!
        for cand in candidates:
            if os.path.exists(cand) and os.path.isfile(cand) and os.path.getsize(cand) > 1024:
                if progress_cb:
                    progress_cb("📥 Đã nhận mô hình 3D từ Hugging Face Cloud, đang xử lý…", 78)
                shutil.copy2(cand, target_path)
                return os.path.exists(target_path) and os.path.getsize(target_path) > 1024

        # 2. Second priority: Build valid download URL(s) from server path or URL
        urls_to_try = []
        root_url = getattr(client_obj, "src_prefixed", "https://tencent-hunyuan3d-2-1.hf.space/")
        if not root_url.endswith("/"):
            root_url += "/"

        for cand in candidates:
            c_low = cand.lower()
            if not (".glb" in c_low or ".obj" in c_low or "white_mesh" in c_low or "file=" in c_low or cand.startswith("/tmp/")):
                continue
            if cand.startswith("http://") or cand.startswith("https://"):
                urls_to_try.append(cand)
                # Also construct the canonical Gradio /gradio_api/file= URL if cand is a /file= path
                if "/file=" in cand and "/gradio_api/file=" not in cand:
                    parts = cand.split("/file=", 1)
                    urls_to_try.append(f"{root_url}file={parts[1]}")
            elif cand.startswith("/"):
                encoded_p = urllib.parse.quote(cand, safe="/")
                urls_to_try.append(f"{root_url}file={encoded_p}")
                urls_to_try.append(f"https://tencent-hunyuan3d-2-1.hf.space/file={encoded_p}")

        # Deduplicate while preserving order
        seen = set()
        unique_urls = []
        for u in urls_to_try:
            if u not in seen:
                seen.add(u)
                unique_urls.append(u)

        if not unique_urls:
            logging.error(f"No downloadable URL found from candidates: {candidates}")
            return False

        if progress_cb:
            progress_cb("📥 Đang tải mô hình 3D nguyên bản từ Hugging Face Cloud…", 78)

        # Prepare headers & session-affinity cookies from Gradio Client
        headers = {}
        cookies = {}
        if client_obj is not None:
            try:
                if getattr(client_obj, "headers", None):
                    headers.update(dict(client_obj.headers))
                if getattr(client_obj, "cookies", None):
                    cookies.update(dict(client_obj.cookies))
            except Exception:
                pass
        if token and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {token}"

        dl_timeout = httpx.Timeout(35.0, connect=10.0, read=15.0)

        for url in unique_urls:
            logging.info(f"Hugging Face streaming model from: {url}")
            for use_auth in (True, False) if headers.get("Authorization") else (False,):
                req_headers = dict(headers)
                if not use_auth:
                    req_headers.pop("Authorization", None)
                try:
                    t_dl_start = time.time()
                    with httpx.Client(timeout=dl_timeout, cookies=cookies, follow_redirects=True) as http_client:
                        with http_client.stream("GET", url, headers=req_headers) as resp:
                            if resp.status_code in (401, 403, 404):
                                continue
                            resp.raise_for_status()
                            total_bytes = int(resp.headers.get("content-length", 0) or 0)
                            downloaded = 0
                            with open(target_path, "wb") as f_out:
                                for chunk in resp.iter_bytes(chunk_size=16384):
                                    if chunk:
                                        f_out.write(chunk)
                                        downloaded += len(chunk)
                                        elapsed_dl = max(0.1, time.time() - t_dl_start)
                                        speed_bps = downloaded / elapsed_dl
                                        # Smart ISP bandwidth throttle check: if international route to HF Space
                                        # is throttled below 25 KB/s after 14s and remaining time > 35s, abort early
                                        # so local RTX 3050 fallback finishes it much faster without hanging at 78%
                                        if elapsed_dl > 14.0 and speed_bps < 25600:
                                            rem_bytes = max(0, total_bytes - downloaded) if total_bytes > 0 else 1500000
                                            if rem_bytes / max(1.0, speed_bps) > 35.0:
                                                raise TimeoutError(
                                                    f"Đường truyền quốc tế tới Hugging Face đang nghẽn ({speed_bps/1024:.1f} KB/s)"
                                                )
                                        if progress_cb and downloaded > 0:
                                            kb = int(downloaded / 1024)
                                            if total_bytes > 0:
                                                tot_kb = int(total_bytes / 1024)
                                                dl_pct = min(83, 78 + int((downloaded / total_bytes) * 5))
                                                progress_cb(
                                                    f"📥 Đang tải 3D từ Cloud: {kb} / {tot_kb} KB ({int(speed_bps/1024)} KB/s)…",
                                                    dl_pct
                                                )
                                            else:
                                                progress_cb(f"📥 Đang tải 3D từ Cloud ({kb} KB)…", 79)
                    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024:
                        return True
                except TimeoutError as e_to:
                    logging.warning(f"International download throttled ({url}): {e_to}")
                    break
                except Exception as e_dl:
                    logging.warning(f"Stream download attempt failed ({url}): {e_dl}")

        return os.path.exists(target_path) and os.path.getsize(target_path) > 1024

    def _prepare_clean_rgba_input(self, img_path: str, out_path: str, progress_cb=None) -> tuple:
        """
        Guarantees that the image sent to Hugging Face Cloud 3D is a cleanly segmented,
        collar-repaired, tightly cropped RGBA PNG with 100% transparent background (alpha = 0).
        This prevents Cloud 3D from ever treating the photo background as a flat rectangular picture frame.
        Returns (saved_rgba_path, needs_cloud_rembg).
        """
        import numpy as np
        import scipy.ndimage as ndi

        orig = Image.open(img_path)
        orig_rgb = np.array(orig.convert("RGB"))
        clean_rgba = None

        if orig.mode == "RGBA":
            arr_a = np.array(orig)[:, :, 3]
            # Must have genuine transparency around the borders (at least 12% transparent and border median == 0)
            border_a = np.concatenate([arr_a[0, :], arr_a[-1, :], arr_a[:, 0], arr_a[:, -1]])
            if np.min(arr_a) < 80 and np.mean(arr_a < 30) >= 0.12 and np.median(border_a) < 15:
                clean_rgba = orig

        if clean_rgba is None:
            if progress_cb:
                progress_cb("🤖 AI đang tách sạch nền trong suốt 100% (Chống tạo khung ảnh phẳng)…", 18)
            try:
                cache_u2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "u2net")
                if os.path.exists(cache_u2):
                    os.environ["U2NET_HOME"] = cache_u2
                import rembg
                clean_rgba = rembg.remove(orig)
            except Exception as e_rem:
                logging.warning(f"HF Free local rembg notice: {e_rem}")
                try:
                    from tsr.utils import remove_background
                    clean_rgba = remove_background(orig.convert("RGB"))
                except Exception:
                    clean_rgba = orig.convert("RGBA")

        if clean_rgba.mode != "RGBA":
            clean_rgba = clean_rgba.convert("RGBA")

        arr = np.array(clean_rgba)
        rgb = arr[:, :, :3].copy()
        alpha = arr[:, :, 3].copy()

        # 1. Repair symmetrical collar & interior rembg holes
        try:
            from texture_engine import repair_symmetrical_collar
            rgb, alpha, _ = repair_symmetrical_collar(rgb, alpha)
            h_a, w_a = alpha.shape
            if orig_rgb.shape[:2] == (h_a, w_a):
                corners = np.concatenate([
                    orig_rgb[:12, :12].reshape(-1, 3),
                    orig_rgb[:12, -12:].reshape(-1, 3),
                    orig_rgb[-12:, :12].reshape(-1, 3),
                    orig_rgb[-12:, -12:].reshape(-1, 3),
                ], axis=0).astype(np.float32)
                if np.mean(np.std(corners, axis=0)) < 10.0:
                    bg_col = np.median(corners, axis=0)
                    col_diff = np.linalg.norm(orig_rgb.astype(np.float32) - bg_col[None, None, :], axis=2)
                    near_fg = ndi.binary_dilation(alpha > 160, iterations=max(6, int(max(h_a, w_a) * 0.015)))
                    missed_fg = near_fg & (col_diff > 48.0) & (alpha < 180)
                    if np.any(missed_fg):
                        rgb[missed_fg] = orig_rgb[missed_fg]
                        alpha[missed_fg] = 255
        except Exception as e_rep:
            logging.warning(f"HF Free collar/feature repair notice: {e_rep}")

        # 2. Strictly zero out faint background alpha noise & remove isolated dust islands
        fg_binary = alpha > 75
        if np.any(fg_binary):
            labeled, num_features = ndi.label(fg_binary)
            if num_features > 1:
                counts = np.bincount(labeled.ravel())
                counts[0] = 0
                max_area = counts.max()
                keep_labels = np.where(counts >= max(32, int(max_area * 0.005)))[0]
                fg_binary = np.isin(labeled, keep_labels)
            alpha[~fg_binary] = 0
            alpha[alpha > 210] = 255

        # 3. Tightly crop around the 3D subject with a clean 6% transparent border
        coords = np.nonzero(alpha > 20)
        if len(coords[0]) > 0:
            y_min, y_max = int(coords[0].min()), int(coords[0].max())
            x_min, x_max = int(coords[1].min()), int(coords[1].max())
            h_fg, w_fg = max(1, y_max - y_min), max(1, x_max - x_min)
            pad_y = max(8, int(round(h_fg * 0.06)))
            pad_x = max(8, int(round(w_fg * 0.06)))

            # Build a fresh canvas with guaranteed 0-alpha transparent border padding
            new_h = (y_max - y_min + 1) + 2 * pad_y
            new_w = (x_max - x_min + 1) + 2 * pad_x
            out_rgba = np.zeros((new_h, new_w, 4), dtype=np.uint8)
            out_rgba[pad_y:pad_y + (y_max - y_min + 1), pad_x:pad_x + (x_max - x_min + 1), :3] = rgb[y_min:y_max + 1, x_min:x_max + 1]
            out_rgba[pad_y:pad_y + (y_max - y_min + 1), pad_x:pad_x + (x_max - x_min + 1), 3] = alpha[y_min:y_max + 1, x_min:x_max + 1]
            clean_pil = Image.fromarray(out_rgba, mode="RGBA")
        else:
            clean_pil = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")

        clean_pil.save(out_path)
        final_a = np.array(clean_pil)[:, :, 3]
        needs_cloud_rembg = bool(np.mean(final_a < 30) < 0.10)
        return out_path, needs_cloud_rembg

    def _execute_space_job(self, c, image_path, back_image_path, left_image_path, right_image_path,
                           steps, octree_res, progress_cb, start_pct=35, max_wait_sec=75, rembg_needed=False):
        from gradio_client import handle_file

        # Cap Cloud octree_resolution to 180 so raw white_mesh.glb is ~1.0 MB instead of 4.6 MB
        # (prevents multi-minute downloads over throttled international links while keeping crisp geometry)
        cloud_octree = min(int(octree_res), 180)
        cloud_steps = min(int(steps), 15)

        submit_kwargs = {
            "image": handle_file(image_path),
            "steps": cloud_steps,
            "guidance_scale": 5.0,
            "octree_resolution": cloud_octree,
            "check_box_rembg": bool(rembg_needed),
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

        t_start = time.time()
        poll_count = 0
        while not job.done():
            time.sleep(1.2)
            poll_count += 1
            elapsed = time.time() - t_start
            if elapsed > max_wait_sec:
                try:
                    job.cancel()
                except Exception:
                    pass
                raise TimeoutError(f"Hàng đợi Cloud ZeroGPU phản hồi quá {max_wait_sec}s.")

            try:
                st = job.status()
                code = str(getattr(st, 'code', '')).upper()
                rank = getattr(st, 'rank', None)
                eta = getattr(st, 'eta', None)
                if any(k in code for k in ('STARTING', 'JOINING', 'IN_QUEUE', 'SENDING')):
                    if rank is not None and rank > 0:
                        if eta is not None and eta > 45 and elapsed > 20:
                            try:
                                job.cancel()
                            except Exception:
                                pass
                            raise TimeoutError(f"Hàng đợi Cloud ZeroGPU đang đông (Vị trí #{rank}, dự kiến {int(eta)}s).")
                        msg = f"⏳ Đang xếp hàng Cloud ZeroGPU: Vị trí #{rank} (0đ Miễn phí)…"
                    else:
                        msg = f"⏳ Đang kết nối phân bổ nhân ZeroGPU Cloud ({int(elapsed)}s)…"
                    if progress_cb:
                        progress_cb(msg, min(start_pct + 14, start_pct + poll_count))
                elif any(k in code for k in ('PROCESSING', 'ITERATING', 'PROGRESS', 'LOG')):
                    pct = min(76, start_pct + 15 + poll_count * 2)
                    if progress_cb:
                        progress_cb(f"⚡ ZeroGPU đang suy luận hình khối 3D Flow Matching ({pct}%)…", pct)
                else:
                    pct = min(75, start_pct + poll_count)
                    if progress_cb:
                        progress_cb(f"⚡ Đang xử lý trên máy chủ Hugging Face Cloud ({pct}%)…", pct)
            except TimeoutError:
                raise
            except Exception:
                pass

        return job

    def generate_3d_free(self, image_path: str, progress_cb=None, item_dir=None, steps=15, octree_res=180,
                          back_image_path=None, left_image_path=None, right_image_path=None,
                          color_mode="color") -> dict:
        """
        Executes free 3D generation via Hugging Face Spaces.
        Features dual-strategy resilience:
        1. Always segments & crops input images to 100% transparent RGBA so Cloud generates a real 3D statue (never a picture frame).
        2. Try with user token (if configured).
        3. If user token exceeds ZeroGPU quota, automatically retry anonymously.
        4. Supports single-view and multi-view, plus Clay sculpture vs Full Meshy-grade PBR texture.
        """
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Không tìm thấy ảnh: {image_path}"}

        try:
            from gradio_client import Client
        except ImportError:
            return {"success": False, "error": "Chưa cài đặt gradio_client. Vui lòng cập nhật môi trường!"}

        if progress_cb:
            progress_cb("🌐 Đang kết nối tới máy chủ Hugging Face Cloud Free (ZeroGPU 0đ)…", 15)

        created_temp_dir = False
        if not item_dir:
            ts = int(time.time())
            item_dir = os.path.join(r"H:\AI_3D_Studio\output_app", f"hf_{ts}")
            created_temp_dir = True
        os.makedirs(item_dir, exist_ok=True)
        raw_glb = os.path.join(item_dir, "raw_model.glb")

        # 1. Guarantee 100% transparent background & tight subject crop before sending to Cloud
        clean_front_path = os.path.join(item_dir, "input.png")
        clean_front_path, rembg_needed = self._prepare_clean_rgba_input(image_path, clean_front_path, progress_cb=progress_cb)

        clean_back_path = None
        if back_image_path and os.path.exists(back_image_path):
            clean_back_path, _ = self._prepare_clean_rgba_input(back_image_path, os.path.join(item_dir, "input_back.png"))

        clean_left_path = None
        if left_image_path and os.path.exists(left_image_path):
            clean_left_path, _ = self._prepare_clean_rgba_input(left_image_path, os.path.join(item_dir, "input_left.png"))

        clean_right_path = None
        if right_image_path and os.path.exists(right_image_path):
            clean_right_path, _ = self._prepare_clean_rgba_input(right_image_path, os.path.join(item_dir, "input_right.png"))

        job = None
        raw_res = None
        c = None
        used_token = self.hf_token if self.has_token() else None
        dl_ok = False

        try:
            # --- ATTEMPT 1: With User Token (if present) or Anonymous ---
            try:
                if progress_cb:
                    progress_cb("🐉 Đang nạp mô hình Tencent Hunyuan3D-2.1 trên Cloud GPU…", 25)

                c = Client(
                    "tencent/Hunyuan3D-2.1",
                    token=used_token,
                    httpx_kwargs={"timeout": httpx.Timeout(45.0, connect=15.0)},
                    download_files=False,
                    verbose=False
                )

                if progress_cb:
                    progress_cb("⏳ Đang gửi yêu cầu vào hàng đợi Cloud ZeroGPU (0đ Miễn phí)…", 35)

                job = self._execute_space_job(
                    c, clean_front_path, clean_back_path, clean_left_path, clean_right_path,
                    steps, octree_res, progress_cb, start_pct=35, rembg_needed=rembg_needed
                )

                if progress_cb:
                    progress_cb("📥 Đang tải mô hình 3D nguyên bản từ Hugging Face Cloud…", 78)

                try:
                    raw_res = job.result(timeout=15)
                except Exception as e_res:
                    logging.warning(f"job.result() warning, checking job.outputs(): {e_res}")
                    outs = job.outputs()
                    if outs and len(outs) > 0:
                        raw_res = outs[-1]
                    else:
                        raise e_res

                dl_ok = self._download_mesh_from_result(
                    raw_res, raw_glb, client_obj=c, token=used_token, progress_cb=progress_cb
                )

            except Exception as e_first:
                _safe_close_gradio_client(c)
                c = None
                err_str = str(e_first)
                logging.warning(f"HF Free Attempt 1 warning: {err_str}")

                is_quota_err = any(k in err_str.lower() for k in ("quota", "zerogpu", "rate limit", "exceeded", "429"))

                # If user token had quota exceeded, retry immediately with anonymous pool!
                if used_token and is_quota_err:
                    try:
                        if progress_cb:
                            progress_cb("💡 Token đã chạm hạn mức ngày. Tự động chuyển sang cụm ZeroGPU công cộng (0đ)…", 30)
                        used_token = None
                        c = Client(
                            "tencent/Hunyuan3D-2.1",
                            token=None,
                            httpx_kwargs={"timeout": httpx.Timeout(45.0, connect=15.0)},
                            download_files=False,
                            verbose=False
                        )
                        job = self._execute_space_job(
                            c, clean_front_path, clean_back_path, clean_left_path, clean_right_path,
                            steps, octree_res, progress_cb, start_pct=35, rembg_needed=rembg_needed
                        )
                        if progress_cb:
                            progress_cb("📥 Đang tải mô hình 3D nguyên bản từ Hugging Face Cloud…", 78)
                        try:
                            raw_res = job.result(timeout=15)
                        except Exception:
                            outs = job.outputs()
                            raw_res = outs[-1] if outs else None

                        dl_ok = self._download_mesh_from_result(
                            raw_res, raw_glb, client_obj=c, token=None, progress_cb=progress_cb
                        )
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
        finally:
            _safe_close_gradio_client(c)

        if not dl_ok or not os.path.exists(raw_glb) or os.path.getsize(raw_glb) <= 1024:
            return {
                "success": False,
                "error": "Đường truyền tải mô hình từ Hugging Face Cloud quá chậm hoặc bị ngắt."
            }

        if progress_cb:
            progress_cb("⚙️ Đang đọc & làm sạch cấu trúc lưới 3D nguyên bản…", 84)

        import trimesh
        import numpy as np
        mesh = trimesh.load(raw_glb, force="mesh")

        # 2. Clean disconnected floating fragments & verify true 3D statue geometry (reject flat slabs)
        try:
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                mesh = max(components, key=lambda comp: len(comp.vertices))
            mesh.remove_unreferenced_vertices()
            mesh.export(os.path.join(item_dir, "raw_hunyuan.glb"))
        except Exception as e_clean:
            logging.warning(f"HF Free mesh component cleanup notice: {e_clean}")

        ext = mesh.extents
        if len(ext) == 3 and max(ext[0], ext[1]) > 0:
            depth_ratio = float(ext[2]) / float(max(ext[0], ext[1]))
            if depth_ratio < 0.14:
                logging.warning(f"HF Free mesh depth_ratio={depth_ratio:.3f} is too flat (slab/frame). Triggering local 3D engine.")
                return {
                    "success": False,
                    "error": "Mô hình Cloud bị dẹt dạng khung ảnh, đang chuyển sang dựng khối 3D đầy đủ."
                }

        from texture_engine import bake_meshy_pbr_mesh
        if color_mode == "clay":
            if progress_cb:
                progress_cb("🏛️ Đang điêu khắc Tượng Thạch Cao Clay High-Poly (Chạm nổi chi tiết 3D)…", 88)
        else:
            if progress_cb:
                progress_cb("🎨 AI đang hiệu chỉnh 3D Relief & nướng vân PBR 2K (Tỷ lệ 1:1 chuẩn Meshy)…", 88)
        baked_mesh, _ = bake_meshy_pbr_mesh(
            mesh, clean_front_path,
            back_image_source=clean_back_path,
            left_image_source=clean_left_path,
            right_image_source=clean_right_path,
            color_mode=color_mode
        )

        if progress_cb:
            progress_cb("💾 Đang xuất tệp mô hình GLB sắc nét…", 95)

        glb_path = os.path.join(item_dir, "model.glb")
        obj_path = os.path.join(item_dir, "model.obj")
        baked_mesh.export(glb_path)

        engine_name = "🌐 Hugging Face Free Cloud (0đ - Thạch Cao Clay)" if color_mode == "clay" else "🌐 Hugging Face Free Cloud (0đ)"

        return {
            "success": True,
            "glb_path": glb_path,
            "obj_path": obj_path,
            "item_dir": item_dir,
            "engine_used": engine_name
        }
