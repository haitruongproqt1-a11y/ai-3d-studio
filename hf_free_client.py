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

    def _execute_space_job(self, c, image_path, back_image_path, left_image_path, right_image_path,
                           steps, octree_res, progress_cb, start_pct=35, max_wait_sec=75):
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

        created_temp_dir = False
        if not item_dir:
            ts = int(time.time())
            item_dir = os.path.join(r"H:\AI_3D_Studio\output_app", f"hf_{ts}")
            created_temp_dir = True
        os.makedirs(item_dir, exist_ok=True)
        raw_glb = os.path.join(item_dir, "raw_model.glb")

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
                    c, image_path, back_image_path, left_image_path, right_image_path,
                    steps, octree_res, progress_cb, start_pct=35
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
                            c, image_path, back_image_path, left_image_path, right_image_path,
                            steps, octree_res, progress_cb, start_pct=35
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
                        if created_temp_dir and os.path.exists(item_dir) and not os.listdir(item_dir):
                            shutil.rmtree(item_dir, ignore_errors=True)
                        return {
                            "success": False,
                            "error": f"Hạn mức Cloud Free ZeroGPU tạm hết: {e_anon}",
                            "quota_exceeded": True
                        }
                else:
                    if created_temp_dir and os.path.exists(item_dir) and not os.listdir(item_dir):
                        shutil.rmtree(item_dir, ignore_errors=True)
                    return {
                        "success": False,
                        "error": f"Lỗi Hugging Face Cloud Free: {err_str}",
                        "quota_exceeded": is_quota_err
                    }
        finally:
            _safe_close_gradio_client(c)

        if not dl_ok or not os.path.exists(raw_glb) or os.path.getsize(raw_glb) <= 1024:
            if created_temp_dir and os.path.exists(item_dir) and not os.listdir(item_dir):
                shutil.rmtree(item_dir, ignore_errors=True)
            return {
                "success": False,
                "error": "Đường truyền tải mô hình từ Hugging Face Cloud quá chậm hoặc bị ngắt."
            }

        if progress_cb:
            progress_cb("⚙️ Đang đọc cấu trúc lưới 3D nguyên bản…", 84)

        import trimesh
        mesh = trimesh.load(raw_glb, force="mesh")

        if color_mode == "clay":
            if progress_cb:
                progress_cb("🏛️ Đang tạo Tượng Thạch Cao Clay đơn sắc mịn màng…", 88)
            from texture_engine import create_clay_sculpture_mesh
            baked_mesh = create_clay_sculpture_mesh(mesh)
        else:
            if progress_cb:
                progress_cb("🎨 AI đang nướng bản đồ vân PBR Dual-View HD (Tỷ lệ 1:1)…", 88)
            from texture_engine import bake_meshy_pbr_mesh
            baked_mesh, _ = bake_meshy_pbr_mesh(
                mesh, image_path,
                back_image_source=back_image_path,
                left_image_source=left_image_path,
                right_image_source=right_image_path,
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
