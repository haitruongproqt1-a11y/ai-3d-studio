import base64
import os

glb_path = r"C:\Users\ADMIN\Desktop\Chair_3D\mesh_baked.glb"
html_path = r"C:\Users\ADMIN\Desktop\Chair_3D\view_3d.html"

with open(glb_path, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Model Viewer - Ghế Gỗ (Texture Baked Mới)</title>
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: radial-gradient(circle at 50% 50%, #2a2b3d 0%, #12131a 100%);
            font-family: 'Segoe UI', system-ui, sans-serif;
            color: #fff;
        }}
        model-viewer {{
            width: 100%;
            height: 100%;
            --poster-color: transparent;
        }}
        .overlay {{
            position: absolute;
            top: 24px;
            left: 24px;
            background: rgba(18, 19, 26, 0.85);
            backdrop-filter: blur(12px);
            padding: 16px 24px;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            pointer-events: none;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            z-index: 10;
        }}
        h1 {{
            margin: 0 0 6px 0;
            font-size: 20px;
            font-weight: 600;
            color: #60a5fa;
        }}
        p {{
            margin: 0;
            font-size: 13px;
            color: #94a3b8;
        }}
        .badge {{
            display: inline-block;
            background: #10b981;
            color: #fff;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            margin-top: 6px;
        }}
        .controls-hint {{
            position: absolute;
            bottom: 28px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(18, 19, 26, 0.85);
            backdrop-filter: blur(12px);
            padding: 12px 26px;
            border-radius: 40px;
            font-size: 14px;
            color: #e2e8f0;
            pointer-events: none;
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            white-space: nowrap;
        }}
        .controls-hint b {{
            color: #38bdf8;
        }}
    </style>
</head>
<body>
    <div class="overlay">
        <h1>Mô hình 3D - Bản khử sạch nền & Baked Texture</h1>
        <p>Đã loại bỏ hoàn toàn ô ca-rô xám | Nan tựa và chân ghế tách bạch</p>
        <div class="badge">✓ ĐÃ CẬP NHẬT MỚI NHẤT</div>
    </div>
    <div class="controls-hint">
        Chuột trái: <b>Xoay 360°</b> | Con lăn: <b>Zoom</b> | Chuột phải: <b>Di chuyển</b>
    </div>
    <model-viewer 
        src="data:model/gltf-binary;base64,{b64_data}" 
        camera-controls 
        auto-rotate 
        auto-rotate-delay="1000" 
        rotation-per-second="25deg"
        shadow-intensity="1.5" 
        shadow-softness="0.8" 
        exposure="1.2">
    </model-viewer>
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print("Updated HTML viewer at:", html_path)
