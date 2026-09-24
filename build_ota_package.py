import os
import json
import zipfile

app_dir = os.path.dirname(os.path.abspath(__file__))
v_file = os.path.join(app_dir, "version.json")
with open(v_file, "r", encoding="utf-8") as f:
    v_data = json.load(f)

version = v_data.get("version", "2.0.0")

files_to_pack = [
    "app.py",
    "meshy_client.py",
    "texture_engine.py",
    "version.json",
    "Xem_File_3D.html",
    "Chay_AI_3D_Studio.bat",
    "Tat_AI_3D_Studio.bat",
    os.path.join("TripoSR", "tsr", "bake_texture.py"),
]

# Add hy3dgen package files
hy3dgen_dir = os.path.join(app_dir, "hy3dgen")
if os.path.exists(hy3dgen_dir):
    for root, dirs, files in os.walk(hy3dgen_dir):
        if "__pycache__" in root:
            continue
        for file in files:
            full_p = os.path.join(root, file)
            rel_p = os.path.relpath(full_p, app_dir)
            files_to_pack.append(rel_p)

for zip_name in [f"update_v{version}.zip", "latest_ota_package.zip"]:
    zip_path = os.path.join(app_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in files_to_pack:
            fpath = os.path.join(app_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=fname)
    size_kb = os.path.getsize(zip_path) / 1024
    print(f"Created {zip_name} ({size_kb:.2f} KB, {len(files_to_pack)} files)")
