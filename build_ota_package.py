import os
import json
import zipfile

app_dir = os.path.dirname(os.path.abspath(__file__))
v_file = os.path.join(app_dir, "version.json")
with open(v_file, "r", encoding="utf-8") as f:
    v_data = json.load(f)

version = v_data.get("version", "1.8.0")

files_to_pack = [
    "app.py",
    "version.json",
    "Chay_AI_3D_Studio.bat",
]

for zip_name in [f"update_v{version}.zip", "latest_ota_package.zip"]:
    zip_path = os.path.join(app_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in files_to_pack:
            fpath = os.path.join(app_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=fname)
    size_kb = os.path.getsize(zip_path) / 1024
    print(f"Created {zip_name} ({size_kb:.2f} KB)")
