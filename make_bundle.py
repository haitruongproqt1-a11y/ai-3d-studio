import os
import json
import zipfile

base_dir = r"C:\Users\ADMIN\.gemini\antigravity\scratch\ai_3d_engine"
version = "1.4.0"

out_zip = os.path.join(base_dir, f"update_v{version}.zip")
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(os.path.join(base_dir, "app.py"), "app.py")

zip_size_kb = os.path.getsize(out_zip) / 1024
print(f"Created update_v{version}.zip ({zip_size_kb:.2f} KB)")
