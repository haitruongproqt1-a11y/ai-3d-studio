import os
import json
import zipfile

base_dir = r"C:\Users\ADMIN\.gemini\antigravity\scratch\ai_3d_engine"
version = "1.2.0"
release_notes = [
    "Tich hop Cloud Free 100% (Hugging Face) - Khong can mua API Key, khong can the Visa",
    "Bo sung thuat toan lam min be mat Taubin Smoothing (Khu sach rang cua lom chom, lam chuoi va vat the lang bong)",
    "Tu dong toi uu hinh khoi va loai bo hoan toan cac mang san sui",
    "Sua loi duong link cap nhat GitHub OTA 1-click thanh cong 100%"
]

out_zip = os.path.join(base_dir, f"update_v{version}.zip")
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(os.path.join(base_dir, "app.py"), "app.py")

zip_size_kb = os.path.getsize(out_zip) / 1024
print(f"Created update_v{version}.zip ({zip_size_kb:.2f} KB)")
