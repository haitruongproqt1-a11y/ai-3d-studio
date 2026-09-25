import os
import sys
import json
import urllib.request
import urllib.error
import subprocess

app_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(app_dir, "version.json"), "r", encoding="utf-8") as f:
    vdata = json.load(f)

version = "v" + vdata.get("version", "1.8.0").lstrip("v")
owner = "haitruongproqt1-a11y"
repo = "ai-3d-studio"

# 1. Get token
token = os.environ.get("GITHUB_TOKEN", "")
if not token:
    p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate(input='protocol=https\nhost=github.com\n\n')
    for line in out.splitlines():
        if line.startswith('password='):
            token = line.split('=', 1)[1].strip()
            break

if not token:
    print("Error: No GitHub token found!")
    sys.exit(1)

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "AI-3D-Studio-Release"
}

# 2. Check if release already exists
rel_api = f"https://api.github.com/repos/{owner}/{repo}/releases"
tag_api = f"{rel_api}/tags/{version}"

release_data = None
try:
    req = urllib.request.Request(tag_api, headers=headers)
    with urllib.request.urlopen(req) as resp:
        release_data = json.loads(resp.read().decode())
        print(f"Release {version} already exists. ID: {release_data['id']}")
except urllib.error.HTTPError as e:
    if e.code != 404:
        raise

if not release_data:
    body_notes = "\n".join([f"- {note}" for note in vdata.get("releaseNotes", [])])
    payload = {
        "tag_name": version,
        "name": f"AI 3D Studio {version} – Động Cơ Chiếu Khối Lập Phương 360° 6 Hướng, Khử Dẹt Cánh Tay & AI PBR Normal Map",
        "body": f"### Bản phát hành AI 3D Studio {version}\n\n{body_notes}\n\n*Gói cập nhật OTA tự động tải về qua ứng dụng.*",
        "draft": False,
        "prerelease": False
    }
    req = urllib.request.Request(rel_api, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        release_data = json.loads(resp.read().decode())
        print(f"Created release {version} successfully! ID: {release_data['id']}")

upload_url_tmpl = release_data.get("upload_url", "")
upload_base = upload_url_tmpl.split("{")[0]

# 3. Upload assets (update_vX.X.X.zip and latest_ota_package.zip)
assets_to_upload = [
    f"update_{version}.zip",
    "latest_ota_package.zip"
]

existing_assets = release_data.get("assets", [])

for asset_name in assets_to_upload:
    asset_path = os.path.join(app_dir, asset_name)
    if not os.path.exists(asset_path):
        print(f"Skipping {asset_name}: file not found")
        continue

    # Delete existing asset with same name if present
    for ea in existing_assets:
        if ea.get("name") == asset_name:
            del_url = ea.get("url")
            print(f"Deleting older asset {asset_name} (ID: {ea.get('id')})...")
            try:
                del_req = urllib.request.Request(del_url, headers=headers, method="DELETE")
                urllib.request.urlopen(del_req)
            except Exception as de:
                print(f"Notice deleting asset: {de}")

    print(f"Uploading {asset_name} ({os.path.getsize(asset_path)/1024:.2f} KB)...")
    upload_url = f"{upload_base}?name={asset_name}"
    up_headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/zip",
        "User-Agent": "AI-3D-Studio-Release"
    }
    with open(asset_path, "rb") as f:
        data = f.read()
    up_req = urllib.request.Request(upload_url, data=data, headers=up_headers, method="POST")
    with urllib.request.urlopen(up_req) as resp:
        print(f"Uploaded {asset_name} successfully! Status:", resp.status)

print(f"\nAll OTA packages for {version} successfully published to GitHub Release!")
