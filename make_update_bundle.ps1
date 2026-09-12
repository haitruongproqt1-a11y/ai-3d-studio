param (
    [string]$Version = "1.1.0",
    [string[]]$ReleaseNotes = @(
        "Tích hợp Cloud API Siêu Thực (Meshy.ai) - Tạo model 3D giống ảnh 99% chuẩn Game AAA",
        "Hỗ trợ đầy đủ bộ vật liệu PBR (Vỏ bóng, độ nhám, chi tiết lồi lõm)",
        "Thêm chế độ Dual Engine: Chuyển đổi linh hoạt giữa GPU Offline và Cloud API",
        "Tự động lưu API Key trong phần Cài đặt",
        "Tối ưu hóa bộ lọc khử nền ô vuông giả lập"
    )
)

$baseDir = "C:\Users\ADMIN\.gemini\antigravity\scratch\ai_3d_engine"
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   TAO GOI CAP NHAT OTA CHO AI 3D STUDIO (v$Version)..." -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# 1. Tạo staging
$staging = "$baseDir\dist_update_staging"
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging -Force | Out-Null

Copy-Item "$baseDir\app.py" "$staging\" -Force

# 2. Tạo file version.json
$versionData = @{
    version = $Version
    buildDate = (Get-Date).ToString("yyyy-MM-dd")
    releaseNotes = $ReleaseNotes
    updateUrl = "https://github.com/haitruongproqt1-a11y/ai-3d-studio/releases/download/v$Version/update_v$Version.zip"
} | ConvertTo-Json -Depth 4

$versionJsonPath = "$baseDir\version.json"
$versionData | Set-Content -Path $versionJsonPath -Encoding UTF8
Copy-Item $versionJsonPath "$staging\" -Force

# 3. Đóng gói file update zip
$outZip = "$baseDir\update_v$Version.zip"
if (Test-Path $outZip) { Remove-Item $outZip -Force }

tar -a -cf $outZip -C $staging .
Remove-Item $staging -Recurse -Force

$zipSize = (Get-Item $outZip).Length / 1KB

Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  TAO GOI CAP NHAT THANH CONG!" -ForegroundColor Green
Write-Host "  Goi cap nhat: update_v$Version.zip ($([Math]::Round($zipSize, 2)) KB)" -ForegroundColor Green
Write-Host "  File version: version.json" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
