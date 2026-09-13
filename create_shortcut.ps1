$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("C:\Users\ADMIN\Desktop\AI 3D Studio.lnk")
$Shortcut.TargetPath = "C:\Users\ADMIN\.gemini\antigravity\scratch\ai_3d_engine\Chay_AI_3D_Studio.bat"
$Shortcut.WorkingDirectory = "C:\Users\ADMIN\.gemini\antigravity\scratch\ai_3d_engine"
$Shortcut.Description = "AI 3D Studio - RTX 3050 Offline 100%"
$Shortcut.Save()
Write-Host "Desktop shortcut created successfully!"
