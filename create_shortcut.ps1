$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("C:\Users\ADMIN\Desktop\AI 3D Studio.lnk")
$Shortcut.TargetPath = "H:\AI_3D_Studio\Chay_AI_3D_Studio.bat"
$Shortcut.WorkingDirectory = "H:\AI_3D_Studio"
$Shortcut.Description = "AI 3D Studio - RTX 3050 Offline 100% (SSD H:)"
$Shortcut.Save()
Write-Host "Desktop shortcut created successfully!"
