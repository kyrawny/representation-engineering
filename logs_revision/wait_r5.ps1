# exit 0 while a 14_act_target_ablation python process is still running
$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*14_act_target_ablation*' }
if ($p) { exit 0 } else { exit 1 }
