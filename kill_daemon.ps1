# Kill TFT inference daemon
$port = 8000

# Find process using port 8000
$netstatOutput = netstat -ano | Select-String ":$port"
if ($netstatOutput) {
    $pid = ($netstatOutput -split '\s+')[-1]
    Write-Host "Found process $pid using port $port"
    taskkill /F /PID $pid
    Write-Host "Daemon killed"
} else {
    Write-Host "No process found on port $port"
}
