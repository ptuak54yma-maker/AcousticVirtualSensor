$ErrorActionPreference = 'Stop'

$simRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$firmwareRoot = Split-Path -Parent $simRoot
$repoRoot = (Resolve-Path (Join-Path $firmwareRoot '..\..')).Path
$aliasRoot = Join-Path $env:LOCALAPPDATA 'AVSRepoLink'
$renodeExe = Join-Path $env:LOCALAPPDATA 'RenodeExtracted\Renode\bin\Renode.exe'
$uartLog = Join-Path $simRoot 'uart.log'

if (-not (Test-Path -LiteralPath $renodeExe)) {
    throw "Renode not found: $renodeExe"
}
if (Test-Path -LiteralPath $aliasRoot) {
    $existing = Get-Item -LiteralPath $aliasRoot -Force
    $target = if ($existing.Target -is [array]) { $existing.Target[0] } else { $existing.Target }
    if (-not ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        [IO.Path]::GetFullPath($target).TrimEnd('\') -ine $repoRoot.TrimEnd('\')) {
        throw "ASCII alias exists but does not target this repository: $aliasRoot"
    }
} else {
    New-Item -ItemType Junction -Path $aliasRoot -Target $repoRoot | Out-Null
}
$generatedScript = Join-Path $aliasRoot 'embedded\firmware\sim\run_renode.generated.resc'

$logForRenode = (Join-Path $aliasRoot 'embedded\firmware\sim\uart.log').Replace('\', '/')
$contents = Get-Content -LiteralPath (Join-Path $simRoot 'run_renode.resc') -Raw
$contents = $contents.Replace('__AVS_UART_LOG__', $logForRenode)
Set-Content -LiteralPath $generatedScript -Value $contents -Encoding ascii

$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = ([Net.IPEndPoint]$listener.LocalEndpoint).Port
$listener.Stop()

$runStarted = Get-Date
$process = Start-Process -FilePath $renodeExe `
    -ArgumentList @('--disable-xwt', '--plain', '--port', "$port", $generatedScript) `
    -WorkingDirectory $aliasRoot -Wait -PassThru -NoNewWindow
if ($process.ExitCode -ne 0) {
    throw "Renode exited with code $($process.ExitCode)"
}
if ((Get-Item -LiteralPath $uartLog).LastWriteTime -lt $runStarted) {
    throw "Renode did not refresh $uartLog"
}
if (-not (Select-String -LiteralPath $uartLog -SimpleMatch 'AVS_TEST_PASS' -Quiet)) {
    throw "Renode completed without AVS_TEST_PASS. Inspect $uartLog"
}
Write-Output "Renode smoke test PASS; UART log: $uartLog"
