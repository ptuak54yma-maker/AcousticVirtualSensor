$ErrorActionPreference = 'Stop'

$simRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$firmwareRoot = Split-Path -Parent $simRoot
$repoRoot = (Resolve-Path (Join-Path $firmwareRoot '..\..')).Path
$aliasRoot = Join-Path $env:LOCALAPPDATA 'AVSRepoLink'
$renodeExe = Join-Path $env:LOCALAPPDATA 'RenodeExtracted\Renode\bin\Renode.exe'
if (-not (Test-Path -LiteralPath $renodeExe)) { throw "Renode not found: $renodeExe" }

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

$generatedScript = Join-Path $aliasRoot 'embedded\firmware\sim\run_renode_gui.generated.resc'
Copy-Item -LiteralPath (Join-Path $simRoot 'run_renode_gui.resc') `
    -Destination $generatedScript -Force
$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = ([Net.IPEndPoint]$listener.LocalEndpoint).Port
$listener.Stop()

$process = Start-Process -FilePath $renodeExe `
    -ArgumentList @('--port', "$port", $generatedScript) `
    -WorkingDirectory $aliasRoot -PassThru -WindowStyle Normal
Write-Output "Renode GUI started (PID $($process.Id)); select the UART4 analyzer to view MCU results."
