$ErrorActionPreference = 'Stop'

$firmwareRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dependencyRoot = Join-Path $firmwareRoot '_deps'
New-Item -ItemType Directory -Force -Path $dependencyRoot | Out-Null

$dependencies = @(
    @{ Name = 'CMSIS-DSP'; Url = 'https://github.com/ARM-software/CMSIS-DSP.git'; Tag = 'v1.16.2'; Commit = 'd5717e454fec0337bef114a21f1d2d01d74f2701' },
    @{ Name = 'CMSIS_6'; Url = 'https://github.com/ARM-software/CMSIS_6.git'; Tag = 'v6.2.0'; Commit = '6f0a58d01aa9bd2feba212097f9afe7acd991d52' }
)

foreach ($dependency in $dependencies) {
    $path = Join-Path $dependencyRoot $dependency.Name
    if (-not (Test-Path -LiteralPath (Join-Path $path '.git'))) {
        git clone --depth 1 --branch $dependency.Tag $dependency.Url $path
        if ($LASTEXITCODE -ne 0) { throw "Failed to fetch $($dependency.Name)" }
    }
    $actual = (git -C $path rev-parse HEAD).Trim()
    if ($actual -ne $dependency.Commit) {
        throw "$($dependency.Name) revision mismatch: expected $($dependency.Commit), got $actual"
    }
    Write-Output "$($dependency.Name): $actual"
}
