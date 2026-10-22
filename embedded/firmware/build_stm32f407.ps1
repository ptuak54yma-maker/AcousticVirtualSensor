$ErrorActionPreference = 'Stop'

$firmwareRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $firmwareRoot '..\..')).Path
$aliasRoot = Join-Path $env:LOCALAPPDATA 'AVSRepoLink'
$toolchainRoot = Join-Path $env:LOCALAPPDATA 'ArmToolchains\12.2.MPACBTI-Rel1'
$toolchainBin = Join-Path $toolchainRoot 'bin'
$cmakeBin = 'C:\Program Files\CMake\bin'
$ninjaBin = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
$buildDir = Join-Path $env:LOCALAPPDATA 'AVSBuild'

# arm-none-eabi GCC on Windows cannot parse the repository's Vietnamese path
# through its current MSYS path conversion. Give CMake/GCC an ASCII-only alias.
if (Test-Path -LiteralPath $aliasRoot) {
    $existing = Get-Item -LiteralPath $aliasRoot -Force
    $existingTarget = if ($existing.Target -is [array]) { $existing.Target[0] } else { $existing.Target }
    if (-not ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        [IO.Path]::GetFullPath($existingTarget).TrimEnd('\') -ine $repoRoot.TrimEnd('\')) {
        throw "ASCII build alias already exists but does not target the repository: $aliasRoot"
    }
} else {
    New-Item -ItemType Junction -Path $aliasRoot -Target $repoRoot | Out-Null
}
$asciiFirmwareRoot = Join-Path $aliasRoot 'embedded\firmware'

foreach ($required in @(
    (Join-Path $toolchainBin 'arm-none-eabi-gcc.exe'),
    (Join-Path $toolchainBin 'arm-none-eabi-g++.exe'),
    (Join-Path $cmakeBin 'cmake.exe'),
    (Join-Path $ninjaBin 'ninja.exe')
)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required build tool not found: $required" }
}

& (Join-Path $asciiFirmwareRoot 'fetch_dependencies.ps1')
$env:PATH = "$toolchainBin;$cmakeBin;$ninjaBin;$env:PATH"

& (Join-Path $cmakeBin 'cmake.exe') -S $asciiFirmwareRoot -B $buildDir -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$(Join-Path $asciiFirmwareRoot 'arm-none-eabi-toolchain.cmake')" `
    "-DAVS_CMSIS_DSP_ROOT=$(Join-Path $asciiFirmwareRoot '_deps\CMSIS-DSP')" `
    "-DAVS_CMSIS_CORE_ROOT=$(Join-Path $asciiFirmwareRoot '_deps\CMSIS_6\CMSIS\Core')" `
    -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }

& (Join-Path $cmakeBin 'cmake.exe') --build $buildDir --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'Firmware build failed' }

$elf = Join-Path $buildDir 'avs_stm32f407.elf'
$outDir = Join-Path $firmwareRoot 'out'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Copy-Item -LiteralPath $elf -Destination (Join-Path $outDir 'avs_stm32f407.elf') -Force
Copy-Item -LiteralPath (Join-Path $buildDir 'avs_stm32f407.map') -Destination (Join-Path $outDir 'avs_stm32f407.map') -Force
& (Join-Path $toolchainBin 'arm-none-eabi-size.exe') $elf
& (Join-Path $toolchainBin 'arm-none-eabi-objdump.exe') -h $elf
Write-Output "ELF: $(Join-Path $outDir 'avs_stm32f407.elf')"
