<#
.SYNOPSIS
  A TacAnaly elemző motor önálló (Python nélkül futó) buildje PyInstallerrel.

.DESCRIPTION
  1. külön virtuális környezet (packaging\.venv-<torch>)
  2. PyTorch telepítése: CPU vagy CUDA változat (-Torch kapcsoló)
  3. a többi függőség (requirements.txt + pyinstaller, psutil, lap)
  4. pyinstaller packaging\tacanaly_engine.spec  ->  packaging\dist\tacanaly_engine\
  5. opcionálisan az ffmpeg.exe bemásolása (-FfmpegExe) és egy füstteszt (READY sor, /health, /shutdown)

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\build_engine.ps1 -Torch cpu -FfmpegExe C:\tools\ffmpeg-lgpl\bin\ffmpeg.exe
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\build_engine.ps1 -Torch cu124 -Clean
#>
param(
    # cpu = bármely gépen fut (integrált GPU-s laptop); cu118 / cu121 / cu124 / cu126 = NVIDIA CUDA build (nagy: több GB)
    [ValidateSet("cpu", "cu118", "cu121", "cu124", "cu126")]
    [string]$Torch = "cpu",
    # A Python indító (py launcher). A torch-hoz a 3.10-3.12 ajánlott.
    [string]$Python = "py -3.11",
    # Mellékelendő ffmpeg.exe (LGPL build ajánlott, lásd README_CSOMAGOLAS.md – Licencek)
    [string]$FfmpegExe = "",
    # A venv és a korábbi build törlése
    [switch]$Clean,
    # Füstteszt kihagyása
    [switch]$NoSmokeTest
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Venv = Join-Path $PSScriptRoot ".venv-$Torch"
$Dist = Join-Path $PSScriptRoot "dist"
$Work = Join-Path $PSScriptRoot "build"
$OutDir = Join-Path $Dist "tacanaly_engine"

Push-Location $Root
try {
    if ($Clean) {
        foreach ($d in @($Venv, $Dist, $Work)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d } }
    }

    if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
        Write-Host "==> Virtuális környezet: $Venv"
        $pyParts = $Python.Split(" ")
        & $pyParts[0] $pyParts[1..($pyParts.Length - 1)] -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "A venv nem hozható létre ($Python)" }
    }
    $Py = Join-Path $Venv "Scripts\python.exe"
    & $Py -m pip install --upgrade pip wheel
    if ($LASTEXITCODE -ne 0) { throw "pip frissítés sikertelen" }

    Write-Host "==> PyTorch ($Torch)"
    & $Py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Torch"
    if ($LASTEXITCODE -ne 0) { throw "A torch telepítése sikertelen" }

    Write-Host "==> Függőségek"
    & $Py -m pip install -r requirements.txt pyinstaller psutil lap
    if ($LASTEXITCODE -ne 0) { throw "A függőségek telepítése sikertelen" }
    # A csomagba a GUI nélküli OpenCV kerüljön (kisebb, nincs benne Qt)
    & $Py -m pip uninstall -y opencv-python 2>$null
    & $Py -m pip install opencv-python-headless
    if ($LASTEXITCODE -ne 0) { throw "opencv-python-headless telepítése sikertelen" }

    Write-Host "==> PyInstaller"
    & $Py -m PyInstaller --noconfirm --clean --distpath $Dist --workpath $Work (Join-Path $PSScriptRoot "tacanaly_engine.spec")
    if ($LASTEXITCODE -ne 0) { throw "A PyInstaller build sikertelen" }

    if ($FfmpegExe) {
        $ffDir = Join-Path $OutDir "ffmpeg"
        New-Item -ItemType Directory -Force -Path $ffDir | Out-Null
        Copy-Item $FfmpegExe (Join-Path $ffDir "ffmpeg.exe") -Force
        $lic = Join-Path (Split-Path $FfmpegExe) "..\LICENSE.txt"
        if (Test-Path $lic) { Copy-Item $lic (Join-Path $ffDir "LICENSE.txt") -Force }
        Write-Host "    ffmpeg: $ffDir\ffmpeg.exe"
    } else {
        Write-Warning "Nincs -FfmpegExe: a telepítőnek kell az ffmpeg.exe-t az engine\ffmpeg\ mappába tennie."
    }

    $Exe = Join-Path $OutDir "tacanaly_engine.exe"
    if (-not $NoSmokeTest) {
        Write-Host "==> Füstteszt"
        $data = Join-Path $env:TEMP ("tacanaly_smoke_" + [guid]::NewGuid().ToString("N"))
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.Arguments = "--port 0 --preload none --data-dir `"$data`""
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.CreateNoWindow = $true
        $p = [System.Diagnostics.Process]::Start($psi)
        $line = $p.StandardOutput.ReadLine()
        if ($line -notmatch "^TACANALY_READY port=(\d+) pid=(\d+)") { $p.Kill(); throw "Nincs READY sor: $line" }
        $port = $Matches[1]
        $health = Invoke-RestMethod "http://127.0.0.1:$port/health"
        Write-Host ("    /health: status={0} version={1} ffmpeg={2}" -f $health.status, $health.version, $health.ffmpeg)
        Invoke-RestMethod -Method Post "http://127.0.0.1:$port/shutdown" | Out-Null
        if (-not $p.WaitForExit(20000)) { $p.Kill(); throw "A motor nem állt le 20 mp alatt" }
        Remove-Item -Recurse -Force $data -ErrorAction SilentlyContinue
        Write-Host "    OK"
    }

    $size = (Get-ChildItem $OutDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ("==> Kész: {0} ({1:N0} MB)" -f $OutDir, $size)
}
finally {
    Pop-Location
}
