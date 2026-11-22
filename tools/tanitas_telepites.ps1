# Tanító környezet telepítése (egyszer kell): külön Python-környezet a tanításhoz, GPU-s PyTorch-csal.
# A termékbe (motor) ebből semmi nem kerül; a tanítás végén csak egy .onnx fájl lesz.
#
# Futtatás a repó gyökerében (VS Code terminál):
#     powershell -ExecutionPolicy Bypass -File tools\tanitas_telepites.ps1
#
# Ha a "python" nem 3.10–3.12-es, add meg: -Python "C:\Python312\python.exe"
param(
    [string]$Python = "python",
    [string]$Cuda = "cu121"   # NVIDIA RTX 30xx/40xx: cu121 jó; CPU-ra: cpu (nagyon lassú tanítás)
)
$ErrorActionPreference = "Stop"
$venv = "venv-tanitas"

Write-Host "== Python-környezet: $venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    & $Python -m venv $venv
}
$py = "$venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Write-Host "== PyTorch ($Cuda)"
if ($Cuda -eq "cpu") {
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    & $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
}

Write-Host "== RF-DETR (Apache-2.0) + ONNX export"
& $py -m pip install "rfdetr[train,loggers]" onnx onnxsim pillow

Write-Host "== Ellenőrzés"
& $py -c "import torch, rfdetr; print('torch', torch.__version__, 'GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

Write-Host ""
Write-Host "Kész. Tanítás:"
Write-Host "  $py tools\tanitas.py adat --cimkek <export>\annotations\instances_default.json --kepek tanito\kepek --ki adat_rfdetr"
Write-Host "  $py tools\tanitas.py tanit --adat adat_rfdetr --ki futasok\elso --korok 40"
