# One-time setup of Track B on this Windows PC (run via "Setup Track B.bat").
# Safe to run again: every step checks first and skips what is already done.
# Everything big goes on the drive this repo lives on (D:), not C:.
# Works in Windows PowerShell 5.1 (Windows 10).

# 'Continue', not 'Stop': in Windows PowerShell 5.1 any stderr text from python/pip would
# otherwise abort the script. Every native step checks $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$Root = 'D:\Video-Bola-Analis'          # videos, models, results (same as "Jalankan Track B.bat")

function Step($text) { Write-Host ''; Write-Host "==> $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host ''; Write-Host "STOPPED: $text" -ForegroundColor Red; exit 1 }

# Keep pip's downloads and temp files off C: (torch alone is ~2.7 GB).
$Drive = Split-Path -Qualifier $PSScriptRoot
$Tmp = Join-Path $Drive '\pip-tmp'
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
$env:TMP = $Tmp
$env:TEMP = $Tmp
$env:PIP_CACHE_DIR = Join-Path $Drive '\pip-cache'

Step 'Python 3.10 / 3.11'
$PyVer = $null
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host 'The Python launcher (py) is missing - installing Python 3.10 with winget ...'
    winget install -e --id Python.Python.3.10 --accept-package-agreements --accept-source-agreements
    Fail 'Python 3.10 was just installed. Close this window and double-click "Setup Track B.bat" again.'
}
foreach ($v in @('3.10', '3.11')) {
    & py "-$v" -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $PyVer = $v; break }
}
if (-not $PyVer) {
    Write-Host 'Python 3.10 not found - installing it with winget ...'
    winget install -e --id Python.Python.3.10 --accept-package-agreements --accept-source-agreements
    Fail 'Python 3.10 was just installed. Close this window and double-click "Setup Track B.bat" again.'
}
Write-Host "Using Python $PyVer"

Step 'Virtual environment (.venv)'
if (Test-Path '.venv\Scripts\python.exe') {
    $have = & '.venv\Scripts\python.exe' -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if (@('3.10', '3.11') -notcontains $have) {
        Write-Host "Existing .venv uses Python $have; moving it aside to .venv-old"
        if (Test-Path '.venv-old') { Remove-Item -Recurse -Force '.venv-old' }
        Rename-Item '.venv' '.venv-old'
    } else {
        Write-Host "Keeping the existing .venv (Python $have)"
    }
}
if (-not (Test-Path '.venv\Scripts\python.exe')) { & py "-$PyVer" -m venv .venv }
$Py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $Py -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Fail 'pip upgrade failed (internet?).' }

Step 'PyTorch with CUDA 11.8 (matches the NVIDIA driver; ~2.7 GB download)'
$cuda = & $Py -c "import torch; print(torch.version.cuda or 'cpu')" 2>$null
if ($cuda -ne '11.8') {
    & $Py -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) { Fail 'PyTorch install failed. Check free space on D: and the internet, then run again.' }
} else {
    Write-Host 'Already installed.'
}

Step 'Track B packages (ultralytics, supervision, trackers, roboflow/sports, ...)'
# The constraints file stops pip from swapping the CUDA build of torch for the CPU-only one.
$Pin = Join-Path $Tmp 'torch-pin.txt'
Set-Content -Path $Pin -Value @('torch==2.6.0', 'torchvision==0.21.0') -Encoding ASCII
& $Py -m pip install -r requirements.txt 'git+https://github.com/roboflow/sports.git' -c $Pin
if ($LASTEXITCODE -ne 0) { Fail 'Package install failed. Is Git installed and on PATH (git --version)?' }

Step 'FFmpeg and Deno (needed to download YouTube videos)'
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
    Write-Host 'FFmpeg installed - it is usable from a NEW window.'
} else { Write-Host 'FFmpeg found.' }
if (-not (Get-Command deno -ErrorAction SilentlyContinue)) {
    winget install -e --id DenoLand.Deno --accept-package-agreements --accept-source-agreements
} else { Write-Host 'Deno found.' }

Step 'Check: GPU visible to PyTorch'
& $Py -c "import torch, ultralytics, supervision, trackers, sports, umap, psutil; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA', ok, '|', torch.cuda.get_device_name(0) if ok else 'no GPU'); import sys; sys.exit(0 if ok else 3)"
if ($LASTEXITCODE -eq 3) { Fail 'PyTorch cannot see the GPU. Update the NVIDIA driver (GeForce Experience) and run this again.' }
if ($LASTEXITCODE -ne 0) { Fail 'A package failed to import; see the error above.' }

Step "Football models -> $Root\models"
New-Item -ItemType Directory -Force -Path "$Root\models" | Out-Null
& $Py scripts\download_sports_models.py --out "$Root\models"
if ($LASTEXITCODE -ne 0) { Fail 'Model download failed; run this setup again later (Google Drive sometimes rate-limits).' }

Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host 'Next: fill the period times in matches\persiba_2026-27.json, then double-click "Jalankan Track B.bat".'
