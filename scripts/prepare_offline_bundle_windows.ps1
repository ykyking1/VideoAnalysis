<#
Internetli bir WINDOWS makinesinde calistirilir. scripts/prepare_offline_bundle.sh
(Linux hazirlik makinesi) zaten calistirilmis olmali - onun urettigi
docker_images\ ve models\ klasorleri AYNEN kullanilabilir (Docker imajlari
linux/amd64 oldugu icin Docker Desktop'in Windows'taki WSL2 motorunda da
calisir; model agirliklari sadece veri dosyasi, isletim sisteminden bagimsiz).

Bu script SADECE Windows'a ozgu eksik parcalari indirir: win_amd64 Python
wheel'leri + ffmpeg/git/Docker Desktop/Python kurulum dosyalari.

    powershell -ExecutionPolicy Bypass -File scripts\prepare_offline_bundle_windows.ps1

Cikti: .\offline_bundle_windows\ (wheels_windows\, installers\). Bunu, Linux
hazirliginin ciktisi olan offline_bundle\ klasorunun ICINE kopyalayip TEK bir
klasor haline getirin (docker_images\, models\ ile ayni seviyeye -
offline_bundle\ icindeki eski Linux wheels\/system_packages\'a DOKUNMAYIN,
kullanilmayacak ama zararsizlar), sonra hedef Windows makinesinde
scripts\setup_windows.ps1'e o birlesik klasoru verin.

NEDEN AYRI BIR SCRIPT: requirements-serving.txt (vLLM) icin Windows wheel'i
YOK - vLLM resmi olarak Windows'u desteklemiyor. Bu yuzden vLLM native
Windows'ta pip ile KURULMUYOR; Docker container'da calistiriliyor (Linux
hazirliginda zaten indirilmis vllm/vllm-openai imaji kullanilir - bkz.
docker-compose.offline.yml'deki vllm servisi ve
docs/internetsiz-kurulum-windows.md).
#>
[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11.9",
    [string]$OutDir = ".\offline_bundle_windows"
)

$ErrorActionPreference = "Stop"

function Say($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }

New-Item -ItemType Directory -Force -Path "$OutDir\wheels_windows" | Out-Null
New-Item -ItemType Directory -Force -Path "$OutDir\installers" | Out-Null

# --- 0. Hedef Python surumunun bu makinede de kurulu olmasi lazim ----------
# NEDEN: pip download, CALISTIGI Python'un platform/ABI etiketine gore wheel
# secer. Hedef makinede Python 3.11 kullanilacaksa, wheel'leri de BURADA
# Python 3.11 ile indirmek gerekir - farkli surumle inen wheel hedefte
# kurulamayabilir (ABI uyusmazligi).
$pyShortVer = ($PythonVersion -split '\.')[0..1] -join '.'
& py "-$pyShortVer" --version 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Bu makinede 'py -$pyShortVer' bulunamadi. Python $pyShortVer'i (ya da hedefte kullanacaginiz surumu) once bu makineye kurun - wheel'ler onun ABI'siyle inmeli."
}

# --- 1. Windows Python wheel'leri (sadece requirements.txt - CORE hat) -----
Say "Python paketleri indiriliyor (win_amd64, Python $pyShortVer) -> $OutDir\wheels_windows"
& py "-$pyShortVer" -m pip install --quiet --upgrade pip
& py "-$pyShortVer" -m pip download -r requirements.txt -d "$OutDir\wheels_windows"
if ($LASTEXITCODE -ne 0) { throw "pip download basarisiz oldu (yukaridaki pip ciktisina bakin)." }
$wheelCount = (Get-ChildItem "$OutDir\wheels_windows").Count
Write-Host "$wheelCount dosya indirildi"
Warn "requirements-serving.txt (vLLM) BILEREK atlandi - Windows wheel'i yok, vLLM resmi desteklenmiyor. vLLM Docker container'da calisacak."

# --- 2. ffmpeg (statik, NVENC/NVDEC dahil - BtbN GPL build) ----------------
Say "ffmpeg indiriliyor -> $OutDir\installers"
Invoke-WebRequest -Uri "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip" `
    -OutFile "$OutDir\installers\ffmpeg-win64.zip"

# --- 3. Git for Windows (64-bit standalone installer) ----------------------
Say "Git for Windows indiriliyor -> $OutDir\installers"
$gitRelease = Invoke-RestMethod -Uri "https://api.github.com/repos/git-for-windows/git/releases/latest"
$gitAsset = $gitRelease.assets | Where-Object { $_.name -like "Git-*-64-bit.exe" } | Select-Object -First 1
if (-not $gitAsset) {
    Warn "Git for Windows indirme linki bulunamadi (GitHub API yaniti degismis olabilir) - elle indirin: https://git-scm.com/download/win"
} else {
    Invoke-WebRequest -Uri $gitAsset.browser_download_url -OutFile "$OutDir\installers\$($gitAsset.name)"
}

# --- 4. Docker Desktop -------------------------------------------------------
Say "Docker Desktop indiriliyor -> $OutDir\installers"
Invoke-WebRequest -Uri "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe" `
    -OutFile "$OutDir\installers\DockerDesktopInstaller.exe"
Warn "Docker Desktop, WSL2 arka ucu icin 'Windows Subsystem for Linux' ve 'Virtual Machine Platform' Windows ozelliklerini ister - bunlar kapaliysa kurulum bir YENIDEN BASLATMA isteyebilir. setup_windows.ps1 bunu otomatik yapmaz, elle onaylamaniz gerekir."

# --- 5. Python kurulum dosyasi (hedefte Python hic yoksa) -------------------
Say "Python $PythonVersion kurulum dosyasi indiriliyor -> $OutDir\installers"
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe" `
    -OutFile "$OutDir\installers\python-$PythonVersion-amd64.exe"

Say "Tamam"
$sizeGB = (Get-ChildItem -Recurse $OutDir | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("Paket boyutu: {0:N2} GB" -f $sizeGB)
Write-Host ""
Write-Host "Simdi bu klasoru ($OutDir), Linux hazirliginin urettigi offline_bundle\"
Write-Host "klasorunun ICINE kopyalayip birlestirin (docker_images\, models\ ile ayni"
Write-Host "seviyeye - wheels_windows\ ve installers\ eklenecek), sonra hedef Windows"
Write-Host "makinesinde:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Bundle <birlesik-klasor>"
