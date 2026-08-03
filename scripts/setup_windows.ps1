<#
Hedef, INTERNETSIZ Windows makinesinde calistirilir. Birlesik bundle klasorunu
(Linux hazirliginin docker_images\ + models\'i ile Windows hazirliginin
wheels_windows\ + installers\'i birlestirilmis hali) bekler.

    powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Bundle C:\yol\offline_bundle

vLLM (yapisal ayristirma/caption/rerank) bu script ile KURULMAZ - Windows
wheel'i yok. Ayri, opsiyonel bir adim olarak Docker container'da baslatilir
(bu script sonunda komutu yazdirir). Onsuz da arama calisir, sadece yapisal
filtreler devre disi kalir.

NEDEN test.venv/pip yerine dogrudan .venv\Scripts\python.exe cagriliyor:
venv'in Activate.ps1'i bazi sistemlerde ExecutionPolicy tarafindan
engellenebiliyor - dogrudan yol vermek bu sorunu tamamen atlar.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Bundle,
    [string]$PythonVersion = "3.11",
    [switch]$InstallMissing
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $RepoRoot

function Say($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Die($msg) { Write-Host "[HATA] $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $Bundle)) { Die "$Bundle bulunamadi." }
$Bundle = (Resolve-Path $Bundle).Path
Say "INTERNETSIZ KURULUM (Windows): $Bundle"

foreach ($sub in @("docker_images", "models", "wheels_windows")) {
    if (-not (Test-Path "$Bundle\$sub")) {
        Die "$Bundle\$sub yok - prepare_offline_bundle.sh (Linux) VE prepare_offline_bundle_windows.ps1 (Windows) ciktilarini birlestirdiniz mi?"
    }
}

# --- 1. Sistem on kosullari (kurmuyoruz, sadece kontrol/opsiyonel kurulum) --
Say "Sistem on kosullari"

$missing = @()
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { $missing += "git" }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { $missing += "ffmpeg" }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { $missing += "docker" }
$pyOk = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py "-$PythonVersion" --version 2>$null
    if ($LASTEXITCODE -eq 0) { $pyOk = $true }
}
if (-not $pyOk) { $missing += "python$PythonVersion" }

if ($missing.Count -gt 0) {
    Write-Host "Eksik: $($missing -join ', ')"
    if (-not $InstallMissing) {
        Write-Host ""
        Write-Host "Kurulum dosyalari $Bundle\installers icinde. Otomatik denemek icin:"
        Write-Host "  .\scripts\setup_windows.ps1 -Bundle $Bundle -InstallMissing"
        Write-Host "(Docker Desktop WSL2 kurulumu yeniden baslatma isteyebilir - o adim otomatik degil, elle onaylamaniz gerekebilir.)"
        Die "Once yukaridakileri kurup tekrar calistirin."
    }

    if ($missing -contains "python$PythonVersion") {
        Say "Python $PythonVersion kuruluyor (sessiz)"
        $pyInstaller = Get-ChildItem "$Bundle\installers\python-*-amd64.exe" | Select-Object -First 1
        if (-not $pyInstaller) { Die "Python kurulum dosyasi bulunamadi ($Bundle\installers icinde)." }
        Start-Process -FilePath $pyInstaller.FullName -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1" -Wait
        Warn "Python kuruldu - PATH'in etkili olmasi icin bu terminali kapatip yeniden acmaniz GEREKEBILIR."
    }
    if ($missing -contains "git") {
        Say "Git for Windows kuruluyor (sessiz)"
        $gitInstaller = Get-ChildItem "$Bundle\installers\Git-*-64-bit.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $gitInstaller) { Warn "Git kurulum dosyasi bulunamadi, atlaniyor - elle kurun: https://git-scm.com/download/win" }
        else { Start-Process -FilePath $gitInstaller.FullName -ArgumentList "/VERYSILENT /NORESTART" -Wait }
    }
    if ($missing -contains "ffmpeg") {
        Say "ffmpeg kuruluyor -> C:\ffmpeg"
        $zip = "$Bundle\installers\ffmpeg-win64.zip"
        if (-not (Test-Path $zip)) { Die "$zip yok." }
        Expand-Archive -Path $zip -DestinationPath "C:\ffmpeg_tmp" -Force
        $binDir = (Get-ChildItem "C:\ffmpeg_tmp" -Directory | Select-Object -First 1).FullName + "\bin"
        if (Test-Path "C:\ffmpeg") { Remove-Item "C:\ffmpeg" -Recurse -Force }
        Move-Item $binDir "C:\ffmpeg"
        Remove-Item "C:\ffmpeg_tmp" -Recurse -Force
        $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
        if ($userPath -notlike "*C:\ffmpeg*") {
            [Environment]::SetEnvironmentVariable("PATH", "$userPath;C:\ffmpeg", "User")
        }
        $env:PATH = "$env:PATH;C:\ffmpeg"
        Write-Host "ffmpeg kuruldu: C:\ffmpeg (PATH'e eklendi)"
    }
    if ($missing -contains "docker") {
        Say "Docker Desktop kuruluyor (sessiz)"
        $dockerInstaller = "$Bundle\installers\DockerDesktopInstaller.exe"
        if (-not (Test-Path $dockerInstaller)) { Die "$dockerInstaller yok." }
        Start-Process -FilePath $dockerInstaller -ArgumentList "install --quiet --accept-license --backend=wsl2" -Wait
        Warn "Docker Desktop kuruldu - WSL2 ozellikleri yeni etkinlestirildiyse YENIDEN BASLATMA gerekebilir. Yeniden baslattiktan sonra Docker Desktop'i acip hazir olmasini bekleyin, sonra bu script'i TEKRAR calistirin."
        exit 0
    }

    Warn "Bazi bilesenler yeni kuruldu - PATH guncellemesinin etkili olmasi icin bu terminali kapatip yeniden acin, sonra script'i tekrar calistirin."
    exit 0
}
Write-Host "python, git, ffmpeg, docker: tamam"

docker info *>$null
if ($LASTEXITCODE -ne 0) {
    Die "Docker Desktop kurulu ama calismiyor - once Docker Desktop'i acip 'Engine running' durumunu bekleyin."
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Warn "nvidia-smi bulunamadi - GPU gorunmuyor. NVIDIA surucusunun onceden kurulu olmasi gerekiyor (bu script'in kapsami disinda)."
}

# --- 2. venv + Python bagimliliklari (sadece requirements.txt - CORE hat) --
Say "Python ortami"
if (-not (Test-Path ".venv")) {
    & py "-$PythonVersion" -m venv .venv
}
$venvPy = ".\.venv\Scripts\python.exe"
& $venvPy -m pip install --quiet --upgrade pip
Write-Host "requirements.txt yerel wheel'lerden kuruluyor (--no-index)..."
& $venvPy -m pip install --no-index --find-links="$Bundle\wheels_windows" -r requirements.txt
if ($LASTEXITCODE -ne 0) { Die "pip install basarisiz oldu (yukaridaki cikti)." }
Warn "vLLM (requirements-serving.txt) bu makineye KURULMADI - Windows wheel'i yok. Yapisal filtreleme/caption/rerank icin asagidaki Docker adimini kullanin."

# --- 3. .env -----------------------------------------------------------------
Say "Yapilandirma"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ".env olusturuldu - URETIMDE parolalari degistirin."
} else {
    Write-Host ".env zaten var, dokunulmadi."
}

Say "Model yollari .env'e yaziliyor (yerel, indirme yok)"
# NOT: PowerShell'de bash tarzi '<<MARKER' heredoc YOK - here-string (@"..."@)
# ile Python script'ini olusturup stdin uzerinden 'python -'e veriyoruz.
# PARSE_MODEL BILEREK degistirilmiyor (HF-ID kalmali) - vLLM Docker'da
# --served-model-name ile ayni HF-ID'yi sunuyor, gercek agirlik yolu
# OFFLINE_BUNDLE_DIR ile ayri veriliyor. Bkz. docker-compose.offline.yml.
$pyScript = @"
import pathlib, re
p = pathlib.Path('.env')
lines = p.read_text(encoding='utf-8').splitlines()
def setkv(lines, k, v):
    out, done = [], False
    for l in lines:
        if re.match(rf'^{k}=', l):
            out.append(f'{k}={v}'); done = True
        else:
            out.append(l)
    if not done: out.append(f'{k}={v}')
    return out
lines = setkv(lines, 'EMBEDDING_MODEL_DIR', r'$Bundle\models\embedding')
lines = setkv(lines, 'YOLO_MODEL', r'$Bundle\models\yolo\yolo26s.pt')
lines = setkv(lines, 'OFFLINE_BUNDLE_DIR', r'$Bundle')
p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
"@
$pyScript | & $venvPy -
if ($LASTEXITCODE -ne 0) { Die ".env'e model yollari yazilamadi (yukaridaki Python hatasi)." }
Write-Host "EMBEDDING_MODEL_DIR, YOLO_MODEL, OFFLINE_BUNDLE_DIR .env'de ayarlandi."
if (Test-Path "$Bundle\models\vl") {
    Warn "models\vl bulundu ama bu script VLM_MODEL/caption'i otomatik baglamiyor (kapsami disinda) - istersen docker-compose.offline.yml'deki vllm servisini ornek alip ikinci bir container/profil ekleyebilirsin."
}

# --- 4. Docker servisleri ------------------------------------------------------
Say "Altyapi servisleri (Qdrant, MinIO, Postgres, Kafka, Temporal)"
Write-Host "Docker imajlari yerel tar'lardan yukleniyor (internet YOK)..."
Get-ChildItem "$Bundle\docker_images\*.tar" | ForEach-Object {
    Write-Host "  yukleniyor: $($_.Name)"
    docker load -i $_.FullName | Out-Null
}
docker compose -f docker-compose.yml -f docker-compose.offline.yml up -d
if ($LASTEXITCODE -ne 0) { Die "docker compose up basarisiz oldu (yukaridaki cikti)." }

Write-Host "Servislerin hazir olmasi bekleniyor..."
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    & $venvPy -c "from common.qdrant_store import get_client; get_client().get_collections()" 2>$null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $ready) { Warn "Qdrant 2 dakikada hazir olmadi - 'docker compose logs qdrant' ile kontrol edin." }

# --- 5. Depo/koleksiyon hazirligi ------------------------------------------
Say "Depolama hazirligi"
& $venvPy -m scripts.init_storage

# --- 6. Dogrulama ------------------------------------------------------------
Say "Onkosul kontrolu"
& $venvPy -m scripts.check_env
if ($LASTEXITCODE -ne 0) { Warn "check_env hata bildirdi - yukariyi okuyun." }

Write-Host ""
Write-Host "============================================================"
Write-Host "KURULUM BITTI. Sirasiyla:"
Write-Host ""
Write-Host "  1) Video ekleyin ve ingest edin  (vLLM KAPALI olmali - tum VRAM embedding'e)"
Write-Host "       .\.venv\Scripts\python.exe -m scripts.register_video --dir C:\videolar\"
Write-Host "       .\.venv\Scripts\python.exe -m scripts.ingest_all"
Write-Host ""
Write-Host "  2) vLLM'i baslatin (Windows'ta start_vllm.sh YERINE Docker profili -"
Write-Host "     vLLM'in Windows wheel'i yok, container'da calisiyor):"
Write-Host "       docker compose --profile gpu -f docker-compose.yml -f docker-compose.offline.yml up -d vllm"
Write-Host ""
Write-Host "  3) Arayin"
Write-Host "       .\.venv\Scripts\python.exe -m scripts.query_cli --interactive"
Write-Host ""
Write-Host "Detay: docs\internetsiz-kurulum-windows.md"
Write-Host "============================================================"
