#!/usr/bin/env bash
# Tek komutluk kurulum: venv + bagimliliklar + Docker servisleri + depo hazirligi.
#
#     ./scripts/setup.sh
#     ./scripts/setup.sh --offline /yol/offline_bundle   # internet YOK
#
# Idempotent - tekrar tekrar calistirilabilir, var olani bozmaz.
# Ubuntu/Debian (ve WSL2 icindeki Ubuntu) icin. Sistem paketleri eksikse
# ne kurmaniz gerektigini soyler, kendisi sudo ile bir sey kurmaz.
#
# --offline: internet olan bir makinede once
#     ./scripts/prepare_offline_bundle.sh
# calistirip cikan offline_bundle/ klasorunu (USB/ag ile) bu makineye
# tasiyin, sonra buraya yolunu verin. Docker/ffmpeg/git/python3'un
# KENDISI de sudo ile kurulmadigi icin bunlarin offline_bundle disinda,
# ayrica (ornegin apt-offline ile) hazirlanmis olmasi gerekir - detay
# docs/internetsiz-kurulum.md.
set -euo pipefail

cd "$(dirname "$0")/.."
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[31m[HATA] %s\033[0m\n' "$*" >&2; exit 1; }

OFFLINE_DIR=""
if [ "${1:-}" = "--offline" ]; then
    OFFLINE_DIR="${2:-}"
    [ -n "$OFFLINE_DIR" ] && [ -d "$OFFLINE_DIR" ] || die "--offline icin gecerli bir klasor yolu verin (bkz. scripts/prepare_offline_bundle.sh)."
    OFFLINE_DIR="$(cd "$OFFLINE_DIR" && pwd)"
    say "INTERNETSIZ KURULUM: $OFFLINE_DIR"
fi

# --- 1. Sistem on kosullari (kurmuyoruz, sadece kontrol) -------------------
say "Sistem on kosullari"
missing=()
for c in python3 ffmpeg ffprobe git docker; do
    command -v "$c" >/dev/null 2>&1 || missing+=("$c")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "Eksik: ${missing[*]}"
    echo
    echo "  sudo apt update"
    echo "  sudo apt install -y git ffmpeg python3-venv python3-pip curl"
    echo "  curl -fsSL https://get.docker.com | sudo sh"
    echo "  sudo usermod -aG docker \$USER && newgrp docker"
    die "Yukaridakileri kurup tekrar calistirin."
fi

PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' \
    || die "Python 3.11+ gerekiyor (bulunan: $PYV). Ubuntu 22.04'te:
  sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt install -y python3.11 python3.11-venv"
echo "python $PYV, ffmpeg, docker: tamam"

if ! nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi calismiyor - GPU gorunmuyor."
    warn "Ubuntu'da: sudo ubuntu-drivers autoinstall && sudo reboot"
    warn "WSL2'de: Windows tarafinda NVIDIA surucusunu guncelleyin (WSL'e CUDA KURMAYIN)."
    warn "GPU olmadan da devam edebilirsiniz ama ~15-20 kat yavas olur."
    read -r -p "Yine de devam edilsin mi? [e/H] " a
    [[ "$a" =~ ^[eE]$ ]] || exit 1
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/GPU: /'
fi

# --- 2. venv + Python bagimliliklari ---------------------------------------
say "Python ortami"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade -q pip

if [ -n "$OFFLINE_DIR" ]; then
    [ -d "$OFFLINE_DIR/wheels" ] || die "$OFFLINE_DIR/wheels yok - prepare_offline_bundle.sh dogru calisti mi?"
    echo "requirements.txt + requirements-serving.txt yerel wheel'lerden kuruluyor (--no-index)..."
    pip install -q --no-index --find-links="$OFFLINE_DIR/wheels" -r requirements.txt
    pip install -q --no-index --find-links="$OFFLINE_DIR/wheels" -r requirements-serving.txt || warn \
        "vLLM yerel wheel'lerden kurulamadi - prepare_offline_bundle.sh'in CIKTISI ile
         BU makinenin mimarisi/Python surumu uyusmuyor olabilir. Arama yine calisir,
         sadece yapisal filtreleme devre disi kalir."
else
    echo "requirements.txt kuruluyor (CUDA'li torch dahil, ilk seferde uzun surer)..."
    pip install -q -r requirements.txt
    echo "vLLM kuruluyor (yapisal filtreleme)..."
    # vLLM v0.20+ "libtorch stable ABI"ye gecti - kendi CUDA13 runtime'ina
    # sabit bagimli. Cozum eski surume sabitlemek DEGIL (0.8.3 -> torch==2.6.0
    # PyPI'dan kaldirilmis, dead-end): guncel vLLM'i ihtiyaci olan torch
    # surumuyle (2.11.0) ACIKCA birlikte istiyoruz - detay requirements-serving.txt'te.
    pip install -q uv
    uv pip install -q -r requirements-serving.txt --torch-backend=auto || warn \
        "vLLM kurulamadi - Linux disi bir ortamda olabilirsiniz ya da CUDA surumu
         tespit edilemedi. Arama yine calisir, sadece yapisal filtreleme devre disi
         kalir. Detay ve alternatifler: requirements-serving.txt"
fi

# --- 3. .env ---------------------------------------------------------------
say "Yapilandirma"
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env olusturuldu - URETIMDE parolalari degistirin."
else
    echo ".env zaten var, dokunulmadi."
fi

if [ -n "$OFFLINE_DIR" ]; then
    say "Model yollari .env'e yaziliyor (yerel, indirme yok)"
    python3 - <<PY
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
lines = setkv(lines, 'EMBEDDING_MODEL_DIR', '$OFFLINE_DIR/models/embedding')
lines = setkv(lines, 'YOLO_MODEL', '$OFFLINE_DIR/models/yolo/yolo26s.pt')
lines = setkv(lines, 'PARSE_MODEL', '$OFFLINE_DIR/models/parse')
p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
    echo "EMBEDDING_MODEL_DIR, YOLO_MODEL, PARSE_MODEL .env'de yerel yollara ayarlandi."
    if [ -d "$OFFLINE_DIR/models/vl" ]; then
        sed -i "s|^VLM_MODEL=.*|VLM_MODEL=$OFFLINE_DIR/models/vl|" .env 2>/dev/null || \
            echo "VLM_MODEL=$OFFLINE_DIR/models/vl" >> .env
        echo "VLM_MODEL de yerel yola ayarlandi (caption/rerank icin)."
    fi
fi

# --- 4. Docker servisleri --------------------------------------------------
say "Altyapi servisleri (Qdrant, MinIO, Postgres, Kafka, Temporal)"
if [ -n "$OFFLINE_DIR" ]; then
    echo "Docker imajlari yerel tar'lardan yukleniyor (internet YOK)..."
    for f in "$OFFLINE_DIR"/docker_images/*.tar; do
        echo "  yukleniyor: $(basename "$f")"
        docker load -i "$f" >/dev/null
    done
    docker compose -f docker-compose.yml -f docker-compose.offline.yml up -d
else
    docker compose up -d
fi
echo "Servislerin hazir olmasi bekleniyor..."
for _ in $(seq 1 60); do
    if python - <<'PY' >/dev/null 2>&1
from common.qdrant_store import get_client
get_client().get_collections()
PY
    then break; fi
    sleep 2
done

# --- 5. Depo/koleksiyon hazirligi ------------------------------------------
say "Depolama hazirligi"
python -m scripts.init_storage

# --- 6. Dogrulama ----------------------------------------------------------
say "Onucus kontrolu"
python -m scripts.check_env || warn "check_env hata bildirdi - yukariyi okuyun."

cat <<'EOF'

============================================================
KURULUM BITTI. Sirasiyla:

  source .venv/bin/activate

  1) Video ekleyin ve ingest edin  (vLLM KAPALI olmali - tum VRAM embedding'e)
       python -m scripts.register_video --dir ~/videolar/
       python -m scripts.ingest_all

  2) vLLM'i baslatin  (ayri terminal, VRAM'e gore modeli kendi secer)
       ./scripts/start_vllm.sh

  3) Arayin
       python -m scripts.query_cli --interactive

Detay ve sorun giderme: docs/sifirdan-kurulum.md
============================================================
EOF
