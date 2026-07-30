#!/usr/bin/env bash
# Tek komutluk kurulum: venv + bagimliliklar + Docker servisleri + depo hazirligi.
#
#     ./scripts/setup.sh
#
# Idempotent - tekrar tekrar calistirilabilir, var olani bozmaz.
# Ubuntu/Debian (ve WSL2 icindeki Ubuntu) icin. Sistem paketleri eksikse
# ne kurmaniz gerektigini soyler, kendisi sudo ile bir sey kurmaz.
set -euo pipefail

cd "$(dirname "$0")/.."
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[31m[HATA] %s\033[0m\n' "$*" >&2; exit 1; }

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
echo "requirements.txt kuruluyor (CUDA'li torch dahil, ilk seferde uzun surer)..."
pip install -q -r requirements.txt
echo "vLLM kuruluyor (yapisal filtreleme)..."
# Duz 'pip install vllm' CUDA 13 runtime bekleyen en guncel paketi cekip
# "libcudart.so.13" hatasiyla cokebiliyor (vLLM'in bilinen, duzeltilmemis
# paketleme sorunu). uv --torch-backend=auto mevcut CUDA suru
# munu tespit edip uyumlu paketi seciyor - vLLM'in resmi onerisi.
pip install -q uv
uv pip install -q -r requirements-serving.txt --torch-backend=auto || warn \
    "vLLM kurulamadi - Linux disi bir ortamda olabilirsiniz ya da CUDA surumu
     tespit edilemedi. Arama yine calisir, sadece yapisal filtreleme devre disi
     kalir. Elle deneyin: uv pip install vllm xgrammar --torch-backend=cu128"

# --- 3. .env ---------------------------------------------------------------
say "Yapilandirma"
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env olusturuldu - URETIMDE parolalari degistirin."
else
    echo ".env zaten var, dokunulmadi."
fi

# --- 4. Docker servisleri --------------------------------------------------
say "Altyapi servisleri (Qdrant, MinIO, Postgres, Kafka, Temporal)"
docker compose up -d
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
