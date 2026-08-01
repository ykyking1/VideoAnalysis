#!/usr/bin/env bash
# İNTERNET OLAN bir Linux makinede çalıştırılır - hedef (internetsiz) makineye
# taşınacak her şeyi (sistem paketleri, Python paketleri, Docker imajları,
# model ağırlıkları) tek bir klasöre indirir.
#
#     ./scripts/prepare_offline_bundle.sh [--with-vlm]
#
# --with-vlm: caption/rerank için Qwen2.5-VL-7B-Instruct-AWQ'yu da indirir
#             (~6.5 GB ek, sadece caption/rerank kullanacaksanız gerekli).
#
# KRİTİK ÖNKOŞUL: Bu makine hedef makineyle **AYNI Ubuntu sürümü** olmalı
# (ör. ikisi de 24.04) - .deb paketleri sürüme özgüdür, uyumsuz sürümden
# kurulum sessizce bozuk bir sisteme yol açabilir. Ayrıca aynı mimaride
# olmalı (x86_64 Linux, aynı Python sürümü) - pip wheel'leri platforma özgü.
#
# Çıktı: ./offline_bundle/ (bunu USB/ağ ile hedef makineye taşıyın, sonra
# orada ÖNCE: ./scripts/install_system_offline.sh (Python henüz yoksa gerekli)
# SONRA: ./scripts/setup.sh --offline ./offline_bundle)
set -euo pipefail
cd "$(dirname "$0")/.."

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[31m[HATA] %s\033[0m\n' "$*" >&2; exit 1; }

WITH_VLM=0
[ "${1:-}" = "--with-vlm" ] && WITH_VLM=1

BUNDLE="$(pwd)/offline_bundle"
mkdir -p "$BUNDLE"/{system_packages,wheels,docker_images,models/embedding,models/yolo,models/parse}
[ "$WITH_VLM" = 1 ] && mkdir -p "$BUNDLE/models/vl"

command -v docker >/dev/null 2>&1 || die "docker gerekiyor (imajlari indirip kaydetmek icin)."
command -v python3 >/dev/null 2>&1 || die "python3 gerekiyor (bu makinede - hedefte olmayabilir, asagida onun icin de paket indiriliyor)."

# --- 0. Sistem paketleri (hedefte python3/git/ffmpeg hic olmayabilir) ------
say "Sistem paketleri indiriliyor (.deb) -> $BUNDLE/system_packages"
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install --download-only -y -o Dir::Cache::Archives="$BUNDLE/system_packages" \
        python3 python3-venv python3-pip git ffmpeg 2>&1 | tail -5
    echo "$(ls "$BUNDLE/system_packages"/*.deb 2>/dev/null | wc -l) .deb dosyasi indirildi"
else
    warn "apt-get yok - sistem paketleri atlandi. Hedefte python3/git/ffmpeg zaten kuruluysa sorun degil."
fi

say "Docker Engine statik ikilileri indiriliyor (apt/repo eslemesi gerektirmez) -> $BUNDLE/system_packages"
DOCKER_TGZ="$BUNDLE/system_packages/docker-static.tgz"
curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz" -o "$DOCKER_TGZ" \
    || warn "Docker statik ikilisi indirilemedi - hedefte Docker zaten kuruluysa sorun degil, degilse elle indirin: https://download.docker.com/linux/static/stable/x86_64/"

# Statik tarball'da docker-compose (v2 plugin) YOK - ayri indirilmesi lazim,
# yoksa "docker compose" komutu hedefte "unknown command" verir (bu proje
# her yerde docker compose v2 kullaniyor, standalone docker-compose degil).
say "docker compose (v2) eklentisi indiriliyor -> $BUNDLE/system_packages"
COMPOSE_BIN="$BUNDLE/system_packages/docker-compose-plugin"
curl -fsSL "https://github.com/docker/compose/releases/download/v2.32.4/docker-compose-linux-x86_64" -o "$COMPOSE_BIN" \
    || warn "docker-compose eklentisi indirilemedi - hedefte zaten kuruluysa sorun degil, degilse elle indirin: https://github.com/docker/compose/releases"

# --- 1. Python paketleri (wheel'ler) ----------------------------------------
say "Python paketleri indiriliyor -> $BUNDLE/wheels"
pip install -q --upgrade pip
pip download -r requirements.txt -d "$BUNDLE/wheels"
pip download -r requirements-serving.txt -d "$BUNDLE/wheels"
echo "$(ls "$BUNDLE/wheels" | wc -l) dosya indirildi"

# --- 2. Docker imajlari ------------------------------------------------------
say "Docker imajlari cekiliyor ve kaydediliyor -> $BUNDLE/docker_images"
IMAGES=(
    minio/minio:latest
    qdrant/qdrant:latest
    postgres:16
    apache/kafka:latest
    temporalio/auto-setup:latest
    temporalio/ui:latest
    vllm/vllm-openai:latest
)
for img in "${IMAGES[@]}"; do
    echo "  cekiliyor: $img"
    docker pull -q "$img" >/dev/null
    fname="$BUNDLE/docker_images/$(echo "$img" | tr '/:' '__').tar"
    docker save "$img" -o "$fname"
    echo "  kaydedildi: $fname ($(du -h "$fname" | cut -f1))"
done

# --- 3. Model agirliklari ----------------------------------------------------
say "Embedding modeli indiriliyor (Qwen3-VL-Embedding-2B, ~4 GB) -> $BUNDLE/models/embedding"
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('Qwen/Qwen3-VL-Embedding-2B', local_dir='$BUNDLE/models/embedding')
print('indirildi:', path)
"

say "YOLO26s agirligi indiriliyor (~20 MB) -> $BUNDLE/models/yolo"
python3 -c "
import urllib.request
url = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt'
dst = '$BUNDLE/models/yolo/yolo26s.pt'
urllib.request.urlretrieve(url, dst)
print('indirildi:', dst)
"

say "vLLM ayristirma modeli indiriliyor (Qwen2.5-7B-Instruct-AWQ, ~5.2 GB) -> $BUNDLE/models/parse"
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('Qwen/Qwen2.5-7B-Instruct-AWQ', local_dir='$BUNDLE/models/parse')
print('indirildi:', path)
"

if [ "$WITH_VLM" = 1 ]; then
    say "VL modeli indiriliyor (Qwen2.5-VL-7B-Instruct-AWQ, ~6.5 GB, caption+rerank icin) -> $BUNDLE/models/vl"
    python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct-AWQ', local_dir='$BUNDLE/models/vl')
print('indirildi:', path)
"
fi

# --- 4. Ozet ------------------------------------------------------------------
say "Tamam"
echo "Paket boyutu: $(du -sh "$BUNDLE" | cut -f1)"
echo
echo "Simdi $BUNDLE klasorunu (USB/ag ile) hedef makineye tasiyin, sonra orada:"
echo "  git clone <bu repo> && cd VideoAnalysis"
echo "  ./scripts/setup.sh --offline /yol/offline_bundle"
