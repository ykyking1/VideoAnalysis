#!/usr/bin/env bash
# İNTERNET OLAN bir Linux makinede çalıştırılır - hedef (internetsiz) makineye
# taşınacak her şeyi (Python paketleri, Docker imajları, model ağırlıkları)
# tek bir klasöre indirir.
#
#     ./scripts/prepare_offline_bundle.sh [--with-vlm]
#
# --with-vlm: caption/rerank için Qwen2.5-VL-7B-Instruct-AWQ'yu da indirir
#             (~6.5 GB ek, sadece caption/rerank kullanacaksanız gerekli).
#
# ÖNEMLİ: bu makine hedef makineyle AYNI mimaride olmalı (x86_64 Linux, aynı
# Python sürümü) - pip wheel'leri platforma özgü. Farklıysa `pip download`
# icin --platform/--python-version bayraklarini elle ekleyin.
#
# Çıktı: ./offline_bundle/ (bunu USB/ağ ile hedef makineye taşıyın, sonra
# orada: ./scripts/setup.sh --offline ./offline_bundle)
set -euo pipefail
cd "$(dirname "$0")/.."

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[31m[HATA] %s\033[0m\n' "$*" >&2; exit 1; }

WITH_VLM=0
[ "${1:-}" = "--with-vlm" ] && WITH_VLM=1

BUNDLE="$(pwd)/offline_bundle"
mkdir -p "$BUNDLE"/{wheels,docker_images,models/embedding,models/yolo,models/parse}
[ "$WITH_VLM" = 1 ] && mkdir -p "$BUNDLE/models/vl"

command -v docker >/dev/null 2>&1 || die "docker gerekiyor (imajlari indirip kaydetmek icin)."
command -v python3 >/dev/null 2>&1 || die "python3 gerekiyor."

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
