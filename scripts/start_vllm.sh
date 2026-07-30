#!/usr/bin/env bash
# vLLM sunucusunu VRAM'e gore otomatik yapilandirip baslatir.
#
#     ./scripts/start_vllm.sh                 # otomatik model secimi
#     ./scripts/start_vllm.sh --rerank        # ayristirma + rerank icin VL modeli
#     PARSE_MODEL=... ./scripts/start_vllm.sh # elle model secimi
#
# NEDEN OTOMATIK: sorgu aninda embedding modeli (3.96 GB) DE yuklu olur.
# vLLM varsayilan olarak GPU'nun %90'ini kendine ayirir ve embedding modeline
# yer birakmaz - en sik yapilan hata bu. Asagida hem model hem
# --gpu-memory-utilization VRAM'den hesaplaniyor.
set -euo pipefail
cd "$(dirname "$0")/.."

EMBED_RESERVE_MB=5000     # embedding modeli (3.96 GB) + pay

RERANK=0
[ "${1:-}" = "--rerank" ] && RERANK=1

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "[HATA] nvidia-smi yok - GPU gorunmuyor." >&2; exit 1; }
TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

# T4 (compute 7.5) ve oncesi bfloat16 Tensor Core desteklemiyor - vLLM'e
# acikca fp16 soyluyoruz, otomatik tespite guvenmiyoruz (embedding modelinde
# ayni sorunu yasadik, bkz. ingest/activities/clip_embedding.py).
if awk "BEGIN{exit !($CC < 8.0)}"; then
    DTYPE_FLAG=(--dtype half)
    echo "compute capability $CC < 8.0 - --dtype half (fp16) kullanilacak"
else
    DTYPE_FLAG=()
fi

if ! command -v vllm >/dev/null 2>&1; then
    echo "[HATA] vllm komutu bulunamadi. Kurulum:" >&2
    echo "  pip install uv && uv pip install -r requirements-serving.txt --torch-backend=auto" >&2
    exit 1
fi

# --- Model secimi ----------------------------------------------------------
# Agirliklar (HuggingFace'ten dogrulandi):
#   Qwen2.5-3B-Instruct-AWQ     2.50 GB
#   Qwen2.5-7B-Instruct-AWQ     5.19 GB
#   Qwen2.5-VL-7B-Instruct-AWQ  6.45 GB  (ayristirma + rerank, tek model)
if [ -n "${PARSE_MODEL:-}" ]; then
    MODEL="$PARSE_MODEL"
    REASON="elle secildi (PARSE_MODEL)"
elif [ "$RERANK" = 1 ]; then
    if [ "$TOTAL_MB" -lt 14000 ]; then
        echo "[HATA] Rerank icin en az ~16 GB VRAM gerekiyor (bulunan: ${TOTAL_MB} MB)." >&2
        echo "       Embedding 3.96 GB + VL-7B 6.45 GB + KV cache sigmiyor." >&2
        exit 1
    fi
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    REASON="rerank istendi - tek model hem ayristirma hem rerank yapar"
elif [ "$TOTAL_MB" -lt 10000 ]; then
    MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
    REASON="${TOTAL_MB} MB VRAM - 7B embedding modeliyle birlikte sigmaz"
else
    MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
    REASON="${TOTAL_MB} MB VRAM - 7B rahat siginiyor"
fi

# --- GPU pay hesabi --------------------------------------------------------
FRAC=$(python3 -c "
total = $TOTAL_MB
frac = (total - $EMBED_RESERVE_MB) / total
print(f'{min(max(frac, 0.25), 0.75):.2f}')
")

cat <<EOF
GPU toplam        : ${TOTAL_MB} MB
Model             : ${MODEL}
  gerekce         : ${REASON}
gpu-memory-util   : ${FRAC}  (embedding modeline ~${EMBED_RESERVE_MB} MB birakiliyor)

Ilk calistirmada model indirilir (2.5-6.5 GB), birkac dakika surer.
"Application startup complete" gorunce hazirdir.
EOF

# Sorgu tarafinin ayni modeli kullanmasi icin .env'e yaz
python3 - <<PY
import pathlib, re
p = pathlib.Path('.env')
lines = p.read_text(encoding='utf-8').splitlines() if p.exists() else []
def setkv(lines, k, v):
    out, done = [], False
    for l in lines:
        if re.match(rf'^{k}=', l):
            out.append(f'{k}={v}'); done = True
        else:
            out.append(l)
    if not done: out.append(f'{k}={v}')
    return out
lines = setkv(lines, 'PARSE_MODEL', '$MODEL')
if $RERANK:
    lines = setkv(lines, 'VLM_MODEL', '$MODEL')
p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('.env guncellendi: PARSE_MODEL=$MODEL')
PY

exec vllm serve "$MODEL" \
    --guided-decoding-backend xgrammar \
    --gpu-memory-utilization "$FRAC" \
    --max-model-len 8192 \
    "${DTYPE_FLAG[@]}"
