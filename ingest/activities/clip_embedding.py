"""Klip embedding: pencere başına tek vektör (proje-ozeti.md §3.1 madde 3).

MODEL: Qwen/Qwen3-VL-Embedding-2B (Apache-2.0).

NEDEN BU MODEL: Üç aday gerçek SeaDronesSee verisinde karşılaştırıldı
(VideoCLIP-XL, EBind, Qwen3-VL-Embedding-2B). Belirleyici kriter lisans oldu:
VideoCLIP-XL ve EBind CC-BY-NC-SA 4.0 (NonCommercial) - ticari/savunma
kullanımına kapalı. Qwen3-VL-Embedding-2B ticari kullanıma açık tek adaydı.
Retrieval kalitesi açısından "en iyi" olduğu İDDİA EDİLMİYOR - proje-ozeti.md
§5'in gerektirdiği golden set karşılaştırması (InternVideo2/VideoPrism/
LanguageBind dahil) hâlâ yapılmadı. Detay: docs/worklog_2026-07-28.md.

DTYPE: Modelin config'i bfloat16 belirtiyor ama bf16 Tensor Core desteği
compute capability >= 8.0 (Ampere) gerektiriyor. Turing (T4, compute 7.5) ve
öncesinde bf16 emülasyona düşüp ~10x yavaşlıyor - ölçüldü: T4'te 126s -> 12s
(fp16'ya zorlayınca). EMBEDDING_DTYPE=auto bunu donanımdan tespit eder.

MRL: Model Matryoshka temsili destekliyor (64-2048 arası kısaltma). EMBEDDING_DIM
< 2048 ise vektör ilk N boyuta kısaltılıp yeniden normalize edilir. Kısaltmanın
Recall'e etkisi ÖLÇÜLMEDİ - golden set gerekiyor (§7).
"""
import functools
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from temporalio import activity

from common import config
from common.minio_client import download_temp
from ingest.activities.types import TelemetryWindow

_embedder = None


def _resolve_model_dir() -> str:
    """Yerel kopya verilmişse onu, yoksa HF Hub'dan indirileni kullanır.
    Model kendi `scripts/qwen3_vl_embedding.py` dosyasını bundle ettiği için
    tam snapshot gerekiyor (sadece ağırlıklar yetmez)."""
    if config.EMBEDDING_MODEL_DIR:
        return config.EMBEDDING_MODEL_DIR
    from huggingface_hub import snapshot_download
    return snapshot_download(config.EMBEDDING_MODEL)


def _resolve_dtype() -> torch.dtype:
    if config.EMBEDDING_DTYPE == "float16":
        return torch.float16
    if config.EMBEDDING_DTYPE == "bfloat16":
        return torch.bfloat16
    if config.EMBEDDING_DTYPE == "float32":
        return torch.float32

    # auto:
    if not torch.cuda.is_available():
        # CPU'da float16 KULLANILMAZ - PyTorch CPU backend'inde bircok op
        # fp16 icin implemente degil ("not implemented for 'Half'") ve
        # implemente olanlar da fp32'ye gore yavas. bfloat16 CPU'da destekli
        # ve modelin dogal dtype'i (config.json: bfloat16), ayrica fp32'nin
        # yarisi kadar RAM kullaniyor - 2B parametrede 4GB vs 8GB.
        return torch.bfloat16
    # bf16 Tensor Core sadece Ampere+ (compute capability >= 8.0) uzerinde;
    # Turing'de (T4, cc 7.5) emulasyona dusup ~10x yavasliyor - olculdu.
    if torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _get_embedder():
    global _embedder
    if _embedder is None:
        # Model soguk onbellekte ilk kez indirilirken (snapshot_download,
        # dakikalar surebilir - gercek calistirmada 28dk olculdu, yavas agda)
        # heartbeat atilmazsa Temporal bunu tikanma sanip aktiviteyi iptal
        # eder ve tum retry'lar AYNI sekilde basarisiz olur (indirme her
        # denemede tekrar heartbeat'siz kalir). Once heartbeat atip
        # sayacini sifirliyoruz.
        _heartbeat("embedding modeli yukleniyor/indiriliyor")
        model_dir = _resolve_model_dir()
        scripts_dir = str(Path(model_dir) / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from qwen3_vl_embedding import Qwen3VLEmbedder

        dtype = _resolve_dtype()
        _log(f"embedding modeli yukleniyor: {model_dir} (dtype={dtype})")

        base = {"num_frames": config.EMBEDDING_NUM_FRAMES,
                "max_frames": config.EMBEDDING_NUM_FRAMES}

        # SISTEM RAM'i (VRAM degil) - yukleme sirasinda cokme sebebi buydu.
        # from_pretrained varsayilan olarak modeli once CPU'da tam olarak
        # olusturup sonra .to(device) yapiyor; state dict + model ayni anda
        # RAM'de duruyor ve tepe kullanim modelin ~2 katina cikiyor. 4 GB'lik
        # bu modelde ~8 GB demek - Colab'in ~12.7 GB RAM'inde diger kutuphanelerle
        # birlikte cokuyor ("Loading weights: %47"'de oldugu gozlendi).
        # low_cpu_mem_usage=True agirliklari parca parca yukluyor, tepe
        # kullanimi model boyutuna indiriyor.
        attempts = [
            {**base, "dtype": dtype, "low_cpu_mem_usage": True},
            {**base, "dtype": dtype},          # eski transformers/imza
            base,                              # dtype kabul etmiyorsa
        ]
        last_error: Exception | None = None
        for i, kwargs in enumerate(attempts):
            try:
                _embedder = Qwen3VLEmbedder(model_dir, **kwargs)
                if i:
                    _log(f"  (geri cekilme: {sorted(set(kwargs) - set(base)) or 'varsayilan'})")
                break
            except TypeError as exc:
                last_error = exc          # imza bu kwarg'i kabul etmiyor, sonrakini dene
        else:
            raise RuntimeError(
                f"Embedding modeli yuklenemedi: {last_error}"
            ) from last_error
    return _embedder


def unload_model() -> None:
    """Modeli GPU'dan boşaltır. Aynı GPU'da ardından başka bir model
    (YOLO/VLM) çalışacaksa ve VRAM darsa gerekli."""
    global _embedder
    _embedder = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _apply_mrl(vector: np.ndarray) -> list[float]:
    """MRL kısaltması + yeniden normalizasyon. Kosinüs mesafesi kullandığımız
    için kısaltma sonrası normalize etmek şart."""
    if config.EMBEDDING_DIM < vector.shape[-1]:
        vector = vector[: config.EMBEDDING_DIM]
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32).tolist()


def sample_frames(video_path: str, t_start: float, t_end: float,
                   num_frames: int) -> list[Image.Image]:
    """Pencereden eşit aralıklı kare örnekler.

    Tüm videoyu belleğe okumaz - sadece gereken karelere seek eder. 3-5 saatlik
    videolarda bu fark kritik (tam okuma onlarca GB olurdu)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    timestamps = np.linspace(t_start, max(t_end - 1e-3, t_start), num_frames)

    frames: list[Image.Image] = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        elif frames:
            frames.append(frames[-1])  # son gecerli kareyi tekrarla
    cap.release()

    if not frames:
        raise ValueError(f"{video_path} [{t_start:.1f}-{t_end:.1f}] hicbir kare okunamadi")
    return frames


@torch.inference_mode()
def embed_windows(video_path: str, windows: list[tuple[float, float]]) -> list[list[float]]:
    """Pencereleri toplu (batch) embed eder.

    Batch'leme burada kritik: tek tek çağrıda GPU'nun büyük kısmı boş kalıyor.
    proje-ozeti.md §8'deki GPU bütçesi varsayımı (~40x gerçek-zaman) hâlâ
    doğrulanmadı ve ölçtüğümüz tek-akış hızı (T4'te ~0.7x) bunun çok altında -
    gerçek throughput için batch boyutu donanıma göre ayarlanmalı."""
    embedder = _get_embedder()
    vectors: list[list[float]] = []

    for start in range(0, len(windows), config.EMBEDDING_BATCH_SIZE):
        batch = windows[start:start + config.EMBEDDING_BATCH_SIZE]
        inputs = [
            {"video": sample_frames(video_path, t0, t1, config.EMBEDDING_NUM_FRAMES)}
            for t0, t1 in batch
        ]
        outputs = embedder.process(inputs)
        vectors.extend(_apply_mrl(np.asarray(o.float().cpu())) for o in outputs)
        _heartbeat(f"{min(start + len(batch), len(windows))}/{len(windows)} pencere")

    return vectors


@torch.inference_mode()
def embed_text(text: str) -> list[float]:
    """Sorgu tarafındaki semantic_text'i aynı ortak uzaya taşır."""
    embedder = _get_embedder()
    output = embedder.process([{"text": text}])[0]
    return _apply_mrl(np.asarray(output.float().cpu()))


@functools.lru_cache(maxsize=1024)
def embed_text_cached(text: str) -> tuple[float, ...]:
    """Tekrarlanan sorgular için embedding önbelleği (CLI/servis kullanımı)."""
    return tuple(embed_text(text))


def _heartbeat(message: str) -> None:
    try:
        activity.heartbeat(message)
    except RuntimeError:
        pass


def _log(message: str) -> None:
    try:
        activity.logger.info(message)
    except RuntimeError:
        print(message)


@activity.defn
async def embed_clips(video_id: str, proxy_path: str,
                       windows: list[TelemetryWindow]) -> list[list[float]]:
    """Her pencere için proxy videodan tek embedding vektörü üretir."""
    if not windows:
        return []
    with download_temp(config.MINIO_BUCKET_PROXY, proxy_path) as local_path:
        return embed_windows(local_path, [(w.t_start, w.t_end) for w in windows])
