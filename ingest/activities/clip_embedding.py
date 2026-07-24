"""Klip embedding: video-metin modeli, pencere başına tek vektör
(proje-ozeti.md §3.1 madde 3).

Model seçimi kesinleşmedi (bkz. §5): xuguohai/X-CLIP (retrieval varyantı) lider
aday ama pip/HF Hub üzerinden doğrudan yüklenebilir bir paket değil, kendi repo
+ checkpoint'i gerektiriyor. Bu yerel testte, ayrıştırma/depolama akışını uçtan
uca doğrulamak için `microsoft/xclip-base-patch32` (HF transformers'ta yerleşik,
512d - §5/§6'daki 512d varsayımıyla örtüşüyor) kullanılıyor. Bu model Kinetics
sınıflandırma için fine-tune edilmiş (proje-ozeti.md §5'in "karıştırılmamalı"
dediği model) - retrieval kalitesi golden set'te ölçülmeden production kararı
olarak alınmamalı, sadece mekanik doğrulama içindir.
"""
import cv2
import numpy as np
import torch
from temporalio import activity
from transformers import AutoProcessor, XCLIPModel

from common import config
from common.minio_client import download_temp
from ingest.activities.types import TelemetryWindow

MODEL_NAME = "microsoft/xclip-base-patch32"
NUM_FRAMES = 8

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model: XCLIPModel | None = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        _model = XCLIPModel.from_pretrained(MODEL_NAME).to(_device).eval()
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def unload_model() -> None:
    """X-CLIP'i GPU'dan boşaltır. Aynı süreçte hemen ardından Ollama'ya (caption/
    query parsing) GPU çağrısı yapılacaksa gerekli - GT1030 4GB'ta ikisi aynı anda
    yüklüyken Ollama'nın sessizce boş yanıt döndürdüğü gözlemlendi (bellek baskısı,
    hata fırlatmadan). Bir sonraki embed çağrısında model otomatik yeniden yüklenir."""
    global _model, _processor
    _model = None
    _processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sample_frames(video_path: str, t_start: float, t_end: float, num_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    timestamps = np.linspace(t_start, max(t_end - 1e-3, t_start), num_frames)
    frames = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            frame = frames[-1] if frames else np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


@torch.inference_mode()
def embed_window(video_path: str, t_start: float, t_end: float) -> list[float]:
    model, processor = _get_model()
    frames = _sample_frames(video_path, t_start, t_end, NUM_FRAMES)
    inputs = processor(videos=list(frames), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_device)
    # get_video_features bu transformers sürümünde ham tensör değil, ModelOutput
    # (last_hidden_state + pooler_output) döndürüyor - joint (512d) uzaydaki
    # pooled video vektörü pooler_output'ta, last_hidden_state kare/patch düzeyi
    # ara temsil (ölçüldü: last_hidden_state şekli (NUM_FRAMES, 50, 768)).
    video_embeds = model.get_video_features(pixel_values=pixel_values).pooler_output[0]
    return video_embeds.cpu().float().tolist()


@torch.inference_mode()
def embed_text(text: str) -> list[float]:
    """Sorgu tarafındaki semantic_text'i aynı X-CLIP metin kodlayıcısıyla
    embedding'e çevirir (query/hybrid_search.py tarafından kullanılır)."""
    model, processor = _get_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    text_embeds = model.get_text_features(**inputs).pooler_output[0]
    return text_embeds.cpu().float().tolist()


@activity.defn
async def embed_clips(
    video_id: str, proxy_path: str, windows: list[TelemetryWindow]
) -> list[list[float]]:
    """Her telemetri penceresi için proxy videodan tek embedding vektörü üretir."""
    with download_temp(config.MINIO_BUCKET_PROXY, proxy_path) as local_path:
        return [embed_window(local_path, w.t_start, w.t_end) for w in windows]
