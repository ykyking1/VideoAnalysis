"""Model proxy üretimi: ffmpeg + NVDEC, 240-360p HEVC (proje-ozeti.md §3.1 madde 1).

Önizleme için değil, model tüketimi (embedding/detektör/rerank) ve backfill'lerin
ucuz decode'u için. Kalıcı tutulması opsiyonel (bkz. §4, JIT alternatifi).
"""
from temporalio import activity


@activity.defn
async def generate_proxy(video_id: str, source_path: str) -> str:
    """MinIO'daki ham videodan model-kalite proxy üretir, proxy'nin MinIO yolunu döner."""
    raise NotImplementedError
