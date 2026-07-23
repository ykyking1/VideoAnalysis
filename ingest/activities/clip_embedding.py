"""Klip embedding: video-metin modeli, pencere başına tek vektör
(proje-ozeti.md §3.1 madde 3). Model seçimi kesinleşmedi (bkz. §5) - X-CLIP
(xuguohai/X-CLIP) lider aday, VideoCLIP-XL yeni aday. NVIDIA Triton+TensorRT
üzerinden servislenmesi planlanıyor (bkz. §10).
"""
from temporalio import activity

from ingest.activities.telemetry_processing import TelemetryWindow


@activity.defn
async def embed_clips(
    video_id: str, proxy_path: str, windows: list[TelemetryWindow]
) -> list[list[float]]:
    """Her telemetri penceresi için proxy videodan tek embedding vektörü üretir."""
    raise NotImplementedError
