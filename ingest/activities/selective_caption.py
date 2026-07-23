"""Seçici caption: Qwen2.5-VL (vLLM üzerinde), ffmpeg sahne değişim skoruna göre
seçilen ~%10'luk "olay penceresi"ne kısa açıklama üretir; hibrit (vektör+tam metin)
aramada kullanılır (proje-ozeti.md §3.1 madde 5).
"""
from temporalio import activity

from ingest.activities.telemetry_processing import TelemetryWindow

EVENT_WINDOW_FRACTION = 0.10


@activity.defn
async def generate_captions(
    video_id: str, proxy_path: str, windows: list[TelemetryWindow]
) -> dict[tuple[float, float], str]:
    """Sahne değişim skoruna göre en yüksek ~%10 pencereyi seçip Qwen2.5-VL ile
    caption üretir; caption üretilmeyen pencereler için boş string döner."""
    raise NotImplementedError
