"""Terfi etmiş görsel alanlar: YOLO26 (IR fine-tune), sorgu loglarından sık talep
gören ve deterministik çözülebilen görsel kavramları kolonlaştırır (örn.
vehicle_count). Katalog kullanım verisine göre organik büyür (proje-ozeti.md §3.1
madde 4).
"""
from temporalio import activity

from ingest.activities.telemetry_processing import TelemetryWindow


@activity.defn
async def extract_visual_fields(
    video_id: str, proxy_path: str, windows: list[TelemetryWindow]
) -> list[dict]:
    """Her pencere için YOLO26 çıktısından türetilmiş görsel alanları (vehicle_count
    vb.) hesaplar."""
    raise NotImplementedError
