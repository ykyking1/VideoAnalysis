"""Telemetri işleme: pymavlink + polars ile .tlog/MAVLink log ayrıştırma
(proje-ozeti.md §3.1 madde 2). 8sn pencere / 4sn kaydırma, sabit-uzunluk örtüşmeli
chunking (iyileştirme fırsatı için bkz. §9 madde 1).
"""
from dataclasses import dataclass

from temporalio import activity

WINDOW_S = 8.0
STRIDE_S = 4.0


@dataclass
class TelemetryWindow:
    t_start: float
    t_end: float
    avg_speed_kmh: float
    agl_m: float
    sun_elevation: float
    over_sea: bool
    sensor_type: str
    lat: float
    lon: float
    heading_deg: float
    gimbal_pitch_deg: float


@activity.defn
async def process_telemetry(video_id: str) -> list[TelemetryWindow]:
    """Video süresini WINDOW_S/STRIDE_S pencerelerine böler, her pencere için
    türetilmiş alanları (astral/pysolar sun_elevation, Shapely/GeoPandas over_sea)
    hesaplar."""
    raise NotImplementedError
