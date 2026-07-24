"""Ingest aktiviteleri arasında paylaşılan veri tipleri."""
from dataclasses import dataclass


@dataclass
class TelemetryWindow:
    t_start: float
    t_end: float
    avg_speed_kmh: float | None
    agl_m: float | None
    sun_elevation: float | None
    over_sea: bool | None
    sensor_type: str
    lat: float | None
    lon: float | None
    heading_deg: float | None
    gimbal_pitch_deg: float | None
