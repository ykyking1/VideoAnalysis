"""Telemetri işleme: pymavlink + polars ile .tlog/MAVLink log ayrıştırma
(proje-ozeti.md §3.1 madde 2). 8sn pencere / 4sn kaydırma, sabit-uzunluk örtüşmeli
chunking (iyileştirme fırsatı için bkz. §9 madde 1).

Yerel test verisi (SeaDronesSee) uçuş telemetrisi içermiyor - bu, gerçek İHA
arşivinden farklı bir durum ve proje-ozeti.md §7'de zaten "kamuya açık veri
gerçek arşivdeki doğruluğu ölçmez" diye ayrı tutulmuştu. telemetry_path
verilmediğinde WINDOW_S/STRIDE_S pencereleri video süresinden (ffprobe) üretilir,
türetilmiş alanlar None kalır. Gerçek .tlog verilebildiğinde pymavlink+astral+
shapely yolunu ekleyin (bkz. TODO).
"""
import json
import subprocess

from temporalio import activity

from common import config
from common.minio_client import download_temp
from ingest.activities.types import TelemetryWindow

WINDOW_S = 8.0
STRIDE_S = 4.0


def probe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _fixed_windows(duration_s: float) -> list[TelemetryWindow]:
    windows = []
    t_start = 0.0
    while t_start < duration_s:
        t_end = min(t_start + WINDOW_S, duration_s)
        windows.append(TelemetryWindow(
            t_start=t_start, t_end=t_end,
            avg_speed_kmh=None, agl_m=None, sun_elevation=None, over_sea=None,
            sensor_type="rgb", lat=None, lon=None, heading_deg=None, gimbal_pitch_deg=None,
        ))
        if t_end >= duration_s:
            break
        t_start += STRIDE_S
    return windows


@activity.defn
async def process_telemetry(video_id: str, proxy_path: str, telemetry_path: str | None = None) -> list[TelemetryWindow]:
    """telemetry_path verilmemişse (SeaDronesSee gibi) video süresinden sabit
    pencereler üretir, türetilmiş alanlar None kalır. telemetry_path verilirse
    gerçek MAVLink ayrıştırma gerekir (henüz implemente edilmedi - bkz. §11)."""
    if telemetry_path is not None:
        raise NotImplementedError(
            "Gerçek .tlog ayrıştırma (pymavlink+astral+shapely) henüz implemente "
            "edilmedi - şu an sadece telemetrisiz fallback yolu destekleniyor."
        )
    with download_temp(config.MINIO_BUCKET_PROXY, proxy_path) as local_path:
        duration_s = probe_duration(local_path)
    return _fixed_windows(duration_s)
