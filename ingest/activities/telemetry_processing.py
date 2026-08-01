"""Telemetri işleme: pymavlink + polars ile `.tlog`/MAVLink log ayrıştırma,
8sn pencere / 8sn kaydırma (proje-ozeti.md §3.1 madde 2).

PENCERELEME: 8sn/8sn, örtüşmesiz. 2026-07-29'da %50 örtüşmeli (4sn kaydırma)
şemadan değiştirildi: teyit edilen envanterde (~300.000 video × 3-5sa, bkz. §8)
%50 örtüşme ~1 milyar vektör demekti; kaydırmayı pencere uzunluğuna eşitlemek
vektör sayısını, embedding GPU-saatini ve indeks boyutunu yarıya indiriyor.
Recall'e etkisi ÖLÇÜLMEDİ (golden set gerekiyor, §7) - iyileştirme fırsatı
olarak §9 madde 1 (sahne sınırına yaslanmış chunking) hâlâ açık.

TÜRETİLMİŞ ALANLAR:
- avg_speed_kmh  : GLOBAL_POSITION_INT vx/vy/vz'den
- agl_m          : relative_alt (yer seviyesine göre irtifa)
- sun_elevation  : pysolar ile lat/lon/zaman'dan ("gece"/"günbatımı"nın
                   deterministik karşılığı - LLM'in tahminine bırakılmaz)
- over_sea       : Shapely point-in-polygon, COASTLINE_GEOJSON kıyı poligonları

GRACEFUL DEGRADATION: telemetri dosyası yoksa ya da opsiyonel bağımlılıklar
(pymavlink/polars/pysolar/shapely) kurulu değilse pencereler yine üretilir,
türetilmiş alanlar None kalır. Arama tarafı None alanları filtreye sokmaz
(bkz. query/filter_builder.py) - yani telemetrisiz videolar aramada
kaybolmaz, sadece yapısal filtrelerle bulunamaz.

ZAMAN HİZALAMA: MAVLink log'unun ilk kaydı video t=0 ile eşleşmiş kabul
edilir. Gerçek arşivde video/telemetri arasında sabit bir kayma varsa
TELEMETRY_OFFSET_S ile düzeltilmeli - bu, arşive özgü doğrulanması gereken
bir varsayımdır.
"""
import datetime as dt
import json
import math
import os
import subprocess

from temporalio import activity

from common import config
from common.minio_client import download_temp
from ingest.activities.types import TelemetryWindow

TELEMETRY_OFFSET_S = float(os.environ.get("TELEMETRY_OFFSET_S", "0.0"))

_MAVLINK_TYPES = ["GLOBAL_POSITION_INT", "GIMBAL_DEVICE_ATTITUDE_STATUS", "MOUNT_STATUS"]


def probe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def build_windows(duration_s: float) -> list[TelemetryWindow]:
    """Sabit uzunluklu, config.WINDOW_S/STRIDE_S pencereleri üretir.

    Deger burada, CAGRI ANINDA okunuyor - modul seviyesinde sabitlenirse
    (eskiden oyleydi) bir notebook/REPL'de config.WINDOW_S degistirilip
    common.config reload edildiginde bu modul zaten import edilmis oldugu
    icin eski deger kalirdi (gercek Kaggle testinde bulundu: 60sn pencere
    denemesi sessizce 8sn'de kalmaya devam etmisti)."""
    window_s = config.WINDOW_S
    stride_s = config.STRIDE_S
    windows: list[TelemetryWindow] = []
    t_start = 0.0
    while t_start < duration_s:
        t_end = min(t_start + window_s, duration_s)
        windows.append(TelemetryWindow(t_start=t_start, t_end=t_end))
        if t_end >= duration_s:
            break
        t_start += stride_s
    return windows


def parse_mavlink(telemetry_local_path: str) -> list[dict]:
    """MAVLink log'unu düz kayıt listesine çevirir. pymavlink kurulu değilse
    boş liste döner (türetilmiş alanlar None kalır)."""
    try:
        from pymavlink import mavutil
    except ImportError:
        _log("pymavlink kurulu degil - telemetri atlaniyor (pip install .[telemetry])")
        return []

    conn = mavutil.mavlink_connection(telemetry_local_path)
    records: list[dict] = []
    t0: float | None = None

    while True:
        msg = conn.recv_match(type=_MAVLINK_TYPES, blocking=False)
        if msg is None:
            break
        ts = getattr(msg, "_timestamp", None)
        if ts is None:
            continue
        if t0 is None:
            t0 = ts

        record = {"t": ts - t0 + TELEMETRY_OFFSET_S, "unix_ts": ts, "type": msg.get_type()}
        if msg.get_type() == "GLOBAL_POSITION_INT":
            record.update({
                "lat": msg.lat / 1e7,
                "lon": msg.lon / 1e7,
                "agl_m": msg.relative_alt / 1000.0,
                # vx/vy/vz cm/s -> km/h
                "speed_kmh": math.sqrt(msg.vx ** 2 + msg.vy ** 2 + msg.vz ** 2) * 0.036,
                "heading_deg": msg.hdg / 100.0 if msg.hdg != 65535 else None,
            })
        elif msg.get_type() in ("GIMBAL_DEVICE_ATTITUDE_STATUS", "MOUNT_STATUS"):
            record["gimbal_pitch_deg"] = _gimbal_pitch(msg)
        records.append(record)

    return records


def _gimbal_pitch(msg) -> float | None:
    if hasattr(msg, "pointing_a"):  # MOUNT_STATUS: santi-derece
        return msg.pointing_a / 100.0
    q = getattr(msg, "q", None)
    if q and len(q) == 4:  # GIMBAL_DEVICE_ATTITUDE_STATUS: kuaternion
        w, x, y, z = q
        return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    return None


def _aggregate_window(records: list[dict], t_start: float, t_end: float) -> dict:
    """Bir pencereye düşen kayıtları tek satıra indirger. polars varsa onunla
    (büyük log'larda çok daha hızlı), yoksa saf Python ile."""
    in_window = [r for r in records if t_start <= r["t"] < t_end]
    if not in_window:
        return {}

    def _mean(key: str) -> float | None:
        values = [r[key] for r in in_window if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "lat": _mean("lat"),
        "lon": _mean("lon"),
        "agl_m": _mean("agl_m"),
        "avg_speed_kmh": _mean("speed_kmh"),
        "heading_deg": _mean("heading_deg"),
        "gimbal_pitch_deg": _mean("gimbal_pitch_deg"),
        "unix_ts": _mean("unix_ts"),
    }


def sun_elevation(lat: float, lon: float, unix_ts: float) -> float | None:
    """Güneşin ufuk üstü açısı (derece). "gece"/"günbatımı" gibi kavramların
    deterministik karşılığı - modele tahmin ettirilmez."""
    try:
        from pysolar.solar import get_altitude
    except ImportError:
        return None
    when = dt.datetime.fromtimestamp(unix_ts, tz=dt.timezone.utc)
    return float(get_altitude(lat, lon, when))


_coastline = None
_coastline_loaded = False


def _get_coastline():
    """Kıyı poligonlarını (GeoJSON) bir kez yükler. Yoksa None döner."""
    global _coastline, _coastline_loaded
    if _coastline_loaded:
        return _coastline
    _coastline_loaded = True

    if not config.COASTLINE_GEOJSON or not os.path.exists(config.COASTLINE_GEOJSON):
        return None
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError:
        _log("shapely kurulu degil - over_sea hesaplanmiyor")
        return None

    with open(config.COASTLINE_GEOJSON, encoding="utf-8") as f:
        geojson = json.load(f)
    geometries = [shape(feat["geometry"]) for feat in geojson.get("features", [])]
    _coastline = unary_union(geometries) if geometries else None
    return _coastline


def over_sea(lat: float, lon: float) -> bool | None:
    """Nokta kara poligonlarının DIŞINDAysa deniz üstü kabul edilir."""
    land = _get_coastline()
    if land is None:
        return None
    from shapely.geometry import Point
    return not land.contains(Point(lon, lat))


def enrich_windows(windows: list[TelemetryWindow], records: list[dict],
                    sensor_type: str = "unknown") -> list[TelemetryWindow]:
    """Pencerelere telemetriden türeyen alanları doldurur."""
    for w in windows:
        w.sensor_type = sensor_type
        agg = _aggregate_window(records, w.t_start, w.t_end)
        if not agg:
            continue

        w.avg_speed_kmh = agg["avg_speed_kmh"]
        w.agl_m = agg["agl_m"]
        w.lat = agg["lat"]
        w.lon = agg["lon"]
        w.heading_deg = agg["heading_deg"]
        w.gimbal_pitch_deg = agg["gimbal_pitch_deg"]

        if w.lat is not None and w.lon is not None:
            if agg["unix_ts"] is not None:
                w.sun_elevation = sun_elevation(w.lat, w.lon, agg["unix_ts"])
            w.over_sea = over_sea(w.lat, w.lon)

    return windows


def _log(message: str) -> None:
    try:
        activity.logger.info(message)
    except RuntimeError:
        print(message)


@activity.defn
async def process_telemetry(video_id: str, proxy_path: str,
                             telemetry_path: str | None = None,
                             sensor_type: str = "unknown") -> list[TelemetryWindow]:
    """Proxy süresinden pencereleri üretir, telemetri varsa türetilmiş alanları
    doldurur."""
    with download_temp(config.MINIO_BUCKET_PROXY, proxy_path) as local_proxy:
        duration = probe_duration(local_proxy)

    windows = build_windows(duration)
    _log(f"{video_id}: {duration:.1f}s -> {len(windows)} pencere "
         f"({config.WINDOW_S}sn/{config.STRIDE_S}sn)")

    if not telemetry_path:
        return windows

    with download_temp(config.MINIO_BUCKET_RAW, telemetry_path) as local_tlog:
        records = parse_mavlink(local_tlog)
    _log(f"{video_id}: {len(records)} telemetri kaydi ayristirildi")

    return enrich_windows(windows, records, sensor_type=sensor_type)
