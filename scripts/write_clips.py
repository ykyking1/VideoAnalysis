"""Bir videonun pencerelerini/embedding'lerini/görsel alanlarını/caption'larını
tek satır halinde ClickHouse `clips` tablosuna yazar (proje-ozeti.md §3.1
madde 6)."""
import clickhouse_connect

from common import config
from ingest.activities.types import TelemetryWindow


def write_clips(
    video_id: str,
    windows: list[TelemetryWindow],
    embeddings: list[list[float]],
    visual_fields: list[dict],
    captions: dict[str, str],
) -> None:
    client = clickhouse_connect.get_client(
        host=config.CLICKHOUSE_HOST, port=config.CLICKHOUSE_PORT,
        username=config.CLICKHOUSE_USER, password=config.CLICKHOUSE_PASSWORD,
        database=config.CLICKHOUSE_DB,
    )

    rows = []
    for w, embedding, vf in zip(windows, embeddings, visual_fields):
        caption = captions.get(f"{w.t_start}:{w.t_end}", "")
        rows.append([
            video_id, w.t_start, w.t_end, embedding, caption,
            w.sensor_type, w.avg_speed_kmh, w.agl_m, w.sun_elevation, w.over_sea,
            vf.get("vehicle_count", 0),
            w.lat, w.lon, w.heading_deg, w.gimbal_pitch_deg,
        ])

    client.insert(
        "clips",
        rows,
        column_names=[
            "video_id", "t_start", "t_end", "embedding", "caption",
            "sensor_type", "avg_speed_kmh", "agl_m", "sun_elevation", "over_sea",
            "vehicle_count", "lat", "lon", "heading_deg", "gimbal_pitch_deg",
        ],
    )
