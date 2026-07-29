"""Yazım: tüm aktivite çıktılarını tek bir Qdrant noktasında birleştirir
(proje-ozeti.md §3.1 madde 6 - orada ClickHouse `clips` satırıydı, Qdrant
kararı sonrası payload'lı vektör noktası oldu).

İDEMPOTENT: Nokta kimliği (video_id, t_start)'tan deterministik türetildiği
için aynı video yeniden ingest edilirse satır çoğalmaz, üzerine yazılır -
backfill ve retry güvenli.

KISMI ÇIKTI TOLERANSI: visual_fields/captions boş gelebilir (bileşen kapalı
ya da başarısız). Bu durumda ilgili alanlar varsayılana düşer, ingest
düşmez - embedding + pencere zamanları asgari gereksinimdir.
"""
from temporalio import activity

from common import config
from common.qdrant_store import ClipPayload, ensure_collection, get_client, upsert_clips
from ingest.activities.types import TelemetryWindow, VisualFields


def build_payloads(video_id: str, windows: list[TelemetryWindow],
                    visual: list[VisualFields] | None,
                    captions: dict[str, str] | None) -> list[ClipPayload]:
    visual = visual or []
    captions = captions or {}

    payloads = []
    for i, w in enumerate(windows):
        vehicle_count = visual[i].vehicle_count if i < len(visual) else 0
        payloads.append(ClipPayload(
            video_id=video_id,
            t_start=w.t_start,
            t_end=w.t_end,
            sensor_type=w.sensor_type,
            avg_speed_kmh=w.avg_speed_kmh,
            agl_m=w.agl_m,
            sun_elevation=w.sun_elevation,
            over_sea=w.over_sea,
            vehicle_count=vehicle_count,
            caption=captions.get(w.key, ""),
            lat=w.lat,
            lon=w.lon,
            heading_deg=w.heading_deg,
            gimbal_pitch_deg=w.gimbal_pitch_deg,
        ))
    return payloads


def _log(message: str) -> None:
    try:
        activity.logger.info(message)
    except RuntimeError:
        print(message)


@activity.defn
async def write_clips(video_id: str, windows: list[TelemetryWindow],
                       embeddings: list[list[float]],
                       visual: list[VisualFields] | None = None,
                       captions: dict[str, str] | None = None) -> int:
    """Pencereleri Qdrant'a yazar, yazılan nokta sayısını döner."""
    if not windows:
        return 0
    if len(embeddings) != len(windows):
        raise ValueError(
            f"{video_id}: embedding sayisi ({len(embeddings)}) pencere sayisiyla "
            f"({len(windows)}) eslesmiyor - kismi embedding yazilmamali"
        )

    client = get_client()
    collection = ensure_collection(client)
    payloads = build_payloads(video_id, windows, visual, captions)

    written = upsert_clips(client, collection, embeddings, payloads)
    _log(f"{video_id}: {written} nokta yazildi -> {collection}")
    return written
