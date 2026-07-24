"""Manuel uçtan uca ingest çalıştırıcı (Temporal orkestrasyonu olmadan) - ingest
aktivitelerinin mantığını Temporal workflow'una bağlamadan önce doğrulamak için
(proje-ozeti.md §11). Aktiviteler @activity.defn ile işaretli ama düz Python
fonksiyonu olarak doğrudan çağrılabiliyor.

YOLO/görsel alanlar bilinçli olarak atlanıyor (kullanıcı isteği - şu an sadece
embedding modelinin başarısına bakılıyor, telemetri zaten yok). vehicle_count
tüm pencerelerde 0 yazılıyor.

Her çalıştırma sonunda üç başarı kıstası (proje-ozeti.md §7, §11 Adım 0) rapor
edilir: embedding hızı (gerçek-zaman katı), depolama (klip başına byte) ve
başarı oranı `scripts/eval_retrieval.py` ile ayrıca ölçülür (tüm korpus
ingest edildikten sonra anlamlı, tek video ile değil).

Kullanım: python scripts/ingest_video.py <video_id>
"""
import asyncio
import sys
import time

import psycopg

from common import config
from ingest.activities.clip_embedding import embed_clips, unload_model
from ingest.activities.proxy_generation import generate_proxy
from ingest.activities.selective_caption import generate_captions
from ingest.activities.telemetry_processing import process_telemetry
from scripts.write_clips import write_clips


def _get_source_path(video_id: str) -> str:
    with psycopg.connect(config.postgres_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT source_path FROM videos WHERE video_id = %s", (video_id,))
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"videos tablosunda bulunamadı: {video_id}")
    return row[0]


def _set_status(video_id: str, status: str) -> None:
    with psycopg.connect(config.postgres_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos SET status = %s, updated_at = now() WHERE video_id = %s",
                (status, video_id),
            )
        conn.commit()


async def ingest(video_id: str) -> None:
    source_path = _get_source_path(video_id)

    print(f"[{video_id}] proxy üretiliyor...")
    proxy_path = await generate_proxy(video_id, source_path)
    _set_status(video_id, "proxy_done")

    print(f"[{video_id}] telemetri/pencereleme...")
    windows = await process_telemetry(video_id, proxy_path)
    _set_status(video_id, "telemetry_done")
    print(f"[{video_id}] {len(windows)} pencere üretildi")

    print(f"[{video_id}] embedding...")
    t0 = time.perf_counter()
    embeddings = await embed_clips(video_id, proxy_path, windows)
    embed_elapsed_s = time.perf_counter() - t0
    _set_status(video_id, "embedding_done")

    window_duration_s = sum(w.t_end - w.t_start for w in windows)
    realtime_multiplier = window_duration_s / embed_elapsed_s if embed_elapsed_s > 0 else float("inf")
    embedding_bytes = len(embeddings[0]) * 4 if embeddings else 0  # float32
    total_embedding_bytes = embedding_bytes * len(embeddings)

    # X-CLIP'i GPU'dan boşalt: GT1030 4GB'ta embedding modeli yüklüyken Ollama'ya
    # (moondream) caption isteği atmak sessizce boş yanıt üretiyor (bellek baskısı,
    # hata fırlatmıyor - bkz. ingest/activities/clip_embedding.py unload_model).
    unload_model()

    print(f"[{video_id}] caption üretimi...")
    captions = await generate_captions(video_id, proxy_path, windows)
    _set_status(video_id, "caption_done")

    visual_fields = [{"vehicle_count": 0} for _ in windows]
    write_clips(video_id, windows, embeddings, visual_fields, captions)
    _set_status(video_id, "written")
    print(f"[{video_id}] ClickHouse'a yazıldı.")

    print(f"\n--- [{video_id}] başarı kıstasları ---")
    print(f"Hız: {len(windows)} pencere, {embed_elapsed_s:.2f}s embedding süresi "
          f"({window_duration_s:.1f}s video-süresi) -> {realtime_multiplier:.1f}x gerçek-zaman")
    print(f"Depolama: pencere başına {embedding_bytes} byte (512d fp32), "
          f"bu video için toplam {total_embedding_bytes / 1024:.1f} KB")
    print("Başarı oranı: scripts/eval_retrieval.py ile korpus genelinde ölçülüyor "
          "(tek video için istatistiksel olarak anlamsız)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Kullanım: python scripts/ingest_video.py <video_id>")
        sys.exit(1)
    asyncio.run(ingest(sys.argv[1]))
