"""Yerel bir video dosyasını MinIO raw bucket'ına yükler ve Postgres `videos`
tablosuna kaydeder (proje-ozeti.md §2 - ingest durum takibi).

Kullanım: python scripts/register_video.py <video_id> <yerel_dosya_yolu>
"""
import sys

import psycopg

from common import config
from common.minio_client import ensure_buckets, get_client


def register(video_id: str, local_path: str) -> None:
    ensure_buckets()
    client = get_client()
    source_path = f"{video_id}/raw{'.' + local_path.rsplit('.', 1)[-1] if '.' in local_path else ''}"
    client.fput_object(config.MINIO_BUCKET_RAW, source_path, local_path)

    with psycopg.connect(config.postgres_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO videos (video_id, source_path, status)
                VALUES (%s, %s, 'discovered')
                ON CONFLICT (video_id) DO UPDATE SET source_path = EXCLUDED.source_path
                """,
                (video_id, source_path),
            )
        conn.commit()

    print(f"Kaydedildi: {video_id} -> minio://{config.MINIO_BUCKET_RAW}/{source_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Kullanım: python scripts/register_video.py <video_id> <yerel_dosya_yolu>")
        sys.exit(1)
    register(sys.argv[1], sys.argv[2])
