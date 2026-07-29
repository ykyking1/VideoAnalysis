"""Yerel bir video (ve varsa telemetri log'u) MinIO raw bucket'ına yükler,
Postgres durum tablosuna kaydeder.

Bu, Kafka/MinIO-notification zincirini atlayarak elle ingest tetiklemek için;
üretimde videolar bucket'a düştüğünde zincir kendiliğinden çalışır.

Kullanım:
    python -m scripts.register_video <video_id> <video_dosyasi> [--telemetry <tlog>]
    python -m scripts.register_video --dir <klasor>    # klasordeki tum videolar
"""
import argparse
import sys
from pathlib import Path

from common import config
from common.minio_client import backend_name, ensure_buckets, get_client

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".ts", ".avi", ".mpg", ".mpeg", ".m4v")


def _record_state(video_id: str, source_path: str) -> None:
    """Postgres'e durum kaydı. Postgres yoksa uyarı verip geçer - ingest
    Postgres'e bağlı değil (Temporal kendi durumunu kendi tutuyor)."""
    try:
        import psycopg
    except ImportError:
        return
    try:
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
    except Exception as exc:  # noqa: BLE001
        print(f"  [uyari] Postgres durum kaydi atlandi: {exc}")


def register(video_id: str, local_path: str, telemetry_path: str | None = None) -> str:
    ensure_buckets()
    client = get_client()

    suffix = Path(local_path).suffix or ".mp4"
    source_path = f"{video_id}/raw{suffix}"
    client.fput_object(config.MINIO_BUCKET_RAW, source_path, local_path)
    print(f"  video  -> {config.MINIO_BUCKET_RAW}/{source_path}  [{backend_name()}]")

    if telemetry_path:
        telemetry_key = f"{video_id}/telemetry{Path(telemetry_path).suffix or '.tlog'}"
        client.fput_object(config.MINIO_BUCKET_RAW, telemetry_key, telemetry_path)
        print(f"  telemetri -> {config.MINIO_BUCKET_RAW}/{telemetry_key}")

    _record_state(video_id, source_path)
    return source_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_id", nargs="?")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--telemetry", help="MAVLink .tlog dosyasi (opsiyonel)")
    ap.add_argument("--dir", help="Klasordeki tum videolari kaydet")
    args = ap.parse_args()

    if args.dir:
        folder = Path(args.dir)
        if not folder.is_dir():
            print(f"Klasor bulunamadi: {folder}")
            return 1
        videos = sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            print(f"{folder} icinde video bulunamadi")
            return 1
        for path in videos:
            print(f"{path.stem}:")
            register(path.stem, str(path))
        print(f"\n{len(videos)} video kaydedildi")
        return 0

    if not args.video_id or not args.path:
        ap.print_help()
        return 1

    print(f"{args.video_id}:")
    register(args.video_id, args.path, args.telemetry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
