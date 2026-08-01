"""Yerel bir video (ve varsa telemetri log'u) MinIO raw bucket'ına yükler,
Postgres durum tablosuna kaydeder.

Bu, Kafka/MinIO-notification zincirini atlayarak elle ingest tetiklemek için;
üretimde videolar bucket'a düştüğünde zincir kendiliğinden çalışır.

Kullanım:
    python -m scripts.register_video <video_id> <video_dosyasi> [--telemetry <tlog>]
    python -m scripts.register_video --dir <klasor>    # klasordeki tum videolar
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import config
from common.minio_client import backend_name, ensure_buckets, get_client

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".ts", ".m2ts", ".mts", ".avi", ".mpg", ".mpeg", ".m4v")


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


class InvalidVideoError(RuntimeError):
    pass


def validate_video(path: str) -> dict:
    """Dosyanın gerçekten okunabilir bir video olduğunu KAYIT ANINDA doğrular.

    Neden burada: bozuk/eksik bir dosya doğrulanmazsa hata üç adım sonra
    proxy üretiminde `moov atom not found` olarak patlıyor ve orada nedenini
    anlamak imkansız. En sık sebep, yüklemenin yarım kalması."""
    p = Path(path)
    if not p.is_file():
        raise InvalidVideoError(f"Dosya yok: {path}")
    size = p.stat().st_size
    if size == 0:
        raise InvalidVideoError(f"Dosya bos (0 byte): {path}")

    if shutil.which("ffprobe") is None:
        return {"size_bytes": size, "duration_s": None}

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-select_streams", "v:0", "-show_entries", "stream=codec_name",
         "-of", "json", str(p)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "?"
        hint = ""
        if "moov atom not found" in result.stderr or "Invalid data" in result.stderr:
            hint = ("\n  Bu hata dosyanin EKSIK/BOZUK oldugunu gosterir - mp4'un "
                    "indeksi (moov atom)\n  genelde dosya sonunda olur, yukleme "
                    "yarim kalinca kaybolur.\n"
                    "  - Colab'a elden yuklediyseniz: yukleme bitmeden hucreyi "
                    "calistirmis olabilirsiniz\n"
                    "  - Drive'dan aliyorsaniz: once /content'e kopyalayin\n"
                    f"  - Dosya boyutunu kaynakla karsilastirin ({size/1024**2:.1f} MB)")
        raise InvalidVideoError(f"Okunamayan video: {path}\n  ffprobe: {detail}{hint}")

    info = json.loads(result.stdout or "{}")
    duration = info.get("format", {}).get("duration")
    streams = info.get("streams", [])
    if not streams:
        raise InvalidVideoError(f"Dosyada video akisi yok: {path}")

    return {"size_bytes": size,
            "duration_s": float(duration) if duration else None,
            "codec": streams[0].get("codec_name")}


def register(video_id: str, local_path: str, telemetry_path: str | None = None) -> str:
    info = validate_video(local_path)
    ensure_buckets()
    client = get_client()

    suffix = Path(local_path).suffix or ".mp4"
    source_path = f"{video_id}/raw{suffix}"
    client.fput_object(config.MINIO_BUCKET_RAW, source_path, local_path)
    dur = f"{info['duration_s']:.1f}s" if info.get("duration_s") else "?"
    print(f"  video  -> {config.MINIO_BUCKET_RAW}/{source_path}  "
          f"[{info['size_bytes']/1024**2:.1f} MB, {dur}, {info.get('codec','?')}]")

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
