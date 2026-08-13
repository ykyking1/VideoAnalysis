"""AU-AIR videolarının SADECE proxy'sini yerel MinIO'ya yükler - embedding/
YOLO/telemetri/Qdrant yazımı YAPMAZ. Bu adımlar Kaggle'da (gerçek GPU'da)
zaten yapıldı; Qdrant noktaları `poc/auair_import_kaggle_export.py` ile ayrı
aktarılıyor (bkz. docs/worklog_2026-08-13.md, o gün 8 videonun tamamı
Kaggle'da ingest edilip `vehicle_count` AU-AIR gerçek etiketiyle düzeltildi).

NEDEN AYRI BİR SCRIPT: `query_ui.py`'nin önizleme üretimi (fetch_preview_clip)
proxy'yi MinIO'dan (`{video_id}/proxy.mp4`) okuyor - Qdrant'ta nokta olması
yetmiyor, video dosyasının da yerel MinIO'da olması gerekiyor. Proxy üretimi
(ffmpeg transkod) GPU/model gerektirmez - zayıf yerel GPU'da bile hızlı,
8 videonun tamamı için tam yeniden embed etmekten (saatler) çok daha ucuz.

Kullanım:
    python -m poc.auair_local_proxies --all
    python -m poc.auair_local_proxies frame_20190905111947
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auair_adapter import load_auair_records, split_by_source_video  # noqa: E402

_DEFAULT_VIDEOS_DIR = r"c:\Users\PC_4150_YD26\VideoAnalysis\data\auair_sample\videos"
_DEFAULT_ANNOTATIONS = r"c:\Users\PC_4150_YD26\VideoAnalysis\data\auair_sample\annotations.json"


async def make_proxy(prefix: str, videos_dir: Path) -> None:
    from ingest.activities.proxy_generation import generate_proxy
    from scripts.register_video import register

    video_id = f"auair_{prefix}"
    video_path = videos_dir / f"{prefix}.mp4"
    if not video_path.is_file():
        print(f"{video_id}: ATLANDI - dosya yok ({video_path})")
        return
    print(f"{video_id}: register + proxy uretiliyor...")
    source_path = register(video_id, str(video_path))
    proxy_key = await generate_proxy(video_id, source_path)
    print(f"  tamam: {proxy_key}")


async def main_async(args) -> int:
    videos_dir = Path(args.videos_dir)
    if args.all:
        records = load_auair_records(args.annotations)
        groups = split_by_source_video(records)
        targets = sorted(groups)
    else:
        targets = [args.video]
    for prefix in targets:
        await make_proxy(prefix, videos_dir)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?", help="ör. frame_20190905111947")
    ap.add_argument("--all", action="store_true", help="8 videonun hepsi icin proxy uret")
    ap.add_argument("--videos-dir", default=_DEFAULT_VIDEOS_DIR)
    ap.add_argument("--annotations", default=_DEFAULT_ANNOTATIONS)
    args = ap.parse_args()
    if not args.video and not args.all:
        ap.error("bir video adi VEYA --all verin")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
