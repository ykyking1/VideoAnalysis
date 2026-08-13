"""AU-AIR videolarını GERÇEK ingest hattımızdan geçirir - poc/auair_adapter.py
+ poc/auair_build_videos.py'nin devamı, "AU-AIR bizim kullanım amacımıza
uyar mı" sorusunun son adımı (bkz. o dosyaların docstring'leri - lisans/
varsayım notları AYNEN geçerli).

NEDEN scripts/ingest_video.py::run_local()'IN AYNISI DEĞİL DE KOPYASI:
tek fark [2/6] adımı - process_telemetry() (MAVLink) yerine AU-AIR
adaptörünü kullanıyoruz. Geri kalan 5 adım (proxy/embedding/YOLO/caption/
Qdrant yazımı) `ingest/activities/*.py`'den DEĞİŞTİRİLMEDEN import ediliyor
- production kodu bu poc denemesi için değiştirilmedi.

AYRI KOLEKSİYON: QDRANT_COLLECTION=clips_auair_test ile çalıştırılmalı
(bkz. çağrı örneği altta) - gerçek "clips" koleksiyonuna (SeaDroneSee +
gerçek arşiv testleri) karışmasın diye.

Yollar CLI argümanı - bu makinede (Windows) ve Kaggle/Colab'da (Linux)
AYNI kod çalışsın diye sabit yol YOK.

Kullanım:
    QDRANT_COLLECTION=clips_auair_test python -m poc.auair_ingest frame_20190905111947
    QDRANT_COLLECTION=clips_auair_test python -m poc.auair_ingest --all
    # Kaggle/Colab: --videos-dir/--annotations ile yerel yollari verin
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auair_adapter import load_auair_records, split_by_source_video  # noqa: E402

_DEFAULT_VIDEOS_DIR = r"c:\Users\PC_4150_YD26\VideoAnalysis\data\auair_sample\videos"
_DEFAULT_ANNOTATIONS = r"c:\Users\PC_4150_YD26\VideoAnalysis\data\auair_sample\annotations.json"


async def ingest_one(prefix: str, group: list[dict], videos_dir: Path, skip_caption: bool = True) -> dict:
    from ingest.activities.clip_embedding import embed_clips, unload_model
    from ingest.activities.proxy_generation import generate_proxy
    from ingest.activities.selective_caption import generate_captions
    from ingest.activities.telemetry_processing import build_windows, enrich_windows, probe_duration
    from ingest.activities.visual_fields import extract_visual_fields
    from ingest.activities.write_clips import write_clips
    from common.minio_client import download_temp
    from common import config
    from scripts.register_video import register

    video_id = f"auair_{prefix}"
    video_path = videos_dir / f"{prefix}.mp4"
    started = time.perf_counter()

    print(f"\n=== {video_id} ===")
    print("[0/6] register...")
    source_path = register(video_id, str(video_path))

    print("[1/6] proxy uretiliyor...")
    proxy_key = await generate_proxy(video_id, source_path)

    print("[2/6] telemetri (AU-AIR adaptoru - MAVLink DEGIL) isleniyor...")
    with download_temp(config.MINIO_BUCKET_PROXY, proxy_key) as local_proxy:
        duration = probe_duration(local_proxy)
    windows = build_windows(duration)
    windows = enrich_windows(windows, group, sensor_type="rgb")
    print(f"      {len(windows)} pencere ({duration:.1f}s video)")

    print(f"[3/6] embedding ({len(windows)} pencere)...")
    embed_started = time.perf_counter()
    embeddings = await embed_clips(video_id, proxy_key, windows)
    embed_elapsed = time.perf_counter() - embed_started
    unload_model()

    print("[4/6] gorsel alanlar (YOLO)...")
    visual = await extract_visual_fields(video_id, proxy_key, windows)
    print(f"      toplam {sum(v.vehicle_count for v in visual)} arac tespiti "
          f"(AU-AIR gercek etiketiyle KARSILASTIRILMADI - bu adim ayri)")

    captions = {}
    if not skip_caption:
        print("[5/6] secici caption...")
        captions = await generate_captions(video_id, proxy_key, windows)
    else:
        print("[5/6] caption ATLANDI")

    print("[6/6] Qdrant'a yaziliyor...")
    written = await write_clips(video_id, windows, embeddings, visual, captions)

    elapsed = time.perf_counter() - started
    result = {
        "video_id": video_id, "windows": len(windows), "written": written,
        "duration_s": duration, "elapsed_s": elapsed, "embed_s": embed_elapsed,
        "vehicle_total": sum(v.vehicle_count for v in visual),
    }
    print(f"--- {video_id}: {written} nokta, {elapsed:.1f}s "
          f"({duration/elapsed:.2f}x gercek-zaman) ---")
    return result


async def main_async(args) -> int:
    records = load_auair_records(args.annotations)
    groups = split_by_source_video(records)
    videos_dir = Path(args.videos_dir)

    targets = sorted(groups) if args.all else [args.video]
    results = []
    for prefix in targets:
        results.append(await ingest_one(prefix, groups[prefix], videos_dir,
                                         skip_caption=not args.with_caption))

    print(f"\n{'='*60}\nOZET ({len(results)} video)\n{'='*60}")
    for r in results:
        print(f"  {r['video_id']}: {r['written']} nokta, {r['duration_s']/r['elapsed_s']:.2f}x gercek-zaman, "
              f"{r['vehicle_total']} arac tespiti")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?", help="ör. frame_20190905111947")
    ap.add_argument("--all", action="store_true", help="8 videonun hepsini ingest et")
    ap.add_argument("--with-caption", action="store_true", help="vLLM acik ise caption da uret")
    ap.add_argument("--videos-dir", default=_DEFAULT_VIDEOS_DIR)
    ap.add_argument("--annotations", default=_DEFAULT_ANNOTATIONS)
    args = ap.parse_args()
    if not args.video and not args.all:
        ap.error("bir video adi VEYA --all verin")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
