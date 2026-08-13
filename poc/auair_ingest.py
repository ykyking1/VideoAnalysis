"""AU-AIR videolarını GERÇEK ingest hattımızdan geçirir - poc/auair_adapter.py
+ poc/auair_build_videos.py'nin devamı, "AU-AIR bizim kullanım amacımıza
uyar mı" sorusunun son adımı (bkz. o dosyaların docstring'leri - lisans/
varsayım notları AYNEN geçerli).

NEDEN scripts/ingest_video.py::run_local()'IN AYNISI DEĞİL DE KOPYASI:
tek fark [2/6] adımı - process_telemetry() (MAVLink) yerine AU-AIR
adaptörünü kullanıyoruz. Geri kalan 5 adım (proxy/embedding/YOLO/caption/
Qdrant yazımı) `ingest/activities/*.py`'den DEĞİŞTİRİLMEDEN import ediliyor
- production kodu bu poc denemesi için değiştirilmedi.

YOLO vs AU-AIR GERÇEK ETİKET (2026-08-13 worklog): YOLO26'nın bu veri
setinde `vehicle_count` recall'ü ölçüldü - %16 (COCO ön-eğitimli model,
aerial/nadir açı için fine-tune EDİLMEDİ - bkz. worklog'un ilgili bölümü).
AU-AIR'in kendisi zaten gerçek `bbox` etiketi taşıyor - YOLO'nun amacı
insan etiketi OLMAYAN videolarda bu alanı üretmektir, burada gereksiz.
Bu yüzden [4/6] adımında YOLO hâlâ ÇALIŞIYOR (mekanizma sadakati için) ama
AU-AIR'in gerçek etiketinin dolu olduğu pencerelerde YOLO'nun tahminini
EZMİYORUZ - bkz. `_merge_visual_fields()`. `--skip-visual` ile YOLO adımı
tamamen atlanabilir (yerel/yavaş GPU'da hız için - GT zaten var, YOLO'nun
katkısı bu veri setinde marjinal).

AYRI KOLEKSİYON: QDRANT_COLLECTION=clips_auair_test ile çalıştırılmalı
(bkz. çağrı örneği altta) - gerçek "clips" koleksiyonuna (SeaDroneSee +
gerçek arşiv testleri) karışmasın diye.

Yollar CLI argümanı - bu makinede (Windows) ve Kaggle/Colab'da (Linux)
AYNI kod çalışsın diye sabit yol YOK.

Kullanım:
    QDRANT_COLLECTION=clips_auair_test python -m poc.auair_ingest frame_20190905111947
    QDRANT_COLLECTION=clips_auair_test python -m poc.auair_ingest --all
    QDRANT_COLLECTION=clips_auair_test python -m poc.auair_ingest frame_X --skip-visual
    # Kaggle/Colab: --videos-dir/--annotations ile yerel yollari verin
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auair_adapter import (  # noqa: E402
    gt_vehicle_count_for_window,
    load_auair_records,
    load_gt_vehicle_counts,
    split_by_source_video,
)

_DEFAULT_VIDEOS_DIR = r"c:\Users\PC_4150_YD26\VideoAnalysis\data\auair_sample\videos"
_DEFAULT_ANNOTATIONS = r"c:\Users\PC_4150_YD26\VideoAnalysis\data\auair_sample\annotations.json"


def _merge_visual_fields(windows, group, gt_counts_by_image, yolo_visual):
    """AU-AIR gerçek etiketi ile YOLO çıktısını birleştirir - GT varsa YOLO
    ONU OVERRIDE ETMEZ (bkz. modül docstring'i). GT'nin olmadığı (pencereye
    hiç AU-AIR karesi düşmeyen, teorik/kenar durum) pencerelerde YOLO'nun
    tahminine geri düşülür."""
    from ingest.activities.types import VisualFields

    merged, n_gt = [], 0
    for w, yolo in zip(windows, yolo_visual):
        gt = gt_vehicle_count_for_window(w.t_start, w.t_end, group, gt_counts_by_image)
        if gt is not None:
            merged.append(VisualFields(vehicle_count=gt))
            n_gt += 1
        else:
            merged.append(yolo)
    return merged, n_gt


async def ingest_one(prefix: str, group: list[dict], videos_dir: Path,
                      gt_counts_by_image: dict[str, int],
                      skip_caption: bool = True, skip_visual: bool = False) -> dict:
    from ingest.activities.clip_embedding import embed_clips, unload_model
    from ingest.activities.proxy_generation import generate_proxy
    from ingest.activities.selective_caption import generate_captions
    from ingest.activities.telemetry_processing import build_windows, enrich_windows, probe_duration
    from ingest.activities.types import VisualFields
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

    if skip_visual:
        print("[4/6] YOLO ATLANDI (--skip-visual) - sadece AU-AIR gercek etiketi kullanilacak")
        yolo_visual = [VisualFields(vehicle_count=0) for _ in windows]
    else:
        print("[4/6] gorsel alanlar (YOLO)...")
        yolo_visual = await extract_visual_fields(video_id, proxy_key, windows)
        print(f"      YOLO ham tahmini: toplam {sum(v.vehicle_count for v in yolo_visual)} arac "
              f"(asagida AU-AIR gercek etiketiyle EZILECEK, sadece bilgi amacli)")

    visual, n_gt = _merge_visual_fields(windows, group, gt_counts_by_image, yolo_visual)
    print(f"      {n_gt}/{len(windows)} pencerede AU-AIR gercek etiketi kullanildi "
          f"(YOLO var olan sutunu OVERRIDE ETMEDI)")
    print(f"      toplam {sum(v.vehicle_count for v in visual)} arac (nihai, GT-oncelikli)")

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
    gt_counts_by_image = load_gt_vehicle_counts(args.annotations)
    videos_dir = Path(args.videos_dir)

    targets = sorted(groups) if args.all else [args.video]
    results = []
    for prefix in targets:
        results.append(await ingest_one(prefix, groups[prefix], videos_dir, gt_counts_by_image,
                                         skip_caption=not args.with_caption,
                                         skip_visual=args.skip_visual))

    print(f"\n{'='*60}\nOZET ({len(results)} video)\n{'='*60}")
    for r in results:
        print(f"  {r['video_id']}: {r['written']} nokta, {r['duration_s']/r['elapsed_s']:.2f}x gercek-zaman, "
              f"{r['vehicle_total']} arac (AU-AIR gercek etiketi)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?", help="ör. frame_20190905111947")
    ap.add_argument("--all", action="store_true", help="8 videonun hepsini ingest et")
    ap.add_argument("--with-caption", action="store_true", help="vLLM acik ise caption da uret")
    ap.add_argument("--skip-visual", action="store_true",
                     help="YOLO adimini atla (hiz icin - AU-AIR'in gercek etiketi zaten var)")
    ap.add_argument("--videos-dir", default=_DEFAULT_VIDEOS_DIR)
    ap.add_argument("--annotations", default=_DEFAULT_ANNOTATIONS)
    args = ap.parse_args()
    if not args.video and not args.all:
        ap.error("bir video adi VEYA --all verin")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
