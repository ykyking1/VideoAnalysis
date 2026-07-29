"""Tek bir videoyu ingest eder.

İki mod:
  (varsayılan) Temporal'a workflow gönderir - üretim yolu: dayanıklı,
      checkpoint'li, worker'lar arasında dağıtılır. `python -m ingest.worker`
      çalışıyor olmalı.
  --local  aktiviteleri Temporal olmadan doğrudan sırayla çalıştırır. Tek
      makinede hızlı deneme/hata ayıklama için; retry ve checkpoint YOK.

--local modu sonunda ölçülen gerçek-zaman katı basılır. Bu sayı doğrudan
proje-ozeti.md §8'in en kritik doğrulanmamış varsayımını (embedding ~40x
gerçek-zaman -> ~7.500 GPU-saat) test eder - kendi donanımınızda ölçüp
§8'i güncelleyin.

Kullanım:
    python -m scripts.ingest_video <video_id> <minio_object_key>
    python -m scripts.ingest_video <video_id> <key> --local
    python -m scripts.ingest_video <video_id> <key> --telemetry <tlog_key>
"""
import argparse
import asyncio
import sys
import time

from common import config


async def run_temporal(video_id: str, source_path: str, telemetry_path: str | None,
                        sensor_type: str) -> int:
    from temporalio.client import Client

    from ingest.workflow import VideoIngestWorkflow

    client = await Client.connect(config.TEMPORAL_HOST, namespace=config.TEMPORAL_NAMESPACE)
    print(f"Temporal'a gonderiliyor: {video_id} (queue={config.TEMPORAL_TASK_QUEUE})")

    handle = await client.start_workflow(
        VideoIngestWorkflow.run,
        args=(video_id, source_path, telemetry_path, sensor_type),
        id=f"ingest-{video_id}",
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    print(f"workflow_id={handle.id} - sonuc bekleniyor "
          f"(worker calismiyorsa burada bekler: python -m ingest.worker)")

    result = await handle.result()
    print(f"\nTamamlandi: {result.windows_written} pencere yazildi, "
          f"{result.captions_generated} caption uretildi")
    if result.warnings:
        print(f"Uyarilar: {', '.join(result.warnings)}")
    return 0


async def run_local(video_id: str, source_path: str, telemetry_path: str | None,
                     sensor_type: str, skip_embedding: bool = False,
                     skip_visual: bool = False, skip_caption: bool = False) -> int:
    from ingest.activities.clip_embedding import embed_clips, unload_model
    from ingest.activities.proxy_generation import generate_proxy
    from ingest.activities.selective_caption import generate_captions
    from ingest.activities.telemetry_processing import process_telemetry
    from ingest.activities.visual_fields import extract_visual_fields
    from ingest.activities.write_clips import write_clips

    started = time.perf_counter()

    print("[1/6] proxy uretiliyor...")
    proxy_key = await generate_proxy(video_id, source_path)

    print("[2/6] telemetri isleniyor / pencereler uretiliyor...")
    windows = await process_telemetry(video_id, proxy_key, telemetry_path, sensor_type)
    if not windows:
        print("Pencere uretilemedi - video suresi okunamadi mi?")
        return 1
    video_duration_s = windows[-1].t_end
    print(f"      {len(windows)} pencere ({video_duration_s:.1f}s video)")

    embed_elapsed = 0.0
    if skip_embedding:
        print("[3/6] embedding ATLANDI (--skip-embedding)")
        embeddings = []
    else:
        print(f"[3/6] embedding ({len(windows)} pencere, batch={config.EMBEDDING_BATCH_SIZE})...")
        embed_started = time.perf_counter()
        embeddings = await embed_clips(video_id, proxy_key, windows)
        embed_elapsed = time.perf_counter() - embed_started
        unload_model()  # tek GPU'da siradaki modele yer ac

    if skip_visual:
        print("[4/6] gorsel alanlar ATLANDI (--skip-visual)")
        visual = []
    else:
        print("[4/6] gorsel alanlar (YOLO)...")
        visual_started = time.perf_counter()
        visual = await extract_visual_fields(video_id, proxy_key, windows)
        print(f"      {time.perf_counter() - visual_started:.1f}s, "
              f"toplam {sum(v.vehicle_count for v in visual)} arac tespiti")

    if skip_caption:
        print("[5/6] caption ATLANDI (--skip-caption)")
        captions = {}
    else:
        print("[5/6] secici caption...")
        captions = await generate_captions(video_id, proxy_key, windows)

    if skip_embedding:
        # Embedding olmadan Qdrant'a yazamayiz (vektor zorunlu) - bu mod
        # proxy/pencereleme/YOLO adimlarini dogrulamak icin.
        print("[6/6] Qdrant yazimi ATLANDI (embedding yok)")
        written = 0
    else:
        print("[6/6] Qdrant'a yaziliyor...")
        written = await write_clips(video_id, windows, embeddings, visual, captions)

    elapsed = time.perf_counter() - started
    dim = len(embeddings[0]) if embeddings else 0
    print(f"\n--- {video_id} tamamlandi ---")
    print(f"Pencere         : {len(windows)} ({video_duration_s:.1f}s video)")
    print(f"Yazilan nokta   : {written}")
    print(f"Caption         : {len([c for c in captions.values() if c])}")
    print(f"Toplam sure     : {elapsed:.1f}s ({video_duration_s / elapsed if elapsed else 0:.2f}x gercek-zaman)")
    if embed_elapsed:
        print(f"Embedding suresi: {embed_elapsed:.1f}s "
              f"({video_duration_s / embed_elapsed:.2f}x gercek-zaman)")
        print(f"Vektor boyutu   : {dim}d -> {written * dim * 4 / 1024:.1f} KB (fp32)")
        print("\nNOT: proje-ozeti.md §8 embedding icin ~40x gercek-zaman varsayiyor ve bu "
              "\nvarsayim DOGRULANMADI. Yukaridaki olcumu §8 tablosuna islemek, kapasite "
              "\nplanlamasinin dogru temele oturmasi icin sart.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_id")
    ap.add_argument("source_path", help="MinIO raw bucket'indaki object key")
    ap.add_argument("--telemetry", default=None, help="MinIO'daki .tlog object key")
    ap.add_argument("--sensor-type", default="unknown")
    ap.add_argument("--local", action="store_true",
                     help="Temporal olmadan dogrudan calistir (debug)")
    ap.add_argument("--skip-embedding", action="store_true",
                     help="Embedding'i atla (--local). Qdrant yazimi da atlanir - "
                          "proxy/pencereleme/YOLO adimlarini ayri dogrulamak icin.")
    ap.add_argument("--skip-visual", action="store_true", help="YOLO'yu atla (--local)")
    ap.add_argument("--skip-caption", action="store_true",
                     help="Caption'i atla (--local). vLLM yoksa zaten bos gecer.")
    args = ap.parse_args()

    if not args.local and (args.skip_embedding or args.skip_visual or args.skip_caption):
        print("--skip-* secenekleri yalnizca --local ile kullanilabilir "
              "(Temporal workflow'u tum aktiviteleri calistirir).")
        return 1

    if args.local:
        return asyncio.run(run_local(
            args.video_id, args.source_path, args.telemetry, args.sensor_type,
            args.skip_embedding, args.skip_visual, args.skip_caption))
    return asyncio.run(run_temporal(args.video_id, args.source_path,
                                     args.telemetry, args.sensor_type))


if __name__ == "__main__":
    sys.exit(main())
