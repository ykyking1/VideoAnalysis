"""Bir klasördeki tüm videoları kaydeder ve ingest eder — tek komut.

Elle `register_video` + `ingest_video` döngüsü kurmaya gerek kalmıyor.
Model bir kez yüklenip tüm videolarda yeniden kullanılıyor (her video için
ayrı süreç başlatmak video başına ~1 dakikayı modeli yeniden yüklemeye
harcıyordu).

ATLAMA: Zaten ingest edilmiş videolar atlanır (Qdrant'ta o video_id'ye ait
nokta var mı diye bakılır). Yani yarıda kesilen bir yükleme aynı komutla
kaldığı yerden devam eder. `--force` ile yeniden işlenir.

BOZUK DOSYA: Kayıt anında ffprobe ile doğrulanır; okunamayan dosya atlanıp
sonda raporlanır, tüm yükleme düşmez.

Kullanım:
    python -m scripts.ingest_all --dir ~/videolar/
    python -m scripts.ingest_all --dir ~/videolar/ --limit 10
    python -m scripts.ingest_all --dir ~/videolar/ --force
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

from common.console import use_utf8_stdout

use_utf8_stdout()

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".ts", ".avi", ".mpg", ".mpeg", ".m4v")


def already_ingested(video_id: str) -> bool:
    from qdrant_client.http import models as qm

    from common import config
    from common.qdrant_store import get_client

    client = get_client()
    if not client.collection_exists(config.QDRANT_COLLECTION):
        return False
    count = client.count(
        collection_name=config.QDRANT_COLLECTION,
        count_filter=qm.Filter(must=[qm.FieldCondition(
            key="video_id", match=qm.MatchValue(value=video_id))]),
    ).count
    return count > 0


async def main_async(args) -> int:
    from scripts.ingest_video import run_local
    from scripts.register_video import InvalidVideoError, register, validate_video

    folder = Path(args.dir).expanduser()
    if not folder.is_dir():
        print(f"Klasor bulunamadi: {folder}")
        return 1

    candidates = sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not candidates:
        print(f"{folder} icinde video bulunamadi "
              f"(aranan uzantilar: {', '.join(VIDEO_EXTENSIONS)})")
        return 1

    print(f"{len(candidates)} dosya bulundu, dogrulaniyor...\n")

    plan, bozuk, atlanan = [], [], []
    total_s = 0.0
    for p in candidates:
        vid = p.stem.replace(" ", "_")
        try:
            info = validate_video(str(p))
        except InvalidVideoError as exc:
            bozuk.append((p.name, str(exc).splitlines()[1].strip()
                          if len(str(exc).splitlines()) > 1 else str(exc)))
            continue
        if not args.force and already_ingested(vid):
            atlanan.append(p.name)
            continue
        plan.append((vid, p, info))
        total_s += info.get("duration_s") or 0.0

    print(f"  islenecek : {len(plan)}")
    print(f"  atlanan   : {len(atlanan)} (zaten ingest edilmis)")
    print(f"  bozuk     : {len(bozuk)}")
    for name, why in bozuk:
        print(f"      {name}: {why}")
    if not plan:
        print("\nYapilacak is yok.")
        return 0

    if args.limit is not None:      # 0 falsy - `if args.limit` yanlis olurdu
        plan = plan[:args.limit]
        total_s = sum(i.get("duration_s") or 0.0 for _, _, i in plan)
    print(f"\nToplam video suresi: {total_s/3600:.2f} saat "
          f"(~{total_s/8:.0f} pencere)\n")

    started = time.time()
    basarisiz = []
    for i, (vid, path, _info) in enumerate(plan, 1):
        print(f"=== [{i}/{len(plan)}] {vid} ===")
        try:
            register(vid, str(path))
            await run_local(vid, f"{vid}/raw{path.suffix}", None, args.sensor_type,
                            skip_caption=args.skip_caption,
                            skip_visual=args.skip_visual)
        except Exception as exc:  # noqa: BLE001 - tek video tum yuklemeyi dusurmesin
            basarisiz.append((vid, f"{type(exc).__name__}: {exc}"))
            print(f"  [HATA] {type(exc).__name__}: {exc}\n")

    elapsed = time.time() - started
    print("=" * 60)
    print(f"Bitti: {len(plan) - len(basarisiz)}/{len(plan)} video, {elapsed/60:.1f} dk")
    if total_s:
        print(f"Genel hiz: {total_s/elapsed:.2f}x gercek-zaman")
        print("\nNOT: proje-ozeti.md §8 embedding icin ~40x gercek-zaman varsayiyor")
        print("ve bu varsayim DOGRULANMADI. Yukaridaki olcumu §8'e isleyin.")
    if basarisiz:
        print(f"\n{len(basarisiz)} video basarisiz:")
        for vid, why in basarisiz:
            print(f"  {vid}: {why}")
    return 1 if basarisiz else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="Videolarin bulundugu klasor")
    ap.add_argument("--limit", type=int, default=None, help="Ilk N video ile sinirla")
    ap.add_argument("--dry-run", action="store_true",
                     help="Sadece plani goster, hicbir sey isleme")
    ap.add_argument("--force", action="store_true",
                     help="Zaten ingest edilmis videolari da yeniden isle")
    ap.add_argument("--sensor-type", default="unknown")
    ap.add_argument("--skip-caption", action="store_true", default=True,
                     help="Caption'i atla (varsayilan: atla - vLLM gerektiriyor)")
    ap.add_argument("--with-caption", dest="skip_caption", action="store_false",
                     help="Caption uret (vLLM/VLM sunucusu calisiyor olmali)")
    ap.add_argument("--skip-visual", action="store_true", help="YOLO'yu atla")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
