"""AU-AIR'in ayrık karelerinden, GERÇEK zaman damgalarına sadık video
dosyaları üretir - poc/auair_adapter.py'nin devamı (bkz. o dosyanın modül
docstring'i - lisans/varsayım notları AYNEN geçerli).

NEDEN GERÇEK ZAMAN DAMGASI KULLANILIYOR (sabit FPS değil): kareler arası
boşluk çoğunlukla ~0.2sn (medyan, ~5 FPS) ama bazı videolarda 15-50sn'lik
istisnai büyük boşluklar var (ölçüldü, bkz. sohbet kaydı). Sabit FPS ile
birleştirirsek bu boşluklar sessizce yutulur, videonun "gerçek dakika
X'te ne oluyordu" sorusu yanlış cevaplanır. ffmpeg concat demuxer'ının
her kare için AYRI süre belirtme özelliği kullanılıyor.

SONUÇ VİDEO GERÇEK ÇEKİM DEĞİL: kareler ~5 FPS'te (orijinal videonun
tam FPS'i değil) örneklenmiş, yani üretilen video gerçek akıcı çekimden
daha "kesik" görünecek - bu MEKANİZMA testi (ingest/pencereleme/embedding
gerçekten çalışıyor mu), retrieval KALİTESİ testi değil. proje-ozeti.md'nin
diğer "SINIRLI KANIT" notlarıyla aynı kategoride.

Yollar CLI argümanı - bu makinede (Windows) ve Kaggle/Colab'da (Linux,
/kaggle/working gibi) AYNI kod çalışsın diye sabit yol YOK (varsayılanlar
bu makinenin yerel düzenine göre, --images-dir/--out-dir/--annotations
ile değiştirilebilir).
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auair_adapter import load_auair_records, split_by_source_video  # noqa: E402

_DEFAULT_IMAGES_DIR = r"c:\Users\PC_4150_YD26\VideoAnalysis\data\auair_sample\images_archive\images"
_DEFAULT_OUT_DIR = r"c:\Users\PC_4150_YD26\VideoAnalysis\data\auair_sample\videos"
_DEFAULT_ANNOTATIONS = r"c:\Users\PC_4150_YD26\VideoAnalysis\data\auair_sample\annotations.json"


def build_video(prefix: str, group: list[dict], images_dir: Path, out_path: Path) -> None:
    """Bir kaynak videonun kareler listesinden, ffmpeg concat demuxer ile
    gerçek zamanlamalı bir MP4 üretir."""
    group = sorted(group, key=lambda r: r["t"])
    list_path = out_path.with_suffix(".ffconcat.txt")

    lines = ["ffconcat version 1.0"]
    for i, rec in enumerate(group):
        img_path = images_dir / rec["image_name"]
        if not img_path.is_file():
            raise FileNotFoundError(f"kare bulunamadi: {img_path}")
        # Bir sonraki kareye kadar gecen GERCEK sure (son kare icin medyan kullanilir)
        if i < len(group) - 1:
            dur = max(group[i + 1]["t"] - rec["t"], 0.04)
        else:
            dur = 0.2
        lines.append(f"file '{img_path.as_posix()}'")
        lines.append(f"duration {dur:.3f}")
    # ffconcat: son dosya bir kez daha (suresiz) tekrar edilmeli
    lines.append(f"file '{(images_dir / group[-1]['image_name']).as_posix()}'")
    list_path.write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vsync", "vfr", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg basarisiz ({prefix}): {result.stderr[-1000:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?", help="ör. frame_20190905111947 (verilmezse hepsi)")
    ap.add_argument("--images-dir", default=_DEFAULT_IMAGES_DIR)
    ap.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    ap.add_argument("--annotations", default=_DEFAULT_ANNOTATIONS)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_auair_records(args.annotations)
    groups = split_by_source_video(records)

    targets = {args.video: groups[args.video]} if args.video else groups
    for prefix, group in sorted(targets.items()):
        out_path = out_dir / f"{prefix}.mp4"
        print(f"{prefix}: {len(group)} kare -> {out_path.name} ...")
        build_video(prefix, group, images_dir, out_path)
        size_mb = out_path.stat().st_size / 1024**2
        print(f"  tamam: {size_mb:.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
