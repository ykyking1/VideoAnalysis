"""Model proxy üretimi: ffmpeg + NVDEC decode, 240-360p HEVC (proje-ozeti.md §3.1
madde 1). Önizleme için değil, model tüketimi için.

Donanım notu: GT1030 (GP108) NVENC içermiyor (yalnızca NVDEC) - donanım testiyle
doğrulandı. Decode NVDEC ile hızlandırılıyor, encode yazılımla (libx265) yapılıyor.
Bu, düşük çözünürlük/bitrate hedefinde (§4) hâlâ gerçek-zamanın çok üzerinde hız
veriyor (ölçüm: 267 fps @ 360p sentetik testte).
"""
import subprocess
import tempfile
from pathlib import Path

from temporalio import activity

from common import config
from common.minio_client import get_client

TARGET_HEIGHT = 360
TARGET_BITRATE = "800k"


def _transcode(src_path: Path, dst_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", str(src_path),
            "-vf", f"scale_cuda=-2:{TARGET_HEIGHT},hwdownload,format=nv12",
            "-c:v", "libx265", "-preset", "fast", "-b:v", TARGET_BITRATE,
            "-an",
            str(dst_path),
        ],
        check=True, capture_output=True, text=True,
    )


@activity.defn
async def generate_proxy(video_id: str, source_path: str) -> str:
    """MinIO raw bucket'ındaki source_path'i indirir, model-kalite proxy üretir,
    proxy bucket'a yükler ve proxy'nin object key'ini döner."""
    client = get_client()
    proxy_key = f"{video_id}/proxy.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        dst = Path(tmp) / "proxy.mp4"
        client.fget_object(config.MINIO_BUCKET_RAW, source_path, str(src))
        _transcode(src, dst)
        client.fput_object(config.MINIO_BUCKET_PROXY, proxy_key, str(dst))

    return proxy_key
