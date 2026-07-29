"""Model proxy üretimi: ffmpeg + NVDEC decode / NVENC encode, 240-360p HEVC
(proje-ozeti.md §3.1 madde 1, §4).

Önizleme için DEĞİL, model tüketimi için: embedding/detektör/rerank hep bu
proxy'yi okur, ham videoya sorgu hattı hiç dokunmaz (§4 - ham video soğuk
arşive taşınabilir). "İnsan kalitesi" değil "model kalitesi" hedeflenir.

Donanım geri çekilmesi: NVDEC/NVENC yoksa (ya da PROXY_USE_NVENC=false) ffmpeg
yazılım yoluna düşer. Bazı Nvidia kartları (örn. GP108/GT1030) NVDEC içerip
NVENC içermez - bu durumda decode donanımda, encode yazılımda yapılır.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from temporalio import activity

from common import config
from common.minio_client import get_client


class ProxyGenerationError(RuntimeError):
    pass


def _build_command(src: Path, dst: Path, use_nvdec: bool, use_nvenc: bool) -> list[str]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    if use_nvdec:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    cmd += ["-i", str(src)]

    if use_nvdec and use_nvenc:
        # Kare GPU'da kalir - indirme/yukleme yok, en hizli yol.
        cmd += ["-vf", f"scale_cuda=-2:{config.PROXY_HEIGHT}", "-c:v", "hevc_nvenc",
                "-preset", "p4"]
    elif use_nvdec:
        # NVENC yok: kareyi CPU'ya indirip yazilim encode.
        cmd += ["-vf", f"scale_cuda=-2:{config.PROXY_HEIGHT},hwdownload,format=nv12",
                "-c:v", "libx265", "-preset", "fast"]
    elif use_nvenc:
        cmd += ["-vf", f"scale=-2:{config.PROXY_HEIGHT}", "-c:v", "hevc_nvenc", "-preset", "p4"]
    else:
        cmd += ["-vf", f"scale=-2:{config.PROXY_HEIGHT}", "-c:v", "libx265", "-preset", "fast"]

    # -an: ses yok. Sorgu hatti ses kullanmiyor, proxy boyutunu gereksiz buyutur.
    cmd += ["-b:v", config.PROXY_BITRATE, "-an", "-movflags", "+faststart", str(dst)]
    return cmd


def transcode(src_path: Path, dst_path: Path) -> None:
    """Donanım hızlandırmalı yolu dener; başarısız olursa yazılım yoluna düşer.

    Geri çekilme sessizce yapılmaz - hangi yolun kullanıldığı Temporal
    aktivite log'una yazılır ki üretimde 'neden bu kadar yavaş' sorusu
    cevaplanabilsin."""
    if shutil.which("ffmpeg") is None:
        raise ProxyGenerationError("ffmpeg PATH'te bulunamadi")

    attempts = [(config.PROXY_USE_NVDEC, config.PROXY_USE_NVENC)]
    if config.PROXY_USE_NVDEC or config.PROXY_USE_NVENC:
        attempts.append((False, False))  # tam yazilim geri cekilmesi

    last_error = ""
    for use_nvdec, use_nvenc in attempts:
        cmd = _build_command(src_path, dst_path, use_nvdec, use_nvenc)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            _log(f"proxy uretildi (nvdec={use_nvdec}, nvenc={use_nvenc})")
            return
        last_error = result.stderr.strip()[-2000:]
        _log(f"ffmpeg basarisiz (nvdec={use_nvdec}, nvenc={use_nvenc}): {last_error[:300]}")

    raise ProxyGenerationError(f"ffmpeg tum yollarda basarisiz oldu: {last_error}")


def _log(message: str) -> None:
    try:
        activity.logger.info(message)
    except RuntimeError:
        # Aktivite baglami disinda (ornegin scripts/ingest_video.py) cagrildi
        print(message)


@activity.defn
async def generate_proxy(video_id: str, source_path: str) -> str:
    """MinIO raw bucket'ındaki source_path'i indirir, model-kalite proxy üretir,
    proxy bucket'a yükler ve proxy'nin object key'ini döner."""
    client = get_client()
    proxy_key = f"{video_id}/proxy.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / Path(source_path).name
        dst = Path(tmp) / "proxy.mp4"
        client.fget_object(config.MINIO_BUCKET_RAW, source_path, str(src))
        transcode(src, dst)
        client.fput_object(config.MINIO_BUCKET_PROXY, proxy_key, str(dst))

    return proxy_key
