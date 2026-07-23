"""Adım 0, öncelik 1: gerçek envanter rakamlarını toplar (proje-ozeti.md §8, §11
madde 1). "1,5 PB -> 300.000 saat video" varsayımı, tüm depolama/klip sayısı
hesaplarının kökü ve hiç doğrulanmadı - bu script onun yerine gerçek dosya
sayısı/süre/bitrate dağılımını çıkarır.

Kullanım: python poc/inventory_scan.py <minio-bucket>
"""
import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass
class VideoInfo:
    path: str
    duration_s: float
    bitrate_kbps: float
    size_bytes: int
    codec: str


def probe_video(path: str) -> VideoInfo:
    """ffprobe ile tek bir videonun süre/bitrate/codec/boyut bilgisini çıkarır."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,bit_rate,size:stream=codec_name",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    fmt = data["format"]
    codec = data["streams"][0]["codec_name"] if data.get("streams") else "unknown"
    return VideoInfo(
        path=path,
        duration_s=float(fmt["duration"]),
        bitrate_kbps=float(fmt.get("bit_rate", 0)) / 1000,
        size_bytes=int(fmt["size"]),
        codec=codec,
    )


def scan_bucket(bucket: str) -> list[VideoInfo]:
    """MinIO bucket'ındaki tüm video dosyalarını listeleyip probe_video ile
    tarar. mc (MinIO client) kurulu olmalı."""
    raise NotImplementedError(
        "MinIO listeleme mantığı (mc ls / mc find) ortam erişimi netleşince eklenmeli"
    )


def summarize(videos: list[VideoInfo]) -> dict:
    total_duration_h = sum(v.duration_s for v in videos) / 3600
    total_size_tb = sum(v.size_bytes for v in videos) / 1e12
    return {
        "file_count": len(videos),
        "total_duration_hours": total_duration_h,
        "total_size_tb": total_size_tb,
        "avg_bitrate_kbps": sum(v.bitrate_kbps for v in videos) / len(videos) if videos else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bucket")
    args = parser.parse_args()

    videos = scan_bucket(args.bucket)
    print(json.dumps(summarize(videos), indent=2))


if __name__ == "__main__":
    sys.exit(main())
