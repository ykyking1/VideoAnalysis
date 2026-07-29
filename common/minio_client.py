"""Nesne deposu erişimi: MinIO (üretim) veya yerel dosya sistemi (geliştirme).

NEDEN İKİ ARKA UÇ: Colab/Kaggle gibi ortamlarda Docker yok ve MinIO ikilisini
arka planda çalıştırmak fazladan bir kırılma noktası (ölçüldü: sessizce
ölüyor ve hata `Connection refused` olarak çok sonra ortaya çıkıyor).
Pipeline'ın nesne deposundan tek ihtiyacı "dosya koy / dosya al" olduğu için
yerel dizin de aynı işi görüyor.

LOCAL_STORAGE_PATH tanımlıysa dosya sistemi kullanılır, aksi halde MinIO.

ÜRETİMDE MinIO KULLANIN: dosya sistemi arka ucu tek makineye bağlıdır,
dağıtık worker'lar (proje-ozeti.md §3.1) paylaşımlı nesne deposu gerektirir.
"""
import contextlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from common import config


@dataclass
class _Bucket:
    """minio'nun list_buckets() dönüşüyle uyumlu asgari nesne."""
    name: str


class LocalFilesystemStorage:
    """MinIO API'sinin bu projede kullanılan alt kümesini yerel dizinle karşılar.

    Kullanılan metotlar (tam liste): bucket_exists, make_bucket, list_buckets,
    fget_object, fput_object. Başka bir MinIO özelliği gerekirse burada da
    karşılığı yazılmalı - sessizce AttributeError almamak için."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _bucket_path(self, bucket: str) -> Path:
        return self.root / bucket

    def _object_path(self, bucket: str, key: str) -> Path:
        # Anahtardaki "/" alt dizine cevrilir (MinIO'daki gibi)
        return self._bucket_path(bucket) / key

    def bucket_exists(self, bucket: str) -> bool:
        return self._bucket_path(bucket).is_dir()

    def make_bucket(self, bucket: str) -> None:
        self._bucket_path(bucket).mkdir(parents=True, exist_ok=True)

    def list_buckets(self) -> list[_Bucket]:
        return [_Bucket(p.name) for p in self.root.iterdir() if p.is_dir()]

    def fput_object(self, bucket: str, key: str, file_path: str, **_kwargs) -> None:
        dst = self._object_path(bucket, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file_path, dst)

    def fget_object(self, bucket: str, key: str, file_path: str, **_kwargs) -> None:
        src = self._object_path(bucket, key)
        if not src.is_file():
            raise FileNotFoundError(
                f"{bucket}/{key} bulunamadi ({src}). Video kaydedildi mi? "
                f"-> python -m scripts.register_video"
            )
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, file_path)


def get_client():
    """LOCAL_STORAGE_PATH tanımlıysa dosya sistemi, aksi halde MinIO istemcisi."""
    if config.LOCAL_STORAGE_PATH:
        return LocalFilesystemStorage(config.LOCAL_STORAGE_PATH)

    from minio import Minio

    return Minio(
        config.MINIO_ENDPOINT,
        access_key=config.MINIO_ROOT_USER,
        secret_key=config.MINIO_ROOT_PASSWORD,
        secure=config.MINIO_SECURE,
    )


def backend_name() -> str:
    return f"filesystem ({config.LOCAL_STORAGE_PATH})" if config.LOCAL_STORAGE_PATH \
        else f"minio ({config.MINIO_ENDPOINT})"


@contextlib.contextmanager
def download_temp(bucket: str, object_key: str):
    """Bir nesneyi geçici dosyaya indirir; `with` bloğu bitince siler.
    Aktivitelerin (dağıtık worker'larda birbirinden bağımsız çalışabilmesi
    için) her biri kendi girdisini bağımsızca indirmesi gerekir."""
    with tempfile.TemporaryDirectory() as tmp:
        local_path = str(Path(tmp) / Path(object_key).name)
        get_client().fget_object(bucket, object_key, local_path)
        yield local_path


def ensure_buckets() -> None:
    client = get_client()
    for bucket in (config.MINIO_BUCKET_RAW, config.MINIO_BUCKET_PROXY):
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
