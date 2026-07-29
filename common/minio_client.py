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


class StorageUnavailable(RuntimeError):
    """Nesne deposuna ulasilamadi - mesaj ne yapilmasi gerektigini soyler."""


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

    try:
        from minio import Minio
    except ImportError as exc:
        raise StorageUnavailable(
            "minio paketi kurulu degil ve LOCAL_STORAGE_PATH da tanimli degil.\n"
            "  - Docker'li ortamda:  pip install minio\n"
            "  - Docker'siz (Colab/Kaggle):  export LOCAL_STORAGE_PATH=/content/storage\n"
            "\nDIKKAT: LOCAL_STORAGE_PATH'i ayarladiysaniz ama bu hatayi hala\n"
            "aliyorsaniz, common.config ortam degiskeni ayarlanmadan ONCE import\n"
            "edilmis olabilir (config env'i import aninda okuyor). Notebook'ta:\n"
            "  importlib.reload(sys.modules['common.config'])"
        ) from exc

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
    """Bucket'ları (veya dizinleri) hazırlar.

    MinIO'ya ulaşılamadığında urllib3'ün 40 satırlık yeniden-deneme yığını
    asıl sorunu gizliyor - burada anlaşılır bir mesaja çeviriyoruz."""
    client = get_client()
    try:
        for bucket in (config.MINIO_BUCKET_RAW, config.MINIO_BUCKET_PROXY):
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
    except Exception as exc:  # noqa: BLE001
        if config.LOCAL_STORAGE_PATH:
            raise StorageUnavailable(
                f"Yerel depo dizini kullanilamiyor ({config.LOCAL_STORAGE_PATH}): {exc}"
            ) from exc
        raise StorageUnavailable(
            f"MinIO'ya ulasilamiyor ({config.MINIO_ENDPOINT}).\n"
            f"  - Docker calisiyorsa: docker compose up -d minio\n"
            f"  - Docker YOKSA (Colab/Kaggle) MinIO'ya gerek yok, yerel dizin kullanin:\n"
            f"        export LOCAL_STORAGE_PATH=/content/storage\n"
            f"  Ayrinti: {exc}"
        ) from exc
