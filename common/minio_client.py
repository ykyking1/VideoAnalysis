"""Paylaşılan MinIO istemcisi ve bucket başlatma."""
import contextlib
import tempfile
from pathlib import Path

from minio import Minio

from common import config


def get_client() -> Minio:
    return Minio(
        config.MINIO_ENDPOINT,
        access_key=config.MINIO_ROOT_USER,
        secret_key=config.MINIO_ROOT_PASSWORD,
        secure=config.MINIO_SECURE,
    )


@contextlib.contextmanager
def download_temp(bucket: str, object_key: str):
    """Bir MinIO nesnesini geçici dosyaya indirir; `with` bloğu bitince siler.
    Aktivitelerin (gerçek dağıtık Temporal worker'larda birbirinden bağımsız
    çalışabilmesi için) her biri kendi girdisini bağımsızca indirmesi gerekir."""
    with tempfile.TemporaryDirectory() as tmp:
        local_path = str(Path(tmp) / Path(object_key).name)
        get_client().fget_object(bucket, object_key, local_path)
        yield local_path


def ensure_buckets() -> None:
    client = get_client()
    for bucket in (config.MINIO_BUCKET_RAW, config.MINIO_BUCKET_PROXY):
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
