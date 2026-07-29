"""MinIO raw bucket'ına Kafka bildirim kuralı ekler
(proje-ozeti.md §3.1 tetik zinciri: MinIO -> Kafka -> Temporal).

MinIO'nun Kafka HEDEFİ sunucu tarafında tanımlı olmalı; bu, MinIO'nun kendi
ortam değişkenleriyle yapılır ve docker-compose.yml'de zaten ayarlı:

    MINIO_NOTIFY_KAFKA_ENABLE_primary=on
    MINIO_NOTIFY_KAFKA_BROKERS_primary=kafka:9092
    MINIO_NOTIFY_KAFKA_TOPIC_primary=video-ingest

Bu script yalnızca bucket -> hedef eşlemesini kurar (ARN'ı bucket'a bağlar).

Kullanım:
    python -m scripts.setup_minio_notifications
"""
import sys

from minio.notificationconfig import NotificationConfig, QueueConfig

from common import config
from common.minio_client import ensure_buckets, get_client

ARN = "arn:minio:sqs::primary:kafka"


def main() -> int:
    ensure_buckets()
    client = get_client()

    notification = NotificationConfig(
        queue_config_list=[
            QueueConfig(ARN, ["s3:ObjectCreated:*"], config_id="video-ingest"),
        ]
    )
    try:
        client.set_bucket_notification(config.MINIO_BUCKET_RAW, notification)
    except Exception as exc:  # noqa: BLE001
        print(f"Bildirim kurulamadi: {exc}")
        print("\nOlasi neden: MinIO tarafinda Kafka hedefi ('primary') tanimli degil.")
        print("docker-compose.yml'deki MINIO_NOTIFY_KAFKA_* degiskenlerini ve "
              "Kafka'nin ayakta oldugunu kontrol edin.")
        return 1

    print(f"Bildirim kuruldu: {config.MINIO_BUCKET_RAW} -> {ARN} "
          f"(topic={config.KAFKA_TOPIC})")
    print("Artik bu bucket'a dusen her video ingest'i otomatik tetikler.")
    print("Tuketiciyi baslatin: python -m ingest.kafka_consumer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
