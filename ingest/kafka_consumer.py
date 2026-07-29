"""Kafka tüketicisi: MinIO bucket notification -> Kafka -> Temporal workflow
(proje-ozeti.md §3.1 tetik zinciri).

MinIO, `s3:ObjectCreated:*` olaylarını Kafka'ya JSON olarak basar. Bu süreç o
olayları okuyup her yeni video için bir VideoIngestWorkflow başlatır.

NEDEN ARADA KAFKA VAR: MinIO notification'ı ateşle-unut; Temporal geçici
olarak erişilemezse olay kaybolur. Kafka olayı kalıcı tutar ve tüketici
offset'i ancak workflow başarıyla kuyruğa alındıktan sonra ilerler - yani
ingest tetikleyicisi "en az bir kez" garantisine kavuşur. Workflow ID
video_id'den türetildiği için aynı olayın iki kez işlenmesi ikinci bir
ingest başlatmaz (Temporal ID çakışmasını reddeder).

Kullanım:
    python -m ingest.kafka_consumer
"""
import asyncio
import json
import logging
import urllib.parse

from confluent_kafka import Consumer, KafkaError
from temporalio.client import Client
from temporalio.service import RPCError

from common import config
from ingest.workflow import VideoIngestWorkflow

log = logging.getLogger("ingest.kafka_consumer")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".ts", ".avi", ".mpg", ".mpeg", ".m4v")


def video_id_from_key(object_key: str) -> str:
    """Nesne anahtarından kararlı bir video kimliği türetir.
    `missions/2026-07-01/flight_12.mp4` -> `missions_2026-07-01_flight_12`"""
    without_ext = object_key.rsplit(".", 1)[0]
    return without_ext.replace("/", "_").replace(" ", "_")


def parse_minio_event(raw: bytes) -> list[tuple[str, str]]:
    """MinIO notification JSON'undan (bucket, object_key) çiftlerini çıkarır.
    Video olmayan nesneler (telemetri, thumbnail vb.) elenir."""
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("JSON olmayan Kafka mesaji atlandi")
        return []

    results = []
    for record in event.get("Records", []):
        s3 = record.get("s3", {})
        bucket = s3.get("bucket", {}).get("name")
        key = s3.get("object", {}).get("key")
        if not bucket or not key:
            continue
        # MinIO anahtari URL-encoded gonderir (bosluklar %20 olur)
        key = urllib.parse.unquote_plus(key)
        if not key.lower().endswith(VIDEO_EXTENSIONS):
            continue
        results.append((bucket, key))
    return results


async def start_workflow(client: Client, object_key: str) -> bool:
    """Workflow'u başlatır. Zaten çalışıyorsa True döner (olay tüketilmiş
    sayılır - tekrar başlatmaya çalışmak sonsuz döngü yaratır)."""
    video_id = video_id_from_key(object_key)
    try:
        await client.start_workflow(
            VideoIngestWorkflow.run,
            args=(video_id, object_key, None, "unknown"),
            id=f"ingest-{video_id}",
            task_queue=config.TEMPORAL_TASK_QUEUE,
        )
        log.info("workflow baslatildi: %s", video_id)
        return True
    except RPCError as exc:
        if "already" in str(exc).lower():
            log.info("workflow zaten var, atlandi: %s", video_id)
            return True
        log.error("workflow baslatilamadi (%s): %s", video_id, exc)
        return False


async def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    client = await Client.connect(config.TEMPORAL_HOST, namespace=config.TEMPORAL_NAMESPACE)
    consumer = Consumer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "group.id": config.KAFKA_GROUP_ID,
        "auto.offset.reset": "earliest",
        # Offset'i ELLE isliyoruz: workflow kuyruga alinmadan once offset
        # ilerlerse olay kaybolur.
        "enable.auto.commit": False,
    })
    consumer.subscribe([config.KAFKA_TOPIC])
    log.info("Kafka dinleniyor: %s (topic=%s)",
             config.KAFKA_BOOTSTRAP_SERVERS, config.KAFKA_TOPIC)

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                await asyncio.sleep(0)
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error("Kafka hatasi: %s", msg.error())
                continue

            all_started = True
            for _bucket, key in parse_minio_event(msg.value()):
                if not await start_workflow(client, key):
                    all_started = False

            if all_started:
                consumer.commit(message=msg, asynchronous=False)
    finally:
        consumer.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
