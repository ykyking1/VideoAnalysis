"""Temporal ingest workflow (proje-ozeti.md §3.1).

Tetik: MinIO bucket notification -> Kafka -> ingest/kafka_consumer.py -> bu
workflow. Checkpoint + heartbeat sayesinde çöken bir GPU işi kaldığı yerden
devam eder; 3-5 saatlik videolarda bu opsiyonel değil, gereklilik.

AKTİVİTE SIRASI: proxy önce (diğer herkes proxy'yi okur), sonra telemetri
(pencereleri o belirler). Embedding / görsel alanlar / caption üçü de aynı
pencerelere bağlı ve BİRBİRİNDEN BAĞIMSIZ - ama varsayılan olarak sırayla
çalışıyorlar çünkü hepsi aynı GPU'yu kullanıyor; tek GPU'da paralel çalıştırmak
VRAM'i bölüp üçünü birden yavaşlatır (ve OOM riski doğurur). Ayrı GPU'lu
worker havuzlarınız varsa PARALLEL_GPU_ACTIVITIES=true ile paralelleştirin.

RETRY: GPU aktiviteleri 3 denemeye kadar üstel geri çekilmeyle yeniden
denenir. Yazım (write_clips) idempotent olduğu için sınırsız denenebilir -
tekrar yazım satır çoğaltmaz (bkz. common/qdrant_store.point_id).
"""
import asyncio
import os
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ingest.activities.clip_embedding import embed_clips
    from ingest.activities.proxy_generation import generate_proxy
    from ingest.activities.selective_caption import generate_captions
    from ingest.activities.telemetry_processing import process_telemetry
    from ingest.activities.types import IngestResult
    from ingest.activities.visual_fields import extract_visual_fields
    from ingest.activities.write_clips import write_clips

PARALLEL_GPU_ACTIVITIES = os.environ.get(
    "PARALLEL_GPU_ACTIVITIES", "false").strip().lower() in ("1", "true", "yes")

GPU_RETRY = RetryPolicy(maximum_attempts=3, backoff_coefficient=2.0,
                         initial_interval=timedelta(seconds=10))
IO_RETRY = RetryPolicy(maximum_attempts=5, backoff_coefficient=2.0,
                        initial_interval=timedelta(seconds=5))


@workflow.defn
class VideoIngestWorkflow:
    @workflow.run
    async def run(self, video_id: str, source_path: str,
                   telemetry_path: str | None = None,
                   sensor_type: str = "unknown") -> IngestResult:
        warnings: list[str] = []

        proxy_key = await workflow.execute_activity(
            generate_proxy,
            args=(video_id, source_path),
            start_to_close_timeout=timedelta(hours=4),
            retry_policy=GPU_RETRY,
        )

        windows = await workflow.execute_activity(
            process_telemetry,
            args=(video_id, proxy_key, telemetry_path, sensor_type),
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=IO_RETRY,
        )
        if not windows:
            return IngestResult(video_id=video_id, windows_written=0, captions_generated=0,
                                proxy_key=proxy_key, warnings=["pencere uretilemedi"])

        embed_task = workflow.execute_activity(
            embed_clips,
            args=(video_id, proxy_key, windows),
            start_to_close_timeout=timedelta(hours=8),
            retry_policy=GPU_RETRY,
            heartbeat_timeout=timedelta(minutes=10),
        )
        visual_task = workflow.execute_activity(
            extract_visual_fields,
            args=(video_id, proxy_key, windows),
            start_to_close_timeout=timedelta(hours=4),
            retry_policy=GPU_RETRY,
            heartbeat_timeout=timedelta(minutes=10),
        )
        caption_task = workflow.execute_activity(
            generate_captions,
            args=(video_id, proxy_key, windows),
            start_to_close_timeout=timedelta(hours=4),
            retry_policy=GPU_RETRY,
            heartbeat_timeout=timedelta(minutes=10),
        )

        if PARALLEL_GPU_ACTIVITIES:
            embeddings, visual, captions = await asyncio.gather(
                embed_task, visual_task, caption_task)
        else:
            embeddings = await embed_task
            visual = await visual_task
            captions = await caption_task

        if not captions:
            warnings.append("caption uretilmedi")

        written = await workflow.execute_activity(
            write_clips,
            args=(video_id, windows, embeddings, visual, captions),
            start_to_close_timeout=timedelta(hours=1),
            retry_policy=IO_RETRY,
        )

        return IngestResult(
            video_id=video_id,
            windows_written=written,
            captions_generated=len([c for c in captions.values() if c]),
            proxy_key=proxy_key,
            warnings=warnings,
        )
