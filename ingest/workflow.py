"""Temporal ingest workflow (proje-ozeti.md §3.1).

Tetik: MinIO bucket notification -> Kafka -> bu workflow. Checkpoint + heartbeat
sayesinde GPU işi çöktüğü pencereden devam eder. Aktivite sırası ve retry
politikaları Adım 0 ölçümleri sonrası ayarlanmalı.
"""
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ingest.activities.proxy_generation import generate_proxy
    from ingest.activities.telemetry_processing import process_telemetry
    from ingest.activities.clip_embedding import embed_clips
    from ingest.activities.visual_fields import extract_visual_fields
    from ingest.activities.selective_caption import generate_captions


GPU_ACTIVITY_RETRY = RetryPolicy(maximum_attempts=3, backoff_coefficient=2.0)


@workflow.defn
class VideoIngestWorkflow:
    @workflow.run
    async def run(self, video_id: str, source_path: str) -> None:
        proxy_path = await workflow.execute_activity(
            generate_proxy,
            args=(video_id, source_path),
            start_to_close_timeout=timedelta(hours=1),
            retry_policy=GPU_ACTIVITY_RETRY,
        )

        telemetry_windows = await workflow.execute_activity(
            process_telemetry,
            args=(video_id,),
            start_to_close_timeout=timedelta(minutes=30),
        )

        embeddings = await workflow.execute_activity(
            embed_clips,
            args=(video_id, proxy_path, telemetry_windows),
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=GPU_ACTIVITY_RETRY,
            heartbeat_timeout=timedelta(minutes=5),
        )

        visual_fields = await workflow.execute_activity(
            extract_visual_fields,
            args=(video_id, proxy_path, telemetry_windows),
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=GPU_ACTIVITY_RETRY,
            heartbeat_timeout=timedelta(minutes=5),
        )

        captions = await workflow.execute_activity(
            generate_captions,
            args=(video_id, proxy_path, telemetry_windows),
            start_to_close_timeout=timedelta(hours=1),
            retry_policy=GPU_ACTIVITY_RETRY,
        )

        # TODO: tek satır ClickHouse `clips` yazımı - ayrı bir "write_clips"
        # aktivitesi olarak eklenmeli (bkz. schema/clickhouse_clips.sql).
