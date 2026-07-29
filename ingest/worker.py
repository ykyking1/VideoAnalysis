"""Temporal worker: workflow ve aktiviteleri çalıştıran süreç.

ÖLÇEKLEME: Bu süreç yatay ölçeklenir - N makinede N worker çalıştırın, Temporal
işi aralarında dağıtır. proje-ozeti.md §8'deki GPU bütçesi varsayımı (~40x
gerçek-zaman) doğrulanmadı ve ölçtüğümüz tek-akış embedding hızı bunun çok
altında; teyit edilen envanterde (~300.000 video × 3-5sa) tek GPU'nun
yetmeyeceği açık. Worker sayısı = GPU sayısı olacak şekilde planlayın.

max_concurrent_activities=1 (varsayılan): tek GPU'lu bir worker'da aynı anda
iki GPU aktivitesi çalıştırmak VRAM'i bölüp OOM'a yol açar. Çok GPU'lu
makinede MAX_CONCURRENT_ACTIVITIES'i yükseltin.

Kullanım:
    python -m ingest.worker
"""
import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

from common import config
from ingest.activities import ALL_ACTIVITIES
from ingest.workflow import VideoIngestWorkflow

MAX_CONCURRENT_ACTIVITIES = int(os.environ.get("MAX_CONCURRENT_ACTIVITIES", "1"))


def _warn_if_no_gpu(log: logging.Logger) -> None:
    """CPU-only torch sessiz bir felakettir: hicbir sey cokmez, ingest ~15-20
    kat yavas kosar. 300.000 videoluk bir arsivde bu aylar demek. Worker
    baslarken bunu bagirarak soylemek, saatler sonra fark etmekten iyidir.
    Tam kontrol: python -m scripts.check_env"""
    try:
        import torch
    except ImportError:
        log.error("torch kurulu degil - embedding aktivitesi calismayacak")
        return

    if torch.cuda.is_available():
        log.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1024**3)
        return

    log.warning("=" * 62)
    log.warning("GPU KULLANILMIYOR - torch %s", torch.__version__)
    log.warning("Ingest calisir ama ~15-20 kat yavas olur.")
    log.warning("Cozum: pip install torch torchvision --force-reinstall \\")
    log.warning("         --index-url https://download.pytorch.org/whl/cu126")
    log.warning("Tam kontrol: python -m scripts.check_env")
    log.warning("=" * 62)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("ingest.worker")
    _warn_if_no_gpu(log)

    client = await Client.connect(config.TEMPORAL_HOST, namespace=config.TEMPORAL_NAMESPACE)
    log.info("Temporal'a baglanildi: %s (namespace=%s, queue=%s)",
             config.TEMPORAL_HOST, config.TEMPORAL_NAMESPACE, config.TEMPORAL_TASK_QUEUE)

    worker = Worker(
        client,
        task_queue=config.TEMPORAL_TASK_QUEUE,
        workflows=[VideoIngestWorkflow],
        activities=ALL_ACTIVITIES,
        max_concurrent_activities=MAX_CONCURRENT_ACTIVITIES,
    )
    log.info("Worker basladi (max_concurrent_activities=%d). Ctrl+C ile durdurun.",
             MAX_CONCURRENT_ACTIVITIES)
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
