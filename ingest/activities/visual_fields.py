"""Terfi etmiş görsel alanlar: YOLO26 (IR fine-tune), sorgu loglarından sık talep
gören ve deterministik çözülebilen görsel kavramları kolonlaştırır (proje-ozeti.md
§3.1 madde 4).

YOLO26 IR fine-tune modeli mevcut değil (kurumsal eğitim gerektiriyor). Bu yerel
testte genel amaçlı, COCO ön-eğitimli `yolov8n` kullanılıyor; SeaDronesSee deniz
sahnesiyle örtüşmesi için "boat" sınıfı `vehicle_count` alanına sayılıyor. Bu bir
mekanik doğrulama yer tutucusudur, gerçek kategori kataloğu §3.1 madde 4'teki gibi
kullanım verisine göre ayrıca kurulmalı.
"""
import cv2
import numpy as np
from temporalio import activity
from ultralytics import YOLO

from common import config
from common.minio_client import download_temp
from ingest.activities.types import TelemetryWindow

MODEL_NAME = "yolov8n.pt"
VEHICLE_LIKE_CLASSES = {"boat", "car", "truck", "bus"}
SAMPLE_FRAMES_PER_WINDOW = 3

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_NAME)
    return _model


def _sample_frames(video_path: str, t_start: float, t_end: float, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    for t in np.linspace(t_start, max(t_end - 1e-3, t_start), n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def count_vehicles(video_path: str, t_start: float, t_end: float) -> int:
    model = _get_model()
    frames = _sample_frames(video_path, t_start, t_end, SAMPLE_FRAMES_PER_WINDOW)
    if not frames:
        return 0
    max_count = 0
    for result in model.predict(frames, verbose=False):
        names = result.names
        count = sum(1 for c in result.boxes.cls.tolist() if names[int(c)] in VEHICLE_LIKE_CLASSES)
        max_count = max(max_count, count)
    return max_count


@activity.defn
async def extract_visual_fields(
    video_id: str, proxy_path: str, windows: list[TelemetryWindow]
) -> list[dict]:
    """Her pencere için YOLO çıktısından vehicle_count hesaplar."""
    with download_temp(config.MINIO_BUCKET_PROXY, proxy_path) as local_path:
        return [
            {"vehicle_count": count_vehicles(local_path, w.t_start, w.t_end)}
            for w in windows
        ]
