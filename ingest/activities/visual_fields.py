"""Terfi etmiş görsel alanlar: YOLO26 ile sık talep gören ve deterministik
çözülebilen görsel kavramları kolonlaştırır (proje-ozeti.md §3.1 madde 4).

NEDEN AYRI BİR ALAN: "kaç araç var" gibi sayısal bir kavramı embedding'e
sormak güvenilmez; deterministik bir detektörle sayıp kolonlaştırmak hem
kesin hem de yapısal filtreye (WHERE vehicle_count >= 3) açık. Katalog
sabit değil - sorgu loglarında sık geçen ve deterministik çözülebilen
kavramlar zamanla buraya terfi ettirilir.

YOLO26: Ultralytics, Ocak 2026'da yayınlandı; `ultralytics` paketiyle n/s/m/l/x
varyantlarında kullanılabiliyor. Ağırlık dosyası ilk çalıştırmada otomatik
indirilir.

IR FINE-TUNE: proje-ozeti.md §3.1 IR (termal) için fine-tune edilmiş bir
YOLO26 öngörüyor - bu kurumsal eğitim gerektiriyor ve HENÜZ YOK. Varsayılan
COCO ön-eğitimli ağırlıklar RGB'de çalışır; termal görüntüde başarımı
ölçülmedi. YOLO_MODEL env'i ile kendi fine-tune ağırlığınıza geçebilirsiniz.
"""
import cv2
import numpy as np
from temporalio import activity

from common import config
from common.minio_client import download_temp
from ingest.activities.types import TelemetryWindow, VisualFields

# COCO sinif adlari uzerinden "arac benzeri" kume. Kendi fine-tune modelinizde
# sinif adlari farkliysa burasi da guncellenmeli.
VEHICLE_LIKE_CLASSES = {"boat", "car", "truck", "bus", "train", "airplane", "motorcycle"}

_model = None


def _get_model():
    """ultralytics tembel import ediliyor - sorgu-only dagitimda YOLO
    kurulu olmak zorunda degil (bkz. ingest/activities/__init__.py)."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        _log(f"YOLO modeli yukleniyor: {config.YOLO_MODEL}")
        _model = YOLO(config.YOLO_MODEL)
    return _model


def unload_model() -> None:
    global _model
    _model = None
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    """Pencere içindeki EN YÜKSEK eş-zamanlı araç sayısı.

    Toplam değil maksimum: aynı tekne 3 karede de görünüyorsa 3 değil 1
    sayılmalı. Kare-başına sayıp maksimumu almak, takip (tracking) kurmadan
    bu soruna yeterince iyi bir yaklaşım."""
    model = _get_model()
    frames = _sample_frames(video_path, t_start, t_end, config.YOLO_SAMPLE_FRAMES)
    if not frames:
        return 0

    max_count = 0
    for result in model.predict(frames, verbose=False, conf=config.YOLO_CONF):
        names = result.names
        count = sum(1 for c in result.boxes.cls.tolist()
                    if names[int(c)] in VEHICLE_LIKE_CLASSES)
        max_count = max(max_count, count)
    return max_count


def _log(message: str) -> None:
    try:
        activity.logger.info(message)
    except RuntimeError:
        print(message)


def _heartbeat(message: str) -> None:
    try:
        activity.heartbeat(message)
    except RuntimeError:
        pass


@activity.defn
async def extract_visual_fields(video_id: str, proxy_path: str,
                                 windows: list[TelemetryWindow]) -> list[VisualFields]:
    """Her pencere için görsel alanları hesaplar."""
    if not windows:
        return []
    with download_temp(config.MINIO_BUCKET_PROXY, proxy_path) as local_path:
        fields = []
        for i, w in enumerate(windows):
            fields.append(VisualFields(vehicle_count=count_vehicles(local_path, w.t_start, w.t_end)))
            if i % 50 == 0:
                _heartbeat(f"{i}/{len(windows)} pencere")
        return fields
