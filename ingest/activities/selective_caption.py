"""Seçici caption: Qwen2.5-VL (vLLM üzerinde), sahne değişim skoruna göre
seçilen ~%10'luk "olay penceresi"ne kısa açıklama üretir
(proje-ozeti.md §3.1 madde 5).

NEDEN SEÇİCİ: Her pencereye caption üretmek VLM maliyetini ~10x artırır ve
monoton İHA görüntüsünde (örn. 20 dk boş deniz) neredeyse aynı cümleyi
tekrar üretir. Sahne değişimi yüksek pencereler "olay" içerme olasılığı en
yüksek olanlar - caption bütçesi oraya harcanır. Caption, Qdrant'ta tam-metin
indeksli bir payload alanı olarak saklanır ve hibrit (vektör + tam metin)
aramada kullanılır.

SAHNE SKORU: ffmpeg `select='gt(scene,...)'` filtresi yerine küçültülmüş
gri karelerin ardışık farkı kullanılıyor - aynı sıralamayı çok daha ucuz
üretiyor ve ffmpeg'i pencere başına yeniden çağırmayı gerektirmiyor.
"""
import cv2
import numpy as np
from temporalio import activity

from common import config
from common.llm import LLMError, chat_vision, encode_image_jpeg
from common.minio_client import download_temp
from ingest.activities.types import TelemetryWindow

CAPTION_PROMPT = (
    "This is a frame from an aerial drone surveillance video. "
    "Describe what is visible in one short, factual sentence. "
    "Focus on objects, vehicles, people and terrain. Do not speculate."
)
SCENE_SCORE_WIDTH = 64
SCENE_SCORE_HEIGHT = 36


def scene_change_score(video_path: str, t_start: float, t_end: float,
                        max_samples: int = 16) -> float:
    """Pencere içindeki en büyük ardışık kare farkı.

    max_samples ile sınırlı: 60sn'lik bir pencerede 30fps'te 1800 kare var,
    hepsini okumak gereksiz - eşit aralıklı örneklem aynı sıralamayı veriyor."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    prev = None
    max_diff = 0.0
    for t in np.linspace(t_start, max(t_end - 1e-3, t_start), max_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(
            cv2.resize(frame, (SCENE_SCORE_WIDTH, SCENE_SCORE_HEIGHT)), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        if prev is not None:
            max_diff = max(max_diff, float(np.abs(gray - prev).mean()))
        prev = gray
    cap.release()
    return max_diff


def _middle_frame_jpeg(video_path: str, t_start: float, t_end: float) -> bytes | None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((t_start + t_end) / 2 * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    encoded, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if encoded else None


def caption_frame(frame_jpeg: bytes) -> str:
    return chat_vision(CAPTION_PROMPT, [encode_image_jpeg(frame_jpeg)], max_tokens=64)


def select_event_windows(video_path: str, windows: list[TelemetryWindow],
                          fraction: float) -> set[str]:
    """En yüksek sahne-değişim skoruna sahip ~fraction'lık pencerelerin
    anahtar kümesini döner."""
    scored = [(scene_change_score(video_path, w.t_start, w.t_end), w.key) for w in windows]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    n_event = max(1, int(len(windows) * fraction))
    return {key for _, key in scored[:n_event]}


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
async def generate_captions(video_id: str, proxy_path: str,
                             windows: list[TelemetryWindow]) -> dict[str, str]:
    """Olay pencerelerine caption üretir. Anahtar: TelemetryWindow.key.

    VLM çağrısı başarısız olursa (sunucu kapalı, OOM) o pencere boş caption
    ile geçilir - caption opsiyonel bir zenginleştirme olduğu için tüm
    ingest'i düşürmesi doğru olmaz."""
    if not windows or not config.CAPTION_ENABLED:
        return {}

    captions: dict[str, str] = {}
    failures = 0

    with download_temp(config.MINIO_BUCKET_PROXY, proxy_path) as local_path:
        event_keys = select_event_windows(local_path, windows, config.CAPTION_WINDOW_FRACTION)
        _log(f"{video_id}: {len(event_keys)}/{len(windows)} olay penceresi secildi")

        for i, w in enumerate(windows):
            if w.key not in event_keys:
                continue
            frame = _middle_frame_jpeg(local_path, w.t_start, w.t_end)
            if frame is None:
                continue
            try:
                captions[w.key] = caption_frame(frame)
            except LLMError as exc:
                failures += 1
                if failures <= 3:
                    _log(f"{video_id}: caption basarisiz ({w.key}): {exc}")
            _heartbeat(f"{i}/{len(windows)} pencere")

    if failures:
        _log(f"{video_id}: {failures} caption uretilemedi (bos gecildi)")
    return captions
