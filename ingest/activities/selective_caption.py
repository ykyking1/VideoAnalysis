"""Seçici caption: Qwen2.5-VL (vLLM üzerinde), ffmpeg sahne değişim skoruna göre
seçilen ~%10'luk "olay penceresi"ne kısa açıklama üretir (proje-ozeti.md §3.1
madde 5).

Yerel testte Qwen2.5-VL+vLLM (4GB VRAM'e sığmıyor) yerine Ollama üzerinden
`moondream` (küçük, quantize vision-language modeli) kullanılıyor. Sahne değişim
skoru için ffmpeg'in `select='gt(scene,...)'` filtresi yerine, basit ardışık kare
farkı (frame diff) ısı skoru kullanıldı - aynı fikri (en çok değişen ~%10 pencere)
çok daha az bağımlılıkla uyguluyor.
"""
import base64

import cv2
import numpy as np
import requests
from temporalio import activity

from common import config
from common.minio_client import download_temp
from ingest.activities.types import TelemetryWindow

EVENT_WINDOW_FRACTION = 0.10


def _middle_frame(video_path: str, t_start: float, t_end: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((t_start + t_end) / 2 * fps))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _scene_change_score(video_path: str, t_start: float, t_end: float) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_start * fps))
    prev = None
    max_diff = 0.0
    end_frame = int(t_end * fps)
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            max_diff = max(max_diff, float(np.abs(gray - prev).mean()))
        prev = gray
    cap.release()
    return max_diff


CAPTION_PROMPTS = [
    "Briefly caption this image in one sentence.",
    "Briefly describe this image.",
]


def _caption_frame(frame: np.ndarray) -> str:
    # Ölçüldü: moondream, "Describe this drone camera frame in one short
    # sentence." promptunda tutarlı biçimde tek adımda dur-tokenı üretip boş yanıt
    # döndürüyor (eval_count=1) - "drone"/ifade kalıbına özgü bir tıkanma, hata
    # fırlatmıyor. Farklı ifadeler (aynı frame'de) düzgün çalışıyor - birden fazla
    # prompt deneyip ilk boş-olmayan yanıtı kullanıyoruz.
    ok, buf = cv2.imencode(".jpg", frame)
    image_b64 = base64.b64encode(buf.tobytes()).decode()
    for prompt in CAPTION_PROMPTS:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
                "model": config.OLLAMA_CAPTION_MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        caption = resp.json()["response"].strip()
        if caption:
            return caption
    return ""


@activity.defn
async def generate_captions(
    video_id: str, proxy_path: str, windows: list[TelemetryWindow]
) -> dict[str, str]:
    """En yüksek sahne-değişim skoruna sahip ~%10 pencereyi seçip caption üretir;
    diğerleri için boş string döner. Anahtar `f"{t_start}:{t_end}"`."""
    if not windows:
        return {}

    with download_temp(config.MINIO_BUCKET_PROXY, proxy_path) as local_path:
        scores = [(_scene_change_score(local_path, w.t_start, w.t_end), w) for w in windows]
        scores.sort(key=lambda pair: pair[0], reverse=True)
        n_event = max(1, int(len(windows) * EVENT_WINDOW_FRACTION))
        event_windows = {id(w) for _, w in scores[:n_event]}

        captions: dict[str, str] = {}
        for w in windows:
            key = f"{w.t_start}:{w.t_end}"
            if id(w) in event_windows:
                frame = _middle_frame(local_path, w.t_start, w.t_end)
                captions[key] = _caption_frame(frame) if frame is not None else ""
            else:
                captions[key] = ""
    return captions
