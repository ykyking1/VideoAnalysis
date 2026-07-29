"""Opsiyonel rerank: VLM top adayları doğrular (proje-ozeti.md §3.2 madde 4).

NEDEN: Embedding modeli görsel olarak benzer ama anlamca yanlış klipleri
üste taşıyabiliyor (klasik örnek: "gün batımı" vs "gün doğumu" - neredeyse
aynı görüntü, zıt kavram). VLM adayın kareyi görüp sorguya gerçekten uyup
uymadığına karar verir.

MALİYET: Aday başına bir VLM çağrısı. RERANK_CANDIDATES adayla sınırlı ve
varsayılan olarak KAPALI (RERANK_ENABLED=false) - proje-ozeti.md §3.2 bunu
zaten "opsiyonel" olarak işaretliyor ve gecikmeye ciddi etkisi var. §8'deki
gecikme rakamları (3-15dk / 8-10sn) hiç ölçülmedi.

DAYANIKLILIK: VLM erişilemezse ya da bir aday için hata dönerse o aday
orijinal skoruyla kalır - rerank aramanın kalitesini artırmalı, tek hata
aramayı düşürmemeli.
"""
from dataclasses import replace

import cv2

from common import config
from common.llm import LLMError, chat_vision, encode_image_jpeg
from common.minio_client import download_temp

VERDICT_PROMPT = (
    "You are checking whether an aerial drone video frame matches a search query.\n"
    "Query: {query}\n"
    "Answer with a single number from 0 to 10: how well does this frame match "
    "the query? 0 means not at all, 10 means a perfect match. "
    "Reply with the number only."
)


def _extract_frame_jpeg(video_path: str, t_start: float, t_end: float) -> bytes | None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((t_start + t_end) / 2 * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    encoded, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if encoded else None


def _parse_score(text: str) -> float | None:
    """VLM yanıtından 0-10 arası skoru çıkarır; ayrıştırılamazsa None."""
    for token in text.replace(",", ".").split():
        try:
            value = float(token.strip(".:/"))
        except ValueError:
            continue
        if 0.0 <= value <= 10.0:
            return value / 10.0
    return None


def rerank(query_text: str, intervals: list, top_n: int | None = None) -> list:
    """Adayları VLM ile doğrulayıp yeniden sıralar.

    Sadece ilk `top_n` aday reranklanır; geri kalanı orijinal sırada altta
    kalır (VLM maliyeti aday sayısıyla doğrusal artıyor)."""
    if not config.RERANK_ENABLED or not intervals:
        return intervals

    top_n = top_n or config.RERANK_CANDIDATES
    head, tail = intervals[:top_n], intervals[top_n:]
    prompt = VERDICT_PROMPT.format(query=query_text)

    rescored = []
    for interval in head:
        vlm_score = _score_interval(prompt, interval)
        if vlm_score is None:
            rescored.append(interval)
        else:
            # Vektor skoru ile VLM skorunu harmanla: VLM tek kareye bakiyor,
            # vektor tum pencereye - ikisini birden dikkate almak tek bir
            # yanlis kare yuzunden iyi bir adayi dusurmeyi onluyor.
            blended = 0.5 * interval.score + 0.5 * vlm_score
            rescored.append(replace(interval, score=blended))

    rescored.sort(key=lambda i: i.score, reverse=True)
    return rescored + tail


def _score_interval(prompt: str, interval) -> float | None:
    proxy_key = f"{interval.video_id}/proxy.mp4"
    try:
        with download_temp(config.MINIO_BUCKET_PROXY, proxy_key) as local_path:
            frame = _extract_frame_jpeg(local_path, interval.t_start, interval.t_end)
        if frame is None:
            return None
        answer = chat_vision(prompt, [encode_image_jpeg(frame)], max_tokens=8)
        return _parse_score(answer)
    except (LLMError, OSError):
        return None
