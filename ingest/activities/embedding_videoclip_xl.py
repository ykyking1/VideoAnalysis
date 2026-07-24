"""VideoCLIP-XL (proje-ozeti.md §5, "yeni aday") embedding sarmalayıcısı -
clip_embedding.py ile aynı embed_window/embed_text arayüzü, model karşılaştırması
için (bkz. poc/compare_embedding_models.py).

alibaba-pai/VideoCLIP-XL, ViT-L/14 tabanlı, 768d ortak uzay. Ağırlıklar
models/VideoCLIP-XL/VideoCLIP-XL.bin'den yükleniyor (huggingface_hub ile
indirildi, internet gerektirmeden tamamen offline çalışıyor - vision/text
encoder'lar pretrained=False ile kuruluyor, tüm ağırlıklar state_dict'ten geliyor).

Not: models/VideoCLIP-XL/utils/text_encoder/text_encoder.py'de bir satır
(`from pkg_resources import packaging` -> `import packaging.version`) yeni
setuptools sürümlerinin pkg_resources'ı kaldırmasından dolayı manuel düzeltildi.

Ölçüldü (GT1030 4GB): model yükleme sonrası ~3.6GB GPU belleği (X-CLIP-base-
patch32'den çok daha ağır - sınırda ama sığıyor), video embedding ~9s/pencere.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "VideoCLIP-XL"
NUM_FRAMES = 8

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_text_encoder = None

_V_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
_V_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def _get_model():
    global _model, _text_encoder
    if _model is None:
        from modeling import VideoCLIP_XL
        from utils.text_encoder import text_encoder

        model = VideoCLIP_XL()
        state_dict = torch.load(str(MODEL_DIR / "VideoCLIP-XL.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(_device).eval()

        _model = model
        _text_encoder = text_encoder
    return _model, _text_encoder


def unload_model() -> None:
    """clip_embedding.unload_model ile aynı amaç: GT1030 4GB'ta Ollama ile
    çakışmayı önlemek için embedding sonrası GPU'yu boşaltır."""
    global _model, _text_encoder
    _model = None
    _text_encoder = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize(data: np.ndarray) -> np.ndarray:
    return (data / 255.0 - _V_MEAN) / _V_STD


def _sample_video_tensor(video_path: str, t_start: float, t_end: float, num_frames: int) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    timestamps = np.linspace(t_start, max(t_end - 1e-3, t_start), num_frames)
    frames = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
        frames.append(frame)
    cap.release()

    vid_tube = []
    for fr in frames:
        fr = fr[:, :, ::-1]  # BGR -> RGB
        fr = cv2.resize(fr, (224, 224))
        fr = np.expand_dims(_normalize(fr), axis=(0, 1))
        vid_tube.append(fr)
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    return torch.from_numpy(vid_tube).float()


@torch.no_grad()
def embed_window(video_path: str, t_start: float, t_end: float) -> list[float]:
    model, _ = _get_model()
    vid_tube = _sample_video_tensor(video_path, t_start, t_end, NUM_FRAMES).to(_device)
    video_features = model.vision_model.get_vid_features(vid_tube).float()
    video_features = video_features / video_features.norm(dim=-1, keepdim=True)
    return video_features[0].cpu().tolist()


@torch.no_grad()
def embed_text(text: str) -> list[float]:
    model, text_encoder = _get_model()
    text_inputs = text_encoder.tokenize([text], truncate=True).to(_device)
    text_features = model.text_model.encode_text(text_inputs).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0].cpu().tolist()
