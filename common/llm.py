"""vLLM (OpenAI-uyumlu) istemcisi: şema-zorlamalı metin ve görüntülü sohbet.

İki tüketici var:
- query/llm_parser.py  -> sorguyu yapısal filtre + semantik metne ayırır
                          (proje-ozeti.md §3.2 madde 1)
- ingest/.../selective_caption.py ve query/rerank.py -> görüntülü VLM çağrıları
                          (§3.1 madde 5, §3.2 madde 4)

ŞEMA ZORLAMA: vLLM'in guided decoding'i (xgrammar backend) `extra_body` içinde
`guided_json` ile veriliyor. Bu, proje-ozeti.md §3.2'deki "xgrammar ile şema
zorlamalı yapısal çıktı" gereksinimini karşılıyor - modelin şema dışına
çıkması gramer düzeyinde engelleniyor, "JSON döndür" ricası değil.

MODEL SEÇİMİ: §3.2 Qwen 14B öngörüyor. Tek 4060 sınıfı GPU'da 14B pratik
değil; varsayılan 7B-AWQ. PARSE_MODEL/VLM_MODEL env'leriyle değiştirilebilir -
daha büyük GPU'da 14B'ye geçmek yalnızca env değişikliği.
"""
import base64

import requests

from common import config


class LLMError(RuntimeError):
    pass


def _post(base_url: str, payload: dict, timeout: int) -> dict:
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.VLLM_API_KEY}"},
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise LLMError(f"vLLM {response.status_code}: {response.text[:500]}")
    data = response.json()
    if not data.get("choices"):
        raise LLMError(f"vLLM bos yanit dondurdu: {str(data)[:500]}")
    return data


def chat_json(system_prompt: str, user_prompt: str, json_schema: dict,
               model: str | None = None, base_url: str | None = None,
               temperature: float = 0.0) -> str:
    """Şemaya zorlanmış JSON yanıtı (ham string olarak) döner."""
    payload = {
        "model": model or config.PARSE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "extra_body": {"guided_json": json_schema, "guided_decoding_backend": "xgrammar"},
    }
    data = _post(base_url or config.VLLM_BASE_URL, payload, config.LLM_TIMEOUT_S)
    return data["choices"][0]["message"]["content"]


def encode_image_jpeg(image_bytes: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"


def chat_vision(prompt: str, image_data_urls: list[str], model: str | None = None,
                 base_url: str | None = None, max_tokens: int = 128,
                 temperature: float = 0.0) -> str:
    """Bir veya daha fazla görüntüyle VLM çağrısı; düz metin yanıt döner."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in image_data_urls]

    payload = {
        "model": model or config.VLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = _post(base_url or config.VLM_BASE_URL, payload, config.LLM_TIMEOUT_S)
    return (data["choices"][0]["message"]["content"] or "").strip()


def health_check(base_url: str | None = None) -> bool:
    try:
        response = requests.get(f"{(base_url or config.VLLM_BASE_URL).rstrip('/')}/models",
                                 headers={"Authorization": f"Bearer {config.VLLM_API_KEY}"},
                                 timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False
