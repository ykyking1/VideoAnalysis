"""LLM ayrıştırma: Qwen 14B + xgrammar (şema zorlamalı yapısal çıktı), SGLang
üzerinde (proje-ozeti.md §3.2 madde 1) - production hedefi.

Yerel testte (4GB VRAM) bunun yerine Ollama üzerinden küçük, quantize bir model
(varsayılan `qwen2.5:3b`) kullanılıyor. xgrammar'ın işlevsel karşılığı olarak
Ollama'nın JSON-schema zorlamalı `format` parametresi kullanılıyor - CFG/grammar
düzeyinde değil ama şema zorlaması aynı amaca hizmet ediyor. Katalogda karşılığı
olmayan kavramlar semantic_text'e düşer (model prompt'ta buna yönlendiriliyor).
"""
import json
from dataclasses import dataclass, field

import requests

from common import config


@dataclass
class StructuredFilters:
    sensor_type: str | None = None
    min_speed_kmh: float | None = None
    max_speed_kmh: float | None = None
    min_agl_m: float | None = None
    max_agl_m: float | None = None
    over_sea: bool | None = None
    is_sunset: bool | None = None  # sun_elevation eşiğinden türetilir
    is_night: bool | None = None
    min_vehicle_count: int | None = None


@dataclass
class ParsedQuery:
    filters: StructuredFilters = field(default_factory=StructuredFilters)
    semantic_text: str = ""


_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sensor_type": {"type": ["string", "null"]},
        "min_speed_kmh": {"type": ["number", "null"]},
        "max_speed_kmh": {"type": ["number", "null"]},
        "min_agl_m": {"type": ["number", "null"]},
        "max_agl_m": {"type": ["number", "null"]},
        "over_sea": {"type": ["boolean", "null"]},
        "is_sunset": {"type": ["boolean", "null"]},
        "is_night": {"type": ["boolean", "null"]},
        "min_vehicle_count": {"type": ["integer", "null"]},
        "semantic_text": {"type": "string"},
    },
    "required": ["semantic_text"],
}

_SYSTEM_PROMPT = """Sen bir İHA video arama sorgusu ayrıştırıcısısın. Kullanıcının \
doğal dil sorgusunu iki parçaya ayır:

1. Yapısal filtreler - SADECE şu alanlar için, sorguda açıkça belirtilmişse doldur:
   sensor_type, min_speed_kmh, max_speed_kmh, min_agl_m, max_agl_m, over_sea \
(deniz üzerinde mi), is_sunset (gün batımı), is_night (gece), min_vehicle_count.
   Belirtilmemiş alanları null bırak - ÖZELLİKLE over_sea/is_sunset/is_night için: \
sorgu bu konudan hiç bahsetmiyorsa false DEĞİL, null yaz. false SADECE sorgu açıkça \
tersini belirtiyorsa kullanılmalı (örn. "karada" -> over_sea=false, "gece değil" \
-> is_night=false). Konu hiç geçmiyorsa varsayılan olarak false yazma, null yaz.
2. semantic_text - yapısal alanlarla karşılığı olmayan, görsel/anlamsal geri kalan \
(örn. "TB2", "takip ediyor", "yüksek hızlarda uçan"). Yapısal alanlara taşınan \
bilgiyi semantic_text'te TEKRARLAMA.

Sadece JSON döndür, başka metin ekleme."""


def parse_query(raw_query: str) -> ParsedQuery:
    """Doğal dil sorgusunu Ollama (varsayılan qwen2.5:3b) ile şemaya zorlanmış
    yapısal filtrelere ve semantik metin artığına ayrıştırır."""
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_PARSE_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw_query},
            ],
            "format": _JSON_SCHEMA,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    parsed = json.loads(resp.json()["message"]["content"])

    semantic_text = parsed.pop("semantic_text", "")
    filters = StructuredFilters(**{k: v for k, v in parsed.items() if k in StructuredFilters.__dataclass_fields__})
    return ParsedQuery(filters=filters, semantic_text=semantic_text)
