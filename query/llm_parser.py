"""LLM ayrıştırma: sorguyu yapısal filtrelere + semantik artığa ayırır
(proje-ozeti.md §3.2 madde 1).

vLLM üzerinde xgrammar guided decoding ile şema zorlaması yapılır - model
şemanın dışına gramer düzeyinde çıkamaz. Katalogda karşılığı olmayan kavramlar
hataya değil, `semantic_text`e düşer ve embedding modeline gider.

KRİTİK TASARIM NOKTASI: Belirtilmemiş alanlar `false` DEĞİL `null` olmalı.
"deniz üzerinde tekne" sorgusunda `is_night=false` yazılırsa gece çekilmiş
tüm klipler yapısal olarak elenir - ve bu oturumda ölçtük ki dar bir filtre
doğru cevabı gerçekten kaybettiriyor (gerçek irtifa testinde 21 sorgunun
17'si). Bu yüzden hem sistem promptu hem `_sanitize` bunu iki kez kontrol
ediyor: LLM yine de `false` üretirse ve sorguda ilgili kelime geçmiyorsa
`None`'a çevriliyor.
"""
import json
from dataclasses import asdict, dataclass, field

from common import config
from common.llm import chat_json

# Sorguda bu kelimelerden hicbiri gecmiyorsa ilgili bool alan null'a cekilir.
_NEGATION_GUARDS: dict[str, tuple[str, ...]] = {
    "over_sea": ("deniz", "sea", "ocean", "kıyı", "kiyi", "coast", "su üstü", "karada", "land"),
    "is_night": ("gece", "night", "karanlık", "karanlik", "dark", "gündüz", "gunduz", "day"),
    "is_sunset": ("gün batımı", "gun batimi", "günbatımı", "gunbatimi", "sunset",
                   "gün doğumu", "gun dogumu", "sunrise", "alacakaranlık", "twilight"),
}


@dataclass
class StructuredFilters:
    sensor_type: str | None = None
    min_speed_kmh: float | None = None
    max_speed_kmh: float | None = None
    min_agl_m: float | None = None
    max_agl_m: float | None = None
    over_sea: bool | None = None
    is_sunset: bool | None = None  # sun_elevation esiginden turetilir
    is_night: bool | None = None
    min_vehicle_count: int | None = None

    def active_fields(self) -> list[str]:
        return [k for k, v in asdict(self).items() if v is not None]

    def is_empty(self) -> bool:
        return not self.active_fields()


@dataclass
class ParsedQuery:
    filters: StructuredFilters = field(default_factory=StructuredFilters)
    semantic_text: str = ""
    raw_query: str = ""


JSON_SCHEMA = {
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
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Sen bir İHA video arama sorgusu ayrıştırıcısısın. Kullanıcının \
doğal dil sorgusunu iki parçaya ayır:

1. Yapısal filtreler - SADECE şu alanlar için, sorguda AÇIKÇA belirtilmişse doldur:
   sensor_type, min_speed_kmh, max_speed_kmh, min_agl_m, max_agl_m,
   over_sea (deniz üzerinde mi), is_sunset (gün batımı), is_night (gece),
   min_vehicle_count.

   EN ÖNEMLİ KURAL: Sorguda hiç bahsedilmeyen alanı null bırak - ASLA false yazma.
   false SADECE kullanıcı açıkça tersini istiyorsa kullanılır:
     "karada uçan"      -> over_sea=false
     "gece olmayan"     -> is_night=false
     "deniz üstünde"    -> over_sea=true
   Sorgu konudan hiç bahsetmiyorsa (örn. "tekne ara") o alan null kalmalı.
   Gereksiz filtre, doğru sonuçların elenmesine yol açar.

2. semantic_text - yapısal alanlarda karşılığı olmayan görsel/anlamsal geri kalan
   (örn. "TB2", "takip ediyor", "dalgalı deniz"). Yapısal alanlara taşıdığın
   bilgiyi semantic_text içinde TEKRARLAMA.

Sadece JSON döndür."""


def _sanitize(parsed: dict, raw_query: str) -> dict:
    """LLM şemaya uysa da anlamsal olarak hatalı `false` üretebilir - sorguda
    ilgili kavram hiç geçmiyorsa bool alanları null'a çeker."""
    lowered = raw_query.casefold()
    for field_name, keywords in _NEGATION_GUARDS.items():
        if parsed.get(field_name) is False and not any(k in lowered for k in keywords):
            parsed[field_name] = None
    return parsed


def parse_query(raw_query: str) -> ParsedQuery:
    """Doğal dil sorgusunu şemaya zorlanmış yapısal filtrelere ve semantik
    metin artığına ayrıştırır.

    LLM erişilemezse sorgunun tamamı semantic_text olarak geçer - arama
    çalışmaya devam eder, sadece yapısal filtreler devre dışı kalır."""
    try:
        content = chat_json(SYSTEM_PROMPT, raw_query, JSON_SCHEMA, model=config.PARSE_MODEL)
        parsed = json.loads(content)
    except Exception as exc:  # noqa: BLE001 - ayristirma hatasi aramayi durdurmamali
        return ParsedQuery(semantic_text=raw_query, raw_query=raw_query,
                           filters=StructuredFilters())

    parsed = _sanitize(parsed, raw_query)
    semantic_text = (parsed.pop("semantic_text", "") or "").strip()
    known = {k: v for k, v in parsed.items() if k in StructuredFilters.__dataclass_fields__}

    return ParsedQuery(
        filters=StructuredFilters(**known),
        semantic_text=semantic_text or raw_query,
        raw_query=raw_query,
    )
