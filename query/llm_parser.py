"""LLM ayrıştırma: Qwen 14B + xgrammar (şema zorlamalı yapısal çıktı), SGLang
üzerinde (proje-ozeti.md §3.2 madde 1). Sorguyu yapısal filtrelere (telemetriden
gelen, deterministik) ve semantik artığa (semantic_text) ayırır. Katalogda
karşılığı olmayan kavramlar hataya değil, tamamen semantik aramaya düşer.
"""
from dataclasses import dataclass, field


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


def parse_query(raw_query: str) -> ParsedQuery:
    """Doğal dil sorgusunu Qwen 14B + xgrammar ile şemaya zorlanmış yapısal
    filtrelere ve semantik metin artığına ayrıştırır."""
    raise NotImplementedError
