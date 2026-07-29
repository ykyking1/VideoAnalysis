"""Uçtan uca sorgu hattı: doğal dil -> video kimliği + zaman aralığı
(proje-ozeti.md §3.2, projenin asıl çıktısı).

    ayrıştır (LLM)  ->  hibrit ara (Qdrant)  ->  aralık birleştir  ->  [rerank]

Bu modül tüm sorgu tüketicilerinin (CLI, servis, değerlendirme scriptleri)
ortak giriş noktası - adımların sırası ve varsayılanları tek yerde.
"""
import time
from dataclasses import dataclass, field

from common import config
from query.hybrid_search import search
from query.interval_merge import Interval, merge_matches
from query.llm_parser import ParsedQuery, parse_query
from query.rerank import rerank


@dataclass
class QueryResponse:
    query: str
    intervals: list[Interval] = field(default_factory=list)
    parsed: ParsedQuery | None = None
    relaxed_fields: list[str] = field(default_factory=list)
    filter_description: str = ""
    elapsed_ms: float = 0.0
    reranked: bool = False

    @property
    def was_relaxed(self) -> bool:
        return bool(self.relaxed_fields)


def run_query(raw_query: str, top_k: int | None = None,
               enable_rerank: bool | None = None) -> QueryResponse:
    """Doğal dil sorgusunu çalıştırır ve birleştirilmiş zaman aralıklarını döner."""
    started = time.perf_counter()

    parsed = parse_query(raw_query)
    result = search(parsed, top_k=top_k)
    intervals = merge_matches(result.matches)

    should_rerank = config.RERANK_ENABLED if enable_rerank is None else enable_rerank
    if should_rerank and intervals:
        intervals = rerank(parsed.semantic_text or raw_query, intervals)

    return QueryResponse(
        query=raw_query,
        intervals=intervals,
        parsed=parsed,
        relaxed_fields=result.relaxed_fields,
        filter_description=result.filter_description,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        reranked=bool(should_rerank and intervals),
    )
