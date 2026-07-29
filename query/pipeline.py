"""Uçtan uca sorgu hattı: doğal dil -> video kimliği + zaman aralığı
(proje-ozeti.md §3.2, projenin asıl çıktısı).

    ayrıştır (vLLM)  ->  hibrit ara (Qdrant)  ->  aralık birleştir  ->  [rerank]

GECİKME KIRILIMI: Her aşamanın süresi `QueryResponse.timings` içinde ayrı ayrı
raporlanır. Bunun sebebi proje-ozeti.md §8'de gecikme rakamlarının (300ms,
rerank 3-15dk) **hiç ölçülmemiş** olması - "vLLM ayrıştırma sistemi ne kadar
yavaşlatıyor" sorusu ancak gerçek çalıştırmada bu kırılıma bakılarak
cevaplanabilir, tahminle değil.

Aşamaların doğası:
- `parse_ms`  : sorgu başına TEK vLLM çağrısı, kısa JSON çıktı (guided
                decoding şemayı zorluyor, model uzun uzun yazamıyor)
- `embed_ms`  : sorgu metninin vektöre çevrilmesi (Qwen3-VL, GPU'da)
- `qdrant_ms` : filtreli HNSW araması; gevşetme tetiklenirse merdivenin
                her adımı buraya eklenir (`ladder_steps`)
- `rerank_ms` : aday başına bir VLM çağrısı - varsayılan KAPALI, açılırsa
                baskın maliyet olması beklenir
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
    timings: dict[str, float] = field(default_factory=dict)
    ladder_steps: int = 1

    @property
    def was_relaxed(self) -> bool:
        return bool(self.relaxed_fields)

    def timing_summary(self) -> str:
        """'parse=812ms embed=41ms qdrant=18ms merge=0ms' gibi tek satir."""
        return " ".join(f"{k}={v:.0f}ms" for k, v in self.timings.items())


def run_query(raw_query: str, top_k: int | None = None,
               enable_rerank: bool | None = None) -> QueryResponse:
    """Doğal dil sorgusunu çalıştırır ve birleştirilmiş zaman aralıklarını döner."""
    started = time.perf_counter()
    timings: dict[str, float] = {}

    t = time.perf_counter()
    parsed = parse_query(raw_query)
    timings["parse"] = (time.perf_counter() - t) * 1000

    result = search(parsed, top_k=top_k)
    timings["embed"] = result.embed_ms
    timings["qdrant"] = result.qdrant_ms

    t = time.perf_counter()
    intervals = merge_matches(result.matches)
    timings["merge"] = (time.perf_counter() - t) * 1000

    should_rerank = config.RERANK_ENABLED if enable_rerank is None else enable_rerank
    if should_rerank and intervals:
        t = time.perf_counter()
        intervals = rerank(parsed.semantic_text or raw_query, intervals)
        timings["rerank"] = (time.perf_counter() - t) * 1000

    return QueryResponse(
        query=raw_query,
        intervals=intervals,
        parsed=parsed,
        relaxed_fields=result.relaxed_fields,
        filter_description=result.filter_description,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        reranked=bool(should_rerank and intervals),
        timings=timings,
        ladder_steps=result.ladder_steps,
    )
