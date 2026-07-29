"""Hibrit arama: yapısal filtre + semantik vektör, tek Qdrant sorgusunda
(proje-ozeti.md §3.2 madde 2).

Qdrant filtreyi HNSW graf gezinmesinin İÇİNDE uyguluyor - ClickHouse'taki
"prefilter (tam ama indekssiz/yavaş) vs postfilter (hızlı ama eksik)" ikilemi
burada yok. Ölçtük: 100K korpusta varsayılan filtreli-HNSW, `exact=True`
brute-force ile 21 sorgunun 21'inde birebir aynı top-3'ü verdi ve ~1.5x daha
hızlıydı. Detay: docs/worklog_2026-07-28.md.

FİLTRE GEVŞETME: Hard filtre sonucu SEARCH_MIN_RESULTS'ın altına düşerse
filtre kademeli gevşetilir (bkz. query/filter_builder.py). Gevşetilmiş
sonuçlar `exact_filter_match=False` ile işaretlenir - kullanıcı hangi
sonucun filtreye tam uyduğunu, hangisinin "yakın eşleşme" olduğunu görür.
"""
from dataclasses import dataclass, field

from qdrant_client.http import models as qm

from common import config
from common.qdrant_store import get_client, search as qdrant_search
from ingest.activities.clip_embedding import embed_text
from query.filter_builder import build_filter, describe, relaxation_ladder
from query.llm_parser import ParsedQuery


@dataclass(frozen=True)
class Match:
    video_id: str
    t_start: float
    t_end: float
    score: float = 0.0
    caption: str = ""
    exact_filter_match: bool = True


@dataclass
class SearchResult:
    matches: list[Match] = field(default_factory=list)
    relaxed_fields: list[str] = field(default_factory=list)
    filter_description: str = ""

    @property
    def was_relaxed(self) -> bool:
        return bool(self.relaxed_fields)


def _to_match(point: qm.ScoredPoint, exact: bool) -> Match:
    payload = point.payload or {}
    return Match(
        video_id=payload.get("video_id", ""),
        t_start=float(payload.get("t_start", 0.0)),
        t_end=float(payload.get("t_end", 0.0)),
        score=float(point.score),
        caption=payload.get("caption", "") or "",
        exact_filter_match=exact,
    )


def search(parsed: ParsedQuery, top_k: int | None = None,
            min_results: int | None = None) -> SearchResult:
    """Yapısal filtre + semantik vektör araması, gerekirse gevşetmeli.

    Semantic_text boşsa (sorgu tamamen yapısalsa) vektör araması yerine
    filtreye uyan kayıtlar zaman sırasıyla döner."""
    top_k = top_k or config.SEARCH_TOP_K
    min_results = min_results if min_results is not None else config.SEARCH_MIN_RESULTS

    client = get_client()
    collection = config.QDRANT_COLLECTION

    if not parsed.semantic_text.strip():
        return _structural_only_search(client, collection, parsed, top_k)

    query_vector = embed_text(parsed.semantic_text)
    ladder = relaxation_ladder(parsed.filters)

    seen: set[tuple[str, float]] = set()
    matches: list[Match] = []
    relaxed_fields: list[str] = []

    for filters, dropped in ladder:
        points = qdrant_search(client, collection, query_vector,
                                build_filter(filters), top_k)
        for point in points:
            match = _to_match(point, exact=not dropped)
            key = (match.video_id, match.t_start)
            if key in seen:
                continue
            seen.add(key)
            matches.append(match)

        if len(matches) >= min_results:
            relaxed_fields = dropped
            break
        relaxed_fields = dropped

    return SearchResult(
        matches=matches[:top_k],
        relaxed_fields=relaxed_fields,
        filter_description=describe(parsed.filters),
    )


def _structural_only_search(client, collection: str, parsed: ParsedQuery,
                             top_k: int) -> SearchResult:
    """Semantik metin yoksa saf yapısal tarama (vektör araması anlamsız)."""
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=build_filter(parsed.filters),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    matches = [
        Match(
            video_id=(p.payload or {}).get("video_id", ""),
            t_start=float((p.payload or {}).get("t_start", 0.0)),
            t_end=float((p.payload or {}).get("t_end", 0.0)),
            caption=(p.payload or {}).get("caption", "") or "",
        )
        for p in points
    ]
    matches.sort(key=lambda m: (m.video_id, m.t_start))
    return SearchResult(matches=matches, filter_description=describe(parsed.filters))
