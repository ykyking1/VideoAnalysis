"""Hibrit arama: ClickHouse tek sorguda skip index'lerle filtre + küçültülmüş
kümede vektör karşılaştırması (proje-ozeti.md §3.2 madde 2). Filtre ve vektör
aynı satırda olduğu için ayrı vektör DB'lerdeki pre-filter/post-filter ikilemi
yok.
"""
from query.interval_merge import Match
from query.llm_parser import ParsedQuery


def search(parsed: ParsedQuery, top_k: int = 20) -> list[Match]:
    """ParsedQuery.filters'ı WHERE koşullarına, semantic_text'i embedding modeline
    çevirip ClickHouse `clips` tablosunda hibrit sorgu çalıştırır."""
    raise NotImplementedError
