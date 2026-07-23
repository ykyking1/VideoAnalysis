"""Opsiyonel rerank: Qwen2.5-VL, top 10-20 adayı doğrular (embedding'in görsel
benzerlik yanılgılarını ayıklar) (proje-ozeti.md §3.2 madde 4).
"""
from query.interval_merge import Match


def rerank(query_text: str, candidates: list[Match]) -> list[Match]:
    """Adayları Qwen2.5-VL ile doğrulayıp güven sırasına göre yeniden sıralar."""
    raise NotImplementedError
