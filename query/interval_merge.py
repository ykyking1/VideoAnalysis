"""Ardışık eşleşen pencereleri sürekli aralıklara birleştirir
(proje-ozeti.md §3.2 madde 3).

Sistemin çıktısı "hangi video" değil "hangi video + hangi zaman aralığı".
Eşleşen 8sn'lik pencereler ardışıksa bunları tek bir aralık olarak sunmak
gerekiyor - kullanıcıya 6 ayrı 8sn'lik satır değil, tek bir 48sn'lik aralık
gösterilmeli.

BOŞLUK TOLERANSI: Pencereler arasındaki boşluk INTERVAL_GAP_TOLERANCE_S'den
küçükse birleştirilir. 8sn/8sn örtüşmesiz pencerelemede ardışık iki pencere
arasındaki boşluk 0'dır; tolerans, aradaki bir pencerenin eşleşmemiş olduğu
(ama olayın sürdüğü) durumları yakalar.
"""
from dataclasses import dataclass

from common import config


@dataclass(frozen=True)
class Interval:
    video_id: str
    t_start: float
    t_end: float
    score: float = 0.0
    n_windows: int = 1
    exact_filter_match: bool = True
    captions: tuple[str, ...] = ()

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


def merge_matches(matches, gap_tolerance_s: float | None = None) -> list[Interval]:
    """Aynı video içinde zaman sırasına göre ardışık eşleşmeleri birleştirir.

    Birleşen aralığın skoru, içindeki pencerelerin EN İYİ skorudur (ortalama
    değil): bir aralığın alaka düzeyini en güçlü kanıtı belirler, aralığı
    uzatan zayıf pencereler skoru sulandırmamalı.

    Bir aralıkta filtreye tam uymayan (gevşetilmiş) tek bir pencere varsa
    aralığın tamamı `exact_filter_match=False` sayılır - kullanıcıya
    olduğundan kesin gösterilmemeli."""
    gap = gap_tolerance_s if gap_tolerance_s is not None else config.INTERVAL_GAP_TOLERANCE_S

    by_video: dict[str, list] = {}
    for m in matches:
        by_video.setdefault(m.video_id, []).append(m)

    intervals: list[Interval] = []
    for video_id, video_matches in by_video.items():
        video_matches.sort(key=lambda m: m.t_start)

        current = _new_group(video_matches[0])
        for m in video_matches[1:]:
            if m.t_start - current["t_end"] <= gap:
                current["t_end"] = max(current["t_end"], m.t_end)
                current["score"] = max(current["score"], getattr(m, "score", 0.0))
                current["n_windows"] += 1
                current["exact"] = current["exact"] and getattr(m, "exact_filter_match", True)
                _add_caption(current, m)
            else:
                intervals.append(_to_interval(video_id, current))
                current = _new_group(m)
        intervals.append(_to_interval(video_id, current))

    intervals.sort(key=lambda i: i.score, reverse=True)
    return intervals


def _new_group(match) -> dict:
    group = {
        "t_start": match.t_start,
        "t_end": match.t_end,
        "score": getattr(match, "score", 0.0),
        "n_windows": 1,
        "exact": getattr(match, "exact_filter_match", True),
        "captions": [],
    }
    _add_caption(group, match)
    return group


def _add_caption(group: dict, match) -> None:
    caption = getattr(match, "caption", "") or ""
    if caption and caption not in group["captions"]:
        group["captions"].append(caption)


def _to_interval(video_id: str, group: dict) -> Interval:
    return Interval(
        video_id=video_id,
        t_start=group["t_start"],
        t_end=group["t_end"],
        score=group["score"],
        n_windows=group["n_windows"],
        exact_filter_match=group["exact"],
        captions=tuple(group["captions"]),
    )
