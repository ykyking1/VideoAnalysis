"""Ardışık eşleşen pencereleri sürekli aralıklara birleştirir (proje-ozeti.md
§3.2 madde 3): ≤GAP_TOLERANCE_S boşluk toleransıyla birleştirme.

Bu mantık model/altyapı seçiminden bağımsız olduğu için diğer query
modüllerinin aksine tam implemente edildi.
"""
from dataclasses import dataclass

GAP_TOLERANCE_S = 10.0


@dataclass(frozen=True)
class Match:
    video_id: str
    t_start: float
    t_end: float


@dataclass(frozen=True)
class Interval:
    video_id: str
    t_start: float
    t_end: float


def merge_matches(matches: list[Match]) -> list[Interval]:
    """Aynı video_id için zaman sırasına göre ardışık eşleşen pencereleri,
    aralarındaki boşluk GAP_TOLERANCE_S'den küçük veya eşitse tek aralığa
    birleştirir."""
    by_video: dict[str, list[Match]] = {}
    for m in matches:
        by_video.setdefault(m.video_id, []).append(m)

    intervals: list[Interval] = []
    for video_id, video_matches in by_video.items():
        video_matches.sort(key=lambda m: m.t_start)

        current_start = video_matches[0].t_start
        current_end = video_matches[0].t_end

        for m in video_matches[1:]:
            if m.t_start - current_end <= GAP_TOLERANCE_S:
                current_end = max(current_end, m.t_end)
            else:
                intervals.append(Interval(video_id, current_start, current_end))
                current_start, current_end = m.t_start, m.t_end

        intervals.append(Interval(video_id, current_start, current_end))

    return intervals
