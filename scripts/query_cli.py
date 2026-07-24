"""Doğal dil sorgusu -> video kimliği + zaman aralığı listesi (proje-ozeti.md §1).

Kullanım: python scripts/query_cli.py "gün batımında deniz üzerinde ..."
"""
import sys

from query.hybrid_search import search
from query.interval_merge import merge_matches
from query.llm_parser import parse_query


def format_timestamp(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def run(raw_query: str) -> None:
    parsed = parse_query(raw_query)
    print(f"Yapısal filtreler: {parsed.filters}")
    print(f"Semantik metin: {parsed.semantic_text!r}")

    matches = search(parsed)
    intervals = merge_matches(matches)

    if not intervals:
        print("Sonuç bulunamadı.")
        return
    for interval in sorted(intervals, key=lambda i: (i.video_id, i.t_start)):
        print(f"{interval.video_id}  {format_timestamp(interval.t_start)} - {format_timestamp(interval.t_end)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Kullanım: python scripts/query_cli.py "sorgu metni"')
        sys.exit(1)
    run(sys.argv[1])
