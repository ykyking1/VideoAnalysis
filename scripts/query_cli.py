"""Doğal dil sorgusu -> video kimliği + zaman aralığı (proje-ozeti.md §1).

Kullanım:
    python -m scripts.query_cli "gun batiminda deniz uzerinde iki tekne"
    python -m scripts.query_cli "..." --top-k 50 --rerank
    python -m scripts.query_cli --interactive
"""
import argparse
import sys

from query.pipeline import run_query

sys.stdout.reconfigure(encoding="utf-8")


def format_timestamp(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def render(response) -> None:
    parsed = response.parsed
    if parsed is not None:
        active = parsed.filters.active_fields()
        print(f"Yapisal filtre : {response.filter_description}"
              f"{'' if active else ' (sorgu tamamen semantik)'}")
        print(f"Semantik metin : {parsed.semantic_text!r}")

    if response.was_relaxed:
        print(f"\n! Filtre gevsetildi: {', '.join(response.relaxed_fields)} dusuruldu.")
        print("  Hard filtreyle yeterli sonuc bulunamadi. Filtreye TAM uymayan")
        print("  sonuclar asagida [yaklasik] olarak isaretli.")

    print(f"\n{len(response.intervals)} aralik ({response.elapsed_ms:.0f}ms"
          f"{', rerank uygulandi' if response.reranked else ''})")
    # Kirilim: "vLLM ayristirma ne kadar yavaslatiyor" sorusunun cevabi
    # parse= degeri. proje-ozeti.md §8'in 300ms tahmini hic olculmedi.
    print(f"  gecikme: {response.timing_summary()}"
          f"{f'  (gevsetme {response.ladder_steps} adim)' if response.ladder_steps > 1 else ''}\n")

    if not response.intervals:
        print("Sonuc bulunamadi.")
        return

    for i, interval in enumerate(response.intervals, 1):
        marker = "" if interval.exact_filter_match else "  [yaklasik]"
        print(f"{i:2d}. {interval.video_id}  "
              f"{format_timestamp(interval.t_start)} - {format_timestamp(interval.t_end)}  "
              f"({interval.duration_s:.0f}s, {interval.n_windows} pencere, "
              f"skor={interval.score:.3f}){marker}")
        for caption in interval.captions[:2]:
            print(f"      \"{caption}\"")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="?")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--rerank", action="store_true", help="VLM rerank'i bu sorgu icin ac")
    ap.add_argument("--interactive", "-i", action="store_true")
    args = ap.parse_args()

    rerank_flag = True if args.rerank else None

    if args.interactive:
        print("Sorgu girin (bos satir = cikis)\n")
        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                break
            render(run_query(raw, top_k=args.top_k, enable_rerank=rerank_flag))
            print()
        return 0

    if not args.query:
        ap.print_help()
        return 1

    render(run_query(args.query, top_k=args.top_k, enable_rerank=rerank_flag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
