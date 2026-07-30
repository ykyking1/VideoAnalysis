"""Golden set üzerinde retrieval başarımı ölçer (proje-ozeti.md §7).

İki mod:

1) --golden <dosya.jsonl>  (ASIL YÖNTEM, §7'nin gerektirdiği)
   Her satır: {"query": "...", "video_id": "...", "t_start": 12.0, "t_end": 40.0}
   Sorgu tam hattan (LLM ayrıştırma + hibrit arama + aralık birleştirme)
   geçirilir; beklenen aralıkla ZAMAN ÖRTÜŞMESİ olan bir sonuç bulunursa
   isabet sayılır. Recall@1/@5/@10 ve MRR raporlanır.

2) --self-retrieval  (ZAYIF PROXY, golden set yokken sağlık sinyali)
   Caption'lı kliplerin kendi caption'ını sorgu olarak kullanır. "Model kendi
   ürettiği açıklamayı tanıyor mu" sorusunu cevaplar - "kullanıcının doğal dil
   sorgusunu doğru anlıyor mu" sorusunu DEĞİL. Model seçimi kararına (§5)
   temel oluşturmamalı.

ÖRNEKLEM UYARISI: Bu projede N=21'lik bir ölçekte tek bir kliplik kaymanın
Recall'ü ~5 puan oynattığını ölçtük. §7 200-500 sorgu öneriyor - altındaki
örneklemlerde çıkan farkları "model A, B'den iyi" diye yorumlamayın.

Kullanım:
    python -m scripts.eval_retrieval --golden poc/golden_set/queries.jsonl
    python -m scripts.eval_retrieval --self-retrieval
"""
import argparse
import json
import sys
from pathlib import Path

from common.console import use_utf8_stdout
from common import config
from common.qdrant_store import get_client
from ingest.activities.clip_embedding import embed_text
from query.hybrid_search import search
from query.interval_merge import merge_matches
from query.llm_parser import ParsedQuery, StructuredFilters, parse_query
from query.pipeline import run_query

use_utf8_stdout()

RECALL_LEVELS = (1, 5, 10)
MIN_OVERLAP_S = 1.0


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return min(a_end, b_end) - max(a_start, b_start) >= MIN_OVERLAP_S


def evaluate_golden(path: Path, top_k: int) -> int:
    if not path.exists():
        print(f"Golden set bulunamadi: {path}")
        print("Bkz. poc/golden_set/README.md - format ve tasarim rehberi orada.")
        return 1

    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        print(f"{path} bos")
        return 1

    hits = {k: 0 for k in RECALL_LEVELS}
    reciprocal_ranks: list[float] = []
    relaxed_count = 0

    for case in cases:
        response = run_query(case["query"], top_k=top_k)
        if response.was_relaxed:
            relaxed_count += 1

        rank = None
        for i, interval in enumerate(response.intervals, start=1):
            if interval.video_id != case["video_id"]:
                continue
            if _overlaps(interval.t_start, interval.t_end,
                          float(case["t_start"]), float(case["t_end"])):
                rank = i
                break

        if rank is None:
            reciprocal_ranks.append(0.0)
            continue
        reciprocal_ranks.append(1.0 / rank)
        for k in RECALL_LEVELS:
            if rank <= k:
                hits[k] += 1

    n = len(cases)
    print(f"--- Golden set sonuclari ({path.name}) ---")
    print(f"Sorgu sayisi: {n}")
    for k in RECALL_LEVELS:
        print(f"Recall@{k}: {hits[k]}/{n} ({100 * hits[k] / n:.1f}%)")
    print(f"MRR      : {sum(reciprocal_ranks) / n:.3f}")
    print(f"Filtre gevsetilen sorgu: {relaxed_count}/{n}")

    if n < 200:
        print(f"\nUYARI: proje-ozeti.md §7 200-500 sorgu oneriyor, elinizde {n} var. "
              f"\nBu orneklemde tek bir sorgunun kaymasi Recall'u ~{100/n:.1f} puan oynatir - "
              f"\nmodel/strateji karsilastirmasi icin yetersiz.")
    return 0


def _recall_and_mrr(cases: list[dict], top_k: int, use_filter: bool) -> dict:
    hits = {k: 0 for k in RECALL_LEVELS}
    reciprocal_ranks: list[float] = []
    relaxed_count = 0

    for case in cases:
        if use_filter:
            parsed = parse_query(case["query"])
        else:
            # Yapisal filtre YOK - LLM ayristirmasi bile devre disi, sorgunun
            # tamami ham metin olarak embedding'e gidiyor. Bu, filtrenin
            # kendisinin katkisini/maliyetini izole eder (LLM ayristirma
            # kalitesinden ayri bir degisken).
            parsed = ParsedQuery(filters=StructuredFilters(),
                                  semantic_text=case["query"], raw_query=case["query"])

        result = search(parsed, top_k=top_k)
        if result.was_relaxed:
            relaxed_count += 1
        intervals = merge_matches(result.matches)

        rank = None
        for i, interval in enumerate(intervals, start=1):
            if interval.video_id != case["video_id"]:
                continue
            if _overlaps(interval.t_start, interval.t_end,
                          float(case["t_start"]), float(case["t_end"])):
                rank = i
                break

        if rank is None:
            reciprocal_ranks.append(0.0)
            continue
        reciprocal_ranks.append(1.0 / rank)
        for k in RECALL_LEVELS:
            if rank <= k:
                hits[k] += 1

    n = len(cases)
    return {"hits": hits, "mrr": sum(reciprocal_ranks) / n, "relaxed": relaxed_count, "n": n}


def evaluate_filter_ablation(path: Path, top_k: int) -> int:
    """Ayni golden set uzerinde FILTRELI vs FILTRESIZ Recall karsilastirmasi.

    docs/worklog_2026-07-28.md'deki hard-filtre olcumunun (sentetik + N=21
    gercek veri) devami - burada gercek pipeline (vLLM ayristirma dahil)
    ve daha buyuk bir korpusla tekrarlaniyor. N<200 ise sonuc EGILIM
    sinyalidir, "model A B'den iyi" turu kesin bir iddia degildir."""
    if not path.exists():
        print(f"Golden set bulunamadi: {path}")
        print("Bkz. poc/golden_set/README.md - format ve tasarim rehberi orada.")
        return 1

    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        print(f"{path} bos")
        return 1

    with_filter = _recall_and_mrr(cases, top_k, use_filter=True)
    without_filter = _recall_and_mrr(cases, top_k, use_filter=False)
    n = with_filter["n"]

    print(f"--- Filtre ablasyonu ({path.name}, N={n}) ---")
    print(f"{'':10}{'FILTRELI':>12}{'FILTRESIZ':>12}{'fark':>10}")
    for k in RECALL_LEVELS:
        wf, wof = with_filter["hits"][k], without_filter["hits"][k]
        diff = 100 * (wf - wof) / n
        print(f"Recall@{k:<3}{100*wf/n:>11.1f}%{100*wof/n:>11.1f}%{diff:>+9.1f}p")
    print(f"{'MRR':<10}{with_filter['mrr']:>12.3f}{without_filter['mrr']:>12.3f}"
          f"{with_filter['mrr']-without_filter['mrr']:>+10.3f}")
    print(f"\nFiltre gevsetilen sorgu (filtreli modda): {with_filter['relaxed']}/{n}")

    if n < 200:
        print(f"\nUYARI: N={n} < proje-ozeti.md §7'nin onerdigi 200-500. "
              f"Egilim sinyali olarak okuyun, kesin karsilastirma olarak degil "
              f"(docs/worklog_2026-07-28.md'deki N=21 olcumunde ayni uyari gecerliydi).")
    return 0


def evaluate_self_retrieval(top_k: int) -> int:
    client = get_client()
    collection = config.QDRANT_COLLECTION

    captioned = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        captioned += [p.payload for p in points if (p.payload or {}).get("caption")]
        if offset is None:
            break

    if not captioned:
        print("Caption'li klip bulunamadi - once caption'li ingest calistirin.")
        return 1

    from qdrant_client.http import models as qm

    hits = {k: 0 for k in RECALL_LEVELS}
    for payload in captioned:
        vector = embed_text(payload["caption"])
        result = client.query_points(
            collection_name=collection, query=vector,
            limit=max(RECALL_LEVELS), with_payload=True,
            search_params=qm.SearchParams(hnsw_ef=config.QDRANT_SEARCH_HNSW_EF),
        )
        for i, point in enumerate(result.points, start=1):
            got = point.payload or {}
            if (got.get("video_id") == payload["video_id"]
                    and abs(float(got.get("t_start", -1)) - float(payload["t_start"])) < 1e-3):
                for k in RECALL_LEVELS:
                    if i <= k:
                        hits[k] += 1
                break

    n = len(captioned)
    print("--- Self-retrieval (ZAYIF PROXY) ---")
    print(f"Degerlendirilen caption: {n}")
    for k in RECALL_LEVELS:
        print(f"Recall@{k}: {hits[k]}/{n} ({100 * hits[k] / n:.1f}%)")
    print("\nUYARI: Bu bir golden-set metrigi DEGIL. Model secimi (§5) kararina "
          "\ntemel olusturmamali - bkz. dosya docstring'i.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, help="JSONL golden set dosyasi")
    ap.add_argument("--self-retrieval", action="store_true")
    ap.add_argument("--compare-filters", action="store_true",
                     help="--golden ile birlikte: filtreli vs filtresiz Recall karsilastirmasi")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    if args.golden and args.compare_filters:
        return evaluate_filter_ablation(args.golden, args.top_k)
    if args.golden:
        return evaluate_golden(args.golden, args.top_k)
    if args.self_retrieval:
        return evaluate_self_retrieval(args.top_k)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
