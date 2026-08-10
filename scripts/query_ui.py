"""Arama arayüzü: doğal dil sorgusu -> video kimliği + zaman aralığı,
tarayıcıda (proje-ozeti.md §1, §3.2 - `scripts/query_cli.py`'nin aynı
`query/pipeline.py` hattını kullanan web arayüzü hali).

KAPSAM (ilk sürüm, bilinçli olarak dar tutuldu): sadece arama + sonuç
metadata'sı. Video önizleme/oynatma YOK - sonuç kartında video_id, zaman
aralığı, skor, pencere sayısı, caption (varsa) gösteriliyor, video dosyasının
kendisi sunulmuyor. Ingest/pipeline durumu izleme de bu sürümde yok, ayrı bir
ihtiyaç olarak kaldı.

Kullanım:
    python -m scripts.query_ui
    python -m scripts.query_ui --port 7861 --share

vLLM erişilemezse (yapısal ayrıştırma kapalıysa) sorgu otomatik olarak
tamamen semantik aramaya düşer - bu, `query/llm_parser.py`'nin kendi geri
çekilme davranışı, bu dosya ayrıca bir şey yapmıyor.
"""
import argparse

import gradio as gr

from common import config
from common.console import use_utf8_stdout
from query.pipeline import QueryResponse, run_query

use_utf8_stdout()

EXAMPLE_QUERIES = [
    "gün batımında kıyıya yaklaşan tekne",
    "en az 3 aracın göründüğü, alçak irtifada uçuşlar",
    "gece deniz üzerinde hareket eden tekne",
]


def format_timestamp(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def render_filter_info(response: QueryResponse) -> str:
    """Yapısal filtre + gevşetme durumunu Markdown olarak özetler
    (scripts/query_cli.py'nin render() fonksiyonunun aynısı, konsol yerine
    Markdown çıktısı üretir)."""
    lines = []
    parsed = response.parsed
    if parsed is not None:
        active = parsed.filters.active_fields()
        semantic_only = " *(sorgu tamamen semantik)*" if not active else ""
        lines.append(f"**Yapısal filtre:** {response.filter_description}{semantic_only}")
        lines.append(f"**Semantik metin:** {parsed.semantic_text!r}")

    if response.was_relaxed:
        lines.append(
            f"\n⚠️ **Filtre gevşetildi:** `{', '.join(response.relaxed_fields)}` düşürüldü. "
            "Hard filtreyle yeterli sonuç bulunamadı - filtreye TAM uymayan sonuçlar "
            "aşağıda **[yaklaşık]** olarak işaretli."
        )

    rerank_note = ", rerank uygulandı" if response.reranked else ""
    ladder_note = f"  (gevşetme {response.ladder_steps} adım)" if response.ladder_steps > 1 else ""
    lines.append(
        f"\n**{len(response.intervals)} aralık** ({response.elapsed_ms:.0f}ms{rerank_note})  \n"
        f"gecikme: `{response.timing_summary()}`{ladder_note}"
    )
    return "\n\n".join(lines)


def render_results(response: QueryResponse) -> str:
    """Aralıkları kart-benzeri Markdown listesi olarak render eder.
    Video ÖNİZLEMESİ/OYNATMA yok (bkz. modül docstring'i) - sadece metadata."""
    if not response.intervals:
        return "_Sonuç bulunamadı._"

    blocks = []
    for i, interval in enumerate(response.intervals, 1):
        badge = "" if interval.exact_filter_match else "  `[yaklaşık]`"
        header = (
            f"**{i}. {interval.video_id}**{badge}  \n"
            f"{format_timestamp(interval.t_start)} – {format_timestamp(interval.t_end)}"
            f"  ·  {interval.duration_s:.0f}sn  ·  {interval.n_windows} pencere"
            f"  ·  skor={interval.score:.3f}"
        )
        captions = "\n".join(f"> \"{c}\"" for c in interval.captions[:2])
        blocks.append(header + ("\n" + captions if captions else ""))
    return "\n\n---\n\n".join(blocks)


def do_search(query: str, top_k: int, rerank: bool):
    query = (query or "").strip()
    if not query:
        return "_Bir sorgu girin._", ""
    try:
        response = run_query(query, top_k=top_k, enable_rerank=rerank)
    except Exception as exc:  # noqa: BLE001 - kullaniciya arayuzde goster, coksun istemiyoruz
        err = (
            f"**Hata:** {exc}\n\n"
            f"Qdrant ({config.QDRANT_HOST}:{config.QDRANT_PORT}) ve gerekiyorsa vLLM "
            f"({config.VLLM_BASE_URL}) erişilebilir mi kontrol edin - "
            "`python -m scripts.check_env`."
        )
        return err, ""
    return render_filter_info(response), render_results(response)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="İHA Video Arama") as app:
        gr.Markdown(
            "# İHA Video Arşivinde Semantik Arama\n"
            "Doğal dil sorgusu → video kimliği + zaman aralığı. "
            "Yapısal kısıtlar (irtifa, hız, araç sayısı, gece/gündüz, deniz üstü) "
            "sorgu metninden otomatik çıkarılır."
        )
        with gr.Row():
            query_box = gr.Textbox(
                label="Sorgu",
                placeholder="ör. \"gün batımında kıyıya yaklaşan tekne\"",
                scale=4,
            )
            search_btn = gr.Button("Ara", variant="primary", scale=1)
        with gr.Row():
            top_k_slider = gr.Slider(
                minimum=1, maximum=100, value=config.SEARCH_TOP_K, step=1,
                label="top_k",
            )
            rerank_checkbox = gr.Checkbox(
                value=config.RERANK_ENABLED,
                label="VLM rerank (yavaş - aday başına bir çağrı)",
            )
        gr.Examples(examples=EXAMPLE_QUERIES, inputs=query_box)

        filter_info = gr.Markdown()
        results = gr.Markdown()

        search_btn.click(do_search, [query_box, top_k_slider, rerank_checkbox], [filter_info, results])
        query_box.submit(do_search, [query_box, top_k_slider, rerank_checkbox], [filter_info, results])

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Gradio'nun genel (public) linkini olustur")
    args = ap.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
