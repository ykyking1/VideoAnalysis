"""Arama arayüzü: doğal dil sorgusu -> video kimliği + zaman aralığı,
tarayıcıda (proje-ozeti.md §1, §3.2 - `scripts/query_cli.py`'nin aynı
`query/pipeline.py` hattını kullanan web arayüzü hali).

KAPSAM: arama + manuel filtre alanları + sonuç önizleme (kısa, yeniden
kodlanmış klip - ham/proxy video dosyasının kendisi DEĞİL). Ingest/pipeline
durumu izleme bu sürümde yok, ayrı bir ihtiyaç olarak kaldı.

Kullanım:
    python -m scripts.query_ui
    python -m scripts.query_ui --port 7861 --share

vLLM erişilemezse (yapısal ayrıştırma kapalıysa) sorgu otomatik olarak
tamamen semantik aramaya düşer - bu, `query/llm_parser.py`'nin kendi geri
çekilme davranışı. MANUEL FİLTRELER tam bunun için var: vLLM kapalıyken de
(ya da vLLM'in bulduğu tek bir alanı düzeltmek için) filtreli arama
yapılabilsin diye - bkz. query/pipeline.py::apply_filter_overrides().
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

from common import config
from common.console import use_utf8_stdout
from common.minio_client import get_client
from query.interval_merge import Interval
from query.llm_parser import StructuredFilters
from query.pipeline import QueryResponse, run_query

use_utf8_stdout()

EXAMPLE_QUERIES = [
    "gün batımında kıyıya yaklaşan tekne",
    "en az 3 aracın göründüğü, alçak irtifada uçuşlar",
    "gece deniz üzerinde hareket eden tekne",
]

_TRISTATE = ["Farketmez", "Evet", "Hayır"]
_TRISTATE_MAP = {"Farketmez": None, "Evet": True, "Hayır": False}


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
    """Aralıkları kart-benzeri Markdown listesi olarak render eder. Aşağıdaki
    "sonuç #" alanına bu listedeki numarayı (1'den başlar) yazıp Önizle'ye
    basarak kısa bir klip görülebilir - bkz. do_preview()."""
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


def build_manual_filters(sensor_type, min_speed, max_speed, min_agl, max_agl,
                          over_sea, is_sunset, is_night, min_vehicles) -> StructuredFilters:
    """UI widget değerlerinden StructuredFilters kurar - boş/"Farketmez"
    bırakılan alanlar None kalır (dokunulmamış sayılır, bkz.
    query/pipeline.py::apply_filter_overrides).

    NEDEN gr.Number DEĞİL METİN KUTUSU: JavaScript'te boş bir sayı girdisi
    okunurken `Number("")` `0` döner (tarayıcı davranışı, Gradio'ya özgü
    değil) - bu yüzden dokunulmamış bir gr.Number alanı Python tarafına
    `None` değil sessizce `0` olarak geliyordu ve GERÇEK bir filtre gibi
    uygulanıp (ör. "yüzen insan" sorgusunda min_speed_kmh=0 aktif filtre
    oldu, gevşetme merdivenini gereksiz tetikledi) - gerçek kullanıcıda
    bulundu. Metin kutusunda boş string `0`'a dönüşmüyor, ayırt edilebiliyor."""
    return StructuredFilters(
        sensor_type=(sensor_type or "").strip() or None,
        min_speed_kmh=_parse_optional_float(min_speed),
        max_speed_kmh=_parse_optional_float(max_speed),
        min_agl_m=_parse_optional_float(min_agl),
        max_agl_m=_parse_optional_float(max_agl),
        over_sea=_TRISTATE_MAP.get(over_sea),
        is_sunset=_TRISTATE_MAP.get(is_sunset),
        is_night=_TRISTATE_MAP.get(is_night),
        min_vehicle_count=_parse_optional_int(min_vehicles),
    )


def _parse_optional_float(text) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    return float(text.replace(",", "."))


def _parse_optional_int(text) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    return int(float(text))


def do_search(query: str, top_k: int, rerank: bool,
              sensor_type, min_speed, max_speed, min_agl, max_agl,
              over_sea, is_sunset, is_night, min_vehicles):
    query = (query or "").strip()
    if not query:
        return "_Bir sorgu girin._", "", []
    try:
        overrides = build_manual_filters(
            sensor_type, min_speed, max_speed, min_agl, max_agl,
            over_sea, is_sunset, is_night, min_vehicles,
        )
        response = run_query(query, top_k=top_k, enable_rerank=rerank, filter_overrides=overrides)
    except ValueError as exc:
        return f"**Hata:** manuel filtre alanlarından biri sayı olarak okunamadı ({exc}).", "", []
    except Exception as exc:  # noqa: BLE001 - kullaniciya arayuzde goster, coksun istemiyoruz
        err = (
            f"**Hata:** {exc}\n\n"
            f"Qdrant ({config.QDRANT_HOST}:{config.QDRANT_PORT}) ve gerekiyorsa vLLM "
            f"({config.VLLM_BASE_URL}) erişilebilir mi kontrol edin - "
            "`python -m scripts.check_env`."
        )
        return err, "", []
    return render_filter_info(response), render_results(response), response.intervals


def fetch_preview_clip(interval: Interval) -> str:
    """Proxy videodan [t_start, t_end] aralığını kırpıp tarayıcı-uyumlu
    (H.264, sesiz - proxy'nin kendisi de sessiz, bkz.
    ingest/activities/proxy_generation.py) kısa bir MP4'e dönüştürür.

    NEDEN YENIDEN KODLAMA: proxy'ler HEVC (240-360p, model kalitesi) -
    tarayıcılarda HEVC desteği tutarsız (ör. Firefox'ta yok). Kırpılan klip
    zaten kısa (birkaç dakika en fazla) oldugu icin yeniden kodlama hizli."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg PATH'te bulunamadı - önizleme için gerekli.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="query_ui_preview_"))
    proxy_path = tmp_dir / "proxy.mp4"
    clip_path = tmp_dir / "clip.mp4"

    client = get_client()
    client.fget_object(config.MINIO_BUCKET_PROXY, f"{interval.video_id}/proxy.mp4", str(proxy_path))

    duration = max(interval.duration_s, 0.5)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(interval.t_start), "-i", str(proxy_path), "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
        "-movflags", "+faststart", str(clip_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg klip kesme başarısız: {result.stderr[-500:]}")
    return str(clip_path)


def do_preview(intervals: list, idx: int):
    if not intervals:
        return None, "_Önce bir arama yapın._"
    if idx is None or not (1 <= int(idx) <= len(intervals)):
        return None, f"_Sonuç # 1 ile {len(intervals)} arasında olmalı._"
    interval = intervals[int(idx) - 1]
    try:
        clip_path = fetch_preview_clip(interval)
    except Exception as exc:  # noqa: BLE001 - kullaniciya arayuzde goster
        return None, (
            f"**Önizleme alınamadı:** {exc}\n\n"
            f"Video: `{interval.video_id}`, proxy bucket'ında "
            f"({config.MINIO_BUCKET_PROXY}) bulunuyor mu kontrol edin."
        )
    return clip_path, f"**{interval.video_id}**  {format_timestamp(interval.t_start)} – {format_timestamp(interval.t_end)}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="İHA Video Arama") as app:
        gr.Markdown(
            "# İHA Video Arşivinde Semantik Arama\n"
            "Doğal dil sorgusu → video kimliği + zaman aralığı. "
            "Yapısal kısıtlar (irtifa, hız, araç sayısı, gece/gündüz, deniz üstü) "
            "sorgu metninden otomatik çıkarılır - vLLM kapalıysa ya da onun "
            "bulduğunu düzeltmek isterseniz aşağıdaki **Manuel filtreler**'i kullanın."
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

        with gr.Accordion("Manuel filtreler (opsiyonel - vLLM'in bulduğunu ezer)", open=False):
            gr.Markdown(
                "_Boş bırakılan alanlar dokunulmamış sayılır. Sayısal alanlar "
                "metin kutusu - `gr.Number`'ın boş alanı sessizce 0'a çevirdiği "
                "(tarayıcı davranışı) gerçek kullanıcıda bulundu, bu yüzden "
                "metin kutusu kullanılıyor._"
            )
            with gr.Row():
                sensor_type_box = gr.Textbox(label="sensor_type", placeholder="ör. rgb, ir")
                min_vehicles_box = gr.Textbox(label="min. araç sayısı", placeholder="ör. 2")
            with gr.Row():
                min_speed_box = gr.Textbox(label="min. hız (km/s)", placeholder="boş = farketmez")
                max_speed_box = gr.Textbox(label="maks. hız (km/s)", placeholder="boş = farketmez")
            with gr.Row():
                min_agl_box = gr.Textbox(label="min. irtifa (m, AGL)", placeholder="boş = farketmez")
                max_agl_box = gr.Textbox(label="maks. irtifa (m, AGL)", placeholder="boş = farketmez")
            with gr.Row():
                over_sea_radio = gr.Radio(_TRISTATE, value="Farketmez", label="deniz üstü")
                is_sunset_radio = gr.Radio(_TRISTATE, value="Farketmez", label="gün batımı")
                is_night_radio = gr.Radio(_TRISTATE, value="Farketmez", label="gece")

        filter_info = gr.Markdown()
        results = gr.Markdown()
        intervals_state = gr.State([])

        search_inputs = [
            query_box, top_k_slider, rerank_checkbox,
            sensor_type_box, min_speed_box, max_speed_box, min_agl_box, max_agl_box,
            over_sea_radio, is_sunset_radio, is_night_radio, min_vehicles_box,
        ]
        search_outputs = [filter_info, results, intervals_state]
        search_btn.click(do_search, search_inputs, search_outputs)
        query_box.submit(do_search, search_inputs, search_outputs)

        gr.Markdown("### Önizleme\nYukarıdaki listeden bir sonuç numarası girip kısa bir klip görün "
                     "(ham video değil, model-kalite proxy'den kırpılmış).")
        with gr.Row():
            preview_idx = gr.Number(label="Sonuç #", value=1, precision=0, scale=1)
            preview_btn = gr.Button("Önizle", scale=1)
        preview_status = gr.Markdown()
        preview_video = gr.Video(label="Önizleme", autoplay=True)

        preview_btn.click(do_preview, [intervals_state, preview_idx], [preview_video, preview_status])

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
