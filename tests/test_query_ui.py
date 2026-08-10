"""scripts/query_ui.py'nin saf biçimlendirme fonksiyonları - harici servis
(Qdrant/vLLM/Gradio sunucusu) gerektirmez, sadece Markdown çıktısını test
eder (proje-ozeti.md §3.2)."""
from query.interval_merge import Interval
from query.llm_parser import ParsedQuery, StructuredFilters
from query.pipeline import QueryResponse
from scripts.query_ui import format_timestamp, render_filter_info, render_results


def test_format_timestamp_under_hour():
    assert format_timestamp(75) == "0:01:15"


def test_format_timestamp_over_hour():
    assert format_timestamp(3725) == "1:02:05"


def test_render_results_empty():
    assert "Sonuç bulunamadı" in render_results(
        QueryResponse(query="x", intervals=[])
    )


def test_render_results_lists_intervals_with_caption():
    interval = Interval(
        video_id="sds_train_4", t_start=0, t_end=47, score=0.368,
        n_windows=5, exact_filter_match=True, captions=("tekne görünüyor",),
    )
    out = render_results(QueryResponse(query="tekne", intervals=[interval]))
    assert "sds_train_4" in out
    assert "0:00:00" in out and "0:00:47" in out
    assert "tekne görünüyor" in out
    assert "[yaklaşık]" not in out


def test_render_results_marks_approximate_match():
    interval = Interval(
        video_id="v1", t_start=0, t_end=10, exact_filter_match=False,
    )
    out = render_results(QueryResponse(query="x", intervals=[interval]))
    assert "[yaklaşık]" in out


def test_render_filter_info_semantic_only():
    parsed = ParsedQuery(filters=StructuredFilters(), semantic_text="tekne", raw_query="tekne")
    response = QueryResponse(
        query="tekne", parsed=parsed, filter_description="filtre yok",
        intervals=[], elapsed_ms=68453.0, timings={"parse": 15.0, "embed": 68303.0},
    )
    out = render_filter_info(response)
    assert "sorgu tamamen semantik" in out
    assert "filtre yok" in out


def test_render_filter_info_shows_relaxation_warning():
    parsed = ParsedQuery(filters=StructuredFilters(min_agl_m=20.0), semantic_text="", raw_query="x")
    response = QueryResponse(
        query="x", parsed=parsed, filter_description="agl_m>=20",
        relaxed_fields=["min_agl_m"], intervals=[], ladder_steps=2,
    )
    out = render_filter_info(response)
    assert "gevşetildi" in out
    assert "min_agl_m" in out
