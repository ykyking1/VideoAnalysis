"""Sorgu ayrıştırıcının güvenlik ağı testleri.

En kritik davranış: sorguda hiç geçmeyen bir kavram için LLM `false`
üretirse bu `None`'a çevrilmeli. Aksi halde "tekne ara" sorgusu sessizce
"gece OLMAYAN tekne ara"ya dönüşür ve gece kayıtları yapısal olarak elenir -
bu oturumda bu tür bir dar filtrenin doğru cevabı gerçekten kaybettirdiğini
ölçtük (bkz. docs/worklog_2026-07-28.md).
"""
from query.llm_parser import StructuredFilters, _sanitize, parse_query


def test_false_without_keyword_becomes_null():
    parsed = _sanitize({"is_night": False, "over_sea": False}, "denizde tekne ara")
    # "deniz" gectigi icin over_sea korunur, "gece" gecmedigi icin is_night dusustur
    assert parsed["is_night"] is None
    assert parsed["over_sea"] is False


def test_false_with_keyword_is_preserved():
    parsed = _sanitize({"is_night": False}, "gece olmayan ucuslar")
    assert parsed["is_night"] is False


def test_true_is_never_touched():
    parsed = _sanitize({"is_night": True, "over_sea": True}, "herhangi bir sorgu")
    assert parsed["is_night"] is True
    assert parsed["over_sea"] is True


def test_english_keywords_recognized():
    parsed = _sanitize({"is_sunset": False}, "not at sunset")
    assert parsed["is_sunset"] is False


def test_keyword_match_is_case_insensitive():
    parsed = _sanitize({"is_night": False}, "GECE olmayan")
    assert parsed["is_night"] is False


def test_parse_falls_back_to_semantic_when_llm_unavailable(monkeypatch):
    """vLLM kapaliyken arama tamamen durmamali - sorgu semantik metne duser."""
    import query.llm_parser as parser

    def _boom(*args, **kwargs):
        raise ConnectionError("vLLM kapali")

    monkeypatch.setattr(parser, "chat_json", _boom)

    result = parse_query("deniz uzerinde tekne")
    assert result.semantic_text == "deniz uzerinde tekne"
    assert result.filters.is_empty()


def test_active_fields_and_is_empty():
    assert StructuredFilters().is_empty()
    assert StructuredFilters(sensor_type="ir").active_fields() == ["sensor_type"]
