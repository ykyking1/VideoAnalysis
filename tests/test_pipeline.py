"""query/pipeline.py::apply_filter_overrides testleri - harici servis
gerektirmez, sadece alan-bazlı birleştirme mantığı (proje-ozeti.md §3.2)."""
from query.llm_parser import StructuredFilters
from query.pipeline import apply_filter_overrides


def test_override_replaces_only_filled_fields():
    base = StructuredFilters(min_agl_m=20.0, is_night=True)
    override = StructuredFilters(min_agl_m=50.0, min_vehicle_count=2)
    merged = apply_filter_overrides(base, override)
    assert merged.min_agl_m == 50.0          # override kazandi
    assert merged.is_night is True           # override'da yok, base korundu
    assert merged.min_vehicle_count == 2     # sadece override'da vardi


def test_override_empty_leaves_base_unchanged():
    base = StructuredFilters(min_agl_m=20.0)
    merged = apply_filter_overrides(base, StructuredFilters())
    assert merged == base


def test_override_can_set_bool_false():
    """False, None'dan ayirt edilmeli - is_empty()/active_fields() bunu
    gerektiriyor, dataclasses.replace ile alan bazli birlestirme bunu dogal
    olarak koruyor ama testle sabitliyoruz."""
    base = StructuredFilters(is_night=True)
    merged = apply_filter_overrides(base, StructuredFilters(is_night=False))
    assert merged.is_night is False
