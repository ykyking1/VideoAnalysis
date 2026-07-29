"""Filtre kurma ve gevşetme mantığı testleri.

Bu mantık, ölçtüğümüz gerçek bir riske karşı yazıldı: dar/yanlış bir hard
filtre doğru cevabı yapısal olarak dışlıyor (gerçek irtifa testinde 21
sorgunun 17'si). Testler o riskin kod tarafında gerçekten ele alındığını
doğruluyor - bkz. docs/worklog_2026-07-28.md.
"""
from query.filter_builder import RELAXATION_ORDER, build_filter, describe, relaxation_ladder
from query.llm_parser import StructuredFilters


def test_empty_filters_produce_no_qdrant_filter():
    assert build_filter(StructuredFilters()) is None


def test_single_field_produces_one_condition():
    f = build_filter(StructuredFilters(sensor_type="ir"))
    assert f is not None and len(f.must) == 1


def test_speed_min_and_max_share_one_range_condition():
    f = build_filter(StructuredFilters(min_speed_kmh=10.0, max_speed_kmh=50.0))
    assert len(f.must) == 1
    assert f.must[0].range.gte == 10.0
    assert f.must[0].range.lte == 50.0


def test_over_sea_false_is_a_real_condition_not_ignored():
    """over_sea=False 'karada' demek - null ile ayni sey degil."""
    f = build_filter(StructuredFilters(over_sea=False))
    assert f is not None and len(f.must) == 1


def test_is_night_true_and_false_produce_opposite_ranges():
    night = build_filter(StructuredFilters(is_night=True)).must[0]
    day = build_filter(StructuredFilters(is_night=False)).must[0]
    assert night.range.lt is not None
    assert day.range.gte is not None


def test_ladder_starts_with_original_and_ends_empty():
    filters = StructuredFilters(sensor_type="ir", is_night=True, over_sea=True)
    ladder = relaxation_ladder(filters)

    assert ladder[0][0] == filters
    assert ladder[0][1] == []
    assert ladder[-1][0].is_empty()


def test_ladder_drops_one_field_per_step():
    filters = StructuredFilters(sensor_type="ir", is_night=True, over_sea=True)
    ladder = relaxation_ladder(filters)
    assert len(ladder) == 4  # orijinal + 3 dusme
    assert [len(dropped) for _, dropped in ladder] == [0, 1, 2, 3]


def test_ladder_drops_inferred_fields_before_explicit_ones():
    """En cok cikarima dayanan alan once dusmeli; kullanicinin acikca
    yazdigi sensor_type en son."""
    filters = StructuredFilters(sensor_type="ir", is_night=True, is_sunset=True)
    dropped_order = [dropped[-1] for _, dropped in relaxation_ladder(filters)[1:]]
    assert dropped_order.index("is_sunset") < dropped_order.index("sensor_type")
    assert dropped_order.index("is_night") < dropped_order.index("sensor_type")


def test_ladder_on_empty_filters_is_single_step():
    assert len(relaxation_ladder(StructuredFilters())) == 1


def test_every_filter_field_appears_in_relaxation_order():
    """Yeni bir filtre alani eklenip RELAXATION_ORDER'a eklenmezse o alan
    hicbir zaman gevsetilmez - sessiz bir Recall kaybi olur."""
    fields = set(StructuredFilters.__dataclass_fields__)
    assert fields == set(RELAXATION_ORDER)


def test_describe_lists_only_active_fields():
    assert describe(StructuredFilters()) == "filtre yok"
    assert "sensor_type=ir" in describe(StructuredFilters(sensor_type="ir"))
