"""poc/auair_adapter.py::load_gt_vehicle_counts / gt_vehicle_count_for_window
ve poc/auair_ingest.py::_merge_visual_fields testleri - harici servis
gerektirmez, saf veri dönüşümü (2026-08-13 worklog: YOLO'nun bu veri
setinde recall %16 ölçülmesi üzerine eklendi - GT'nin YOLO'yu override
ETMEMESİ kritik davranış)."""
import json

from poc.auair_adapter import gt_vehicle_count_for_window, load_gt_vehicle_counts
from poc.auair_ingest import _merge_visual_fields
from ingest.activities.types import TelemetryWindow, VisualFields


def _write_annotations(tmp_path, annotations):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({
        "categories": ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"],
        "annotations": annotations,
    }), encoding="utf-8")
    return str(path)


def test_load_gt_vehicle_counts_excludes_non_coco_classes(tmp_path):
    """Human(0)/Van(3)/Bicycle(5)/Trailer(7) sayilmamali - YOLO'nun
    VEHICLE_LIKE_CLASSES kumesinin COCO karsiligi yok (bkz.
    ingest/activities/visual_fields.py)."""
    path = _write_annotations(tmp_path, [
        {"image_name": "f1.jpg", "bbox": [
            {"class": 0}, {"class": 1}, {"class": 3}, {"class": 5}, {"class": 7},
        ]},
        {"image_name": "f2.jpg", "bbox": [{"class": 2}, {"class": 4}, {"class": 6}]},
    ])
    counts = load_gt_vehicle_counts(path)
    assert counts["f1.jpg"] == 1   # sadece Car(1) sayildi
    assert counts["f2.jpg"] == 3   # Truck(2)+Motorbike(4)+Bus(6)


def test_gt_vehicle_count_for_window_takes_max_not_sum():
    """count_vehicles()'la AYNI yontem: pencere icindeki max es-zamanli,
    toplam degil - ayni aracin birden fazla karede sayilmamasi icin."""
    group = [
        {"t": 0.0, "image_name": "a.jpg"},
        {"t": 30.0, "image_name": "b.jpg"},
        {"t": 59.0, "image_name": "c.jpg"},
    ]
    counts = {"a.jpg": 2, "b.jpg": 7, "c.jpg": 3}
    assert gt_vehicle_count_for_window(0, 60, group, counts) == 7


def test_gt_vehicle_count_for_window_none_when_no_frames():
    """Pencereye hic AU-AIR karesi dusmuyorsa None - cagiran taraf YOLO'ya
    geri dusebilsin diye (bkz. _merge_visual_fields)."""
    group = [{"t": 0.0, "image_name": "a.jpg"}]
    assert gt_vehicle_count_for_window(100, 160, group, {"a.jpg": 5}) is None


def test_merge_visual_fields_gt_overrides_yolo_when_present():
    """AU-AIR gercek etiketi VARSA, YOLO'nun tahmini KULLANILMAZ (2026-08-13
    worklog'daki recall %16 bulgusunun dogrudan sonucu)."""
    windows = [TelemetryWindow(t_start=0.0, t_end=60.0)]
    group = [{"t": 10.0, "image_name": "a.jpg"}]
    gt_counts = {"a.jpg": 25}
    yolo_visual = [VisualFields(vehicle_count=2)]   # YOLO'nun (yanlis) tahmini

    merged, n_gt = _merge_visual_fields(windows, group, gt_counts, yolo_visual)

    assert merged[0].vehicle_count == 25   # GT kazandi, YOLO'nun 2'si atildi
    assert n_gt == 1


def test_merge_visual_fields_falls_back_to_yolo_when_no_gt():
    """Pencereye AU-AIR karesi dusmuyorsa (kenar durum) YOLO'nun tahminine
    geri dusulur - YOLO var olan (bos olmayan) bir sutunu OVERRIDE ETMEZ
    kuralinin simetrigi: GT yoksa YOLO calismaya devam eder."""
    windows = [TelemetryWindow(t_start=1000.0, t_end=1060.0)]
    group = [{"t": 10.0, "image_name": "a.jpg"}]   # pencerenin disinda
    gt_counts = {"a.jpg": 25}
    yolo_visual = [VisualFields(vehicle_count=4)]

    merged, n_gt = _merge_visual_fields(windows, group, gt_counts, yolo_visual)

    assert merged[0].vehicle_count == 4    # GT yok, YOLO korundu
    assert n_gt == 0
