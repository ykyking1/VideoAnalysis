"""Pencereleme ve Qdrant nokta kimliği testleri."""
import math

from common import config
from common.qdrant_store import ClipPayload, point_id
from ingest.activities.telemetry_processing import build_windows
from ingest.activities.types import TelemetryWindow


def test_windows_cover_whole_video_without_gaps():
    windows = build_windows(40.0)
    assert windows[0].t_start == 0.0
    assert windows[-1].t_end == 40.0
    for a, b in zip(windows, windows[1:]):
        assert b.t_start == a.t_end  # ortusmesiz ve bosluksuz


def test_non_overlapping_stride_halves_window_count():
    """Ortusmesiz pencerelemenin (STRIDE_S==WINDOW_S) pencere sayisini
    duration/WINDOW_S formulune gore urettigi dogrulaniyor - degeri degil,
    ORTUSMESIZ olma ozelligini test ediyor (proje-ozeti.md §8 kapasite
    hesabi, WINDOW_S'in kendisi ayri bir olcumle secildi, bkz. common/config.py)."""
    assert config.STRIDE_S == config.WINDOW_S, "varsayilan ortusmesiz olmali"
    duration = 80.0
    windows = build_windows(duration)
    assert len(windows) == math.ceil(duration / config.WINDOW_S)


def test_last_window_is_truncated_not_padded():
    windows = build_windows(20.0)
    assert windows[-1].t_end == 20.0
    assert windows[-1].t_end - windows[-1].t_start < config.WINDOW_S


def test_short_video_produces_single_window():
    windows = build_windows(3.0)
    assert len(windows) == 1
    assert (windows[0].t_start, windows[0].t_end) == (0.0, 3.0)


def test_zero_duration_produces_no_windows():
    assert build_windows(0.0) == []


def test_window_key_is_stable():
    w = TelemetryWindow(t_start=8.0, t_end=16.0)
    assert w.key == TelemetryWindow(t_start=8.0, t_end=16.0).key


def test_point_id_is_deterministic_for_idempotent_reingest():
    """Ayni pencere yeniden ingest edilirse satir cogalmamali."""
    assert point_id("video1", 8.0) == point_id("video1", 8.0)


def test_point_id_differs_across_videos_and_windows():
    assert point_id("video1", 8.0) != point_id("video2", 8.0)
    assert point_id("video1", 8.0) != point_id("video1", 16.0)


def test_payload_keeps_null_telemetry_explicit():
    """Bilinmeyen telemetri alani payload'dan SILINMEMELI - 'alan yok' ile
    'alan null' Qdrant'ta farkli davraniyor."""
    payload = ClipPayload(video_id="v1", t_start=0.0, t_end=8.0).to_dict()
    assert "over_sea" in payload and payload["over_sea"] is None
    assert payload["vehicle_count"] == 0
