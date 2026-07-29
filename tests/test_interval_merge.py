from common import config
from query.hybrid_search import Match
from query.interval_merge import merge_matches

GAP = config.INTERVAL_GAP_TOLERANCE_S


def _spans(intervals):
    return [(i.video_id, i.t_start, i.t_end) for i in intervals]


def test_merges_adjacent_windows():
    matches = [
        Match("video1", 130.0, 140.0),
        Match("video1", 136.0, 146.0),
        Match("video1", 142.0, 152.0),
    ]
    assert _spans(merge_matches(matches)) == [("video1", 130.0, 152.0)]


def test_keeps_separate_when_gap_exceeds_tolerance():
    matches = [
        Match("video1", 0.0, 10.0),
        Match("video1", 10.0 + GAP + 1.0, 30.0),
    ]
    assert _spans(merge_matches(matches)) == [
        ("video1", 0.0, 10.0),
        ("video1", 10.0 + GAP + 1.0, 30.0),
    ]


def test_merges_when_gap_equals_tolerance():
    matches = [
        Match("video1", 0.0, 10.0),
        Match("video1", 10.0 + GAP, 30.0),
    ]
    assert _spans(merge_matches(matches)) == [("video1", 0.0, 30.0)]


def test_separates_by_video_id():
    matches = [Match("video1", 0.0, 10.0), Match("video2", 5.0, 15.0)]
    assert set(_spans(merge_matches(matches))) == {
        ("video1", 0.0, 10.0),
        ("video2", 5.0, 15.0),
    }


def test_unsorted_input_still_merges_correctly():
    matches = [
        Match("video1", 142.0, 152.0),
        Match("video1", 130.0, 140.0),
        Match("video1", 136.0, 146.0),
    ]
    assert _spans(merge_matches(matches)) == [("video1", 130.0, 152.0)]


def test_interval_takes_best_score_not_average():
    """Aralığın alaka düzeyini en güçlü kanıt belirlemeli - araligi uzatan
    zayif pencereler skoru sulandirmamali."""
    matches = [
        Match("video1", 0.0, 8.0, score=0.9),
        Match("video1", 8.0, 16.0, score=0.2),
    ]
    merged = merge_matches(matches)
    assert len(merged) == 1
    assert merged[0].score == 0.9
    assert merged[0].n_windows == 2


def test_one_relaxed_window_marks_whole_interval_inexact():
    """Kullaniciya olduğundan kesin gosterilmemeli."""
    matches = [
        Match("video1", 0.0, 8.0, exact_filter_match=True),
        Match("video1", 8.0, 16.0, exact_filter_match=False),
    ]
    merged = merge_matches(matches)
    assert len(merged) == 1
    assert merged[0].exact_filter_match is False


def test_intervals_sorted_by_score_descending():
    matches = [
        Match("video1", 0.0, 8.0, score=0.3),
        Match("video2", 0.0, 8.0, score=0.8),
    ]
    merged = merge_matches(matches)
    assert [i.video_id for i in merged] == ["video2", "video1"]


def test_captions_deduplicated_within_interval():
    matches = [
        Match("video1", 0.0, 8.0, caption="a boat"),
        Match("video1", 8.0, 16.0, caption="a boat"),
        Match("video1", 16.0, 24.0, caption="two boats"),
    ]
    merged = merge_matches(matches)
    assert merged[0].captions == ("a boat", "two boats")


def test_empty_input():
    assert merge_matches([]) == []
