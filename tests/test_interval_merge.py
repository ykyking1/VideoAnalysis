from query.interval_merge import GAP_TOLERANCE_S, Interval, Match, merge_matches


def test_merges_overlapping_windows():
    matches = [
        Match("video1", 130.0, 140.0),
        Match("video1", 136.0, 146.0),
        Match("video1", 142.0, 152.0),
    ]
    assert merge_matches(matches) == [Interval("video1", 130.0, 152.0)]


def test_keeps_separate_when_gap_exceeds_tolerance():
    matches = [
        Match("video1", 0.0, 10.0),
        Match("video1", 10.0 + GAP_TOLERANCE_S + 1.0, 30.0),
    ]
    result = merge_matches(matches)
    assert result == [
        Interval("video1", 0.0, 10.0),
        Interval("video1", 10.0 + GAP_TOLERANCE_S + 1.0, 30.0),
    ]


def test_merges_when_gap_equals_tolerance():
    matches = [
        Match("video1", 0.0, 10.0),
        Match("video1", 10.0 + GAP_TOLERANCE_S, 30.0),
    ]
    assert merge_matches(matches) == [Interval("video1", 0.0, 30.0)]


def test_separates_by_video_id():
    matches = [
        Match("video1", 0.0, 10.0),
        Match("video2", 5.0, 15.0),
    ]
    result = merge_matches(matches)
    assert set(result) == {
        Interval("video1", 0.0, 10.0),
        Interval("video2", 5.0, 15.0),
    }


def test_unsorted_input_still_merges_correctly():
    matches = [
        Match("video1", 142.0, 152.0),
        Match("video1", 130.0, 140.0),
        Match("video1", 136.0, 146.0),
    ]
    assert merge_matches(matches) == [Interval("video1", 130.0, 152.0)]
