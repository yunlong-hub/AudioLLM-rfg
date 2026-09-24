from rfg.run.resample_generation import indexed_seeds


def test_indexed_seeds_supports_incremental_candidate_pool():
    assert indexed_seeds("404,505,606,707", 4, 3) == [
        (3, 404),
        (4, 505),
        (5, 606),
        (6, 707),
    ]


def test_indexed_seeds_rejects_negative_offset():
    try:
        indexed_seeds("404", 1, -1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative candidate offset should fail")
