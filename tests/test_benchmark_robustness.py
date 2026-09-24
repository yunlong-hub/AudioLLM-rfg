from rfg.score.benchmark_robustness import duration_group, stratified_summary


def test_duration_group_boundaries():
    assert duration_group(15.0) == "le15"
    assert duration_group(15.1) == "15to30"
    assert duration_group(30.0) == "15to30"
    assert duration_group(30.1) == "gt30"


def test_stratified_summary_uses_paired_differences():
    rows = [
        {"duration_sec": 10.0, "correct": {"internal": True, "asr1": False,
                                             "asr2": True, "asr3": False}},
        {"duration_sec": 20.0, "correct": {"internal": False, "asr1": True,
                                             "asr2": False, "asr3": False}},
        {"duration_sec": 40.0, "correct": {"internal": True, "asr1": False,
                                             "asr2": False, "asr3": True}},
    ]
    result = stratified_summary(rows, n_boot=100)
    assert result["all"]["n"] == 3
    assert result["le30"]["n"] == 2
    assert result["gt30"]["n"] == 1
    assert result["all"]["paired_drop_from_internal"]["asr1"]["mean"] == 1 / 3
