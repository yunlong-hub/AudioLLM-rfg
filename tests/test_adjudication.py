from rfg.audit.adjudication import _agreement_stratum, stratified_sample


def test_three_asr_agreement_strata() -> None:
    a, b, c = (("number", "1", "+"),), (("number", "2", "+"),), (("number", "3", "+"),)
    assert _agreement_stratum([a, a, a]) == "all_three_agree"
    assert _agreement_stratum([a, a, b]) == "two_agree"
    assert _agreement_stratum([a, b, c]) == "all_differ"


def test_stratified_sample_keeps_human_fields_blank() -> None:
    rows = []
    for index in range(120):
        rows.append({
            "run_id": "run",
            "model": f"m{index % 3}",
            "item_id": f"num_{index:04d}",
            "category": "number",
            "stratum": ("all_differ", "two_agree", "all_three_agree")[index % 3],
            "human": {"adjudicated": {"transcript": None, "facts": None}},
        })
    sample = stratified_sample(rows, n=100, seed=7)
    assert len(sample) == 100
    assert len({row["audit_id"] for row in sample}) == 100
    assert all(row["human"]["adjudicated"]["transcript"] is None for row in sample)
