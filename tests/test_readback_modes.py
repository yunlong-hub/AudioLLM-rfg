from __future__ import annotations

from rfg.facts.readback import compose_readback
from rfg.facts.schema import Fact


def f(value: str) -> Fact:
    return Fact("number", value, "+")


def test_union_modes_and_companion_keys() -> None:
    facts = {
        "READ": {f("1")},
        "SPEAK#internal": {f("1"), f("2"), f("3")},
        "SPEAK": {f("1")},
        "SPEAK#asr2": {f("2")},
        "SPEAK#asr3": {f("3")},
    }
    assert compose_readback(facts, "asr1")["SPEAK"] == {f("1")}
    assert compose_readback(facts, "union12")["SPEAK"] == {f("1"), f("2")}
    assert compose_readback(facts, "union13")["SPEAK"] == {f("1"), f("3")}
    assert compose_readback(facts, "union123")["SPEAK"] == {f("1"), f("2"), f("3")}
    assert all("#asr" not in key for key in compose_readback(facts, "asr1"))


def test_union_requires_requested_companion() -> None:
    facts = {
        "READ": {f("1")},
        "SPEAK#internal": {f("1")},
        "SPEAK": {f("1")},
        "SPEAK#asr2": {f("1")},
    }
    composed = compose_readback(facts, "union13")
    assert "SPEAK" not in composed
    assert composed["READ"] == {f("1")}
    assert composed["SPEAK#internal"] == {f("1")}
