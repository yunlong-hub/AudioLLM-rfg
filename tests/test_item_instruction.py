from rfg.run.conditions import INSTRUCTION, item_instruction


def test_item_instruction_defaults_for_existing_items():
    assert item_instruction({"question_text": "Q"}) == INSTRUCTION


def test_item_instruction_preserves_dataset_prompt():
    item = {"instruction_text": "  Solve step by step.  "}
    assert item_instruction(item) == "Solve step by step."

