"""The chain cache is the only thing that makes a lost session cheap to resume.

A Colab runtime is reclaimed without warning and a Kaggle session can be
stopped at any point. The cache is rewritten every 25 typed instances, so an
interrupted write is a realistic event, and it used to leave a truncated last
line that made the next run fail to read any of the file at all.
"""
import json
import sys
import types

import pytest


def _stub_heavy_dependencies() -> None:
    if "torch" in sys.modules:
        return
    torch = types.ModuleType("torch")
    torch.bfloat16 = torch.float16 = torch.float32 = torch.long = object()
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, is_bf16_supported=lambda: False
    )
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    data.Dataset = object
    utils.data = data
    torch.utils = utils
    sys.modules.update({"torch": torch, "torch.utils": utils, "torch.utils.data": data})
    for name, attrs in (
        ("peft", ("LoraConfig", "PeftModel", "TaskType", "get_peft_model")),
        ("transformers", ("AutoModelForCausalLM", "AutoTokenizer", "Trainer",
                          "TrainerCallback", "TrainingArguments")),
    ):
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, type(attr, (), {}))
        module.__version__ = "0"
        sys.modules[name] = module


_stub_heavy_dependencies()

from tree_cot_rt.training import _read_candidates, _write_candidates  # noqa: E402

ROWS = [
    {"instance_id": f"s-{i}", "quad_type": "EAEO", "chain": "x" * 50}
    for i in range(20)
]


def test_round_trip(tmp_path):
    path = tmp_path / "stage1_candidates.jsonl"
    _write_candidates(path, ROWS)
    assert _read_candidates(path) == ROWS


def test_no_partial_file_is_left_behind(tmp_path):
    path = tmp_path / "stage1_candidates.jsonl"
    _write_candidates(path, ROWS)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stage1_candidates.jsonl"]


def test_an_interrupted_write_leaves_the_previous_cache_intact(tmp_path, monkeypatch):
    path = tmp_path / "stage1_candidates.jsonl"
    _write_candidates(path, ROWS)

    # The session disappears while the replacement is being written.
    import tree_cot_rt.training as training

    def die(*args, **kwargs):
        raise KeyboardInterrupt("runtime reclaimed")

    monkeypatch.setattr(training.os, "replace", die)
    with pytest.raises(KeyboardInterrupt):
        _write_candidates(path, ROWS + [{"instance_id": "new", "chain": "y"}])

    # The old cache is still complete, which is the whole point.
    assert _read_candidates(path) == ROWS


def test_a_torn_last_line_costs_one_chain_not_the_cache(tmp_path, capsys):
    path = tmp_path / "stage1_candidates.jsonl"
    body = "".join(json.dumps(row) + "\n" for row in ROWS)
    truncated = json.dumps({"instance_id": "cut", "chain": "z" * 50})[:30]
    path.write_text(body + truncated, encoding="utf-8")

    recovered = _read_candidates(path)
    assert recovered == ROWS
    assert "incomplete" in capsys.readouterr().out


def test_corruption_in_the_middle_still_raises(tmp_path):
    # Only a torn *last* line is explainable by an interrupted write; damage
    # anywhere else means something worse and must not pass silently.
    path = tmp_path / "stage1_candidates.jsonl"
    lines = [json.dumps(row) for row in ROWS]
    lines[5] = "{not json"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        _read_candidates(path)


def test_missing_file_is_empty_not_an_error(tmp_path):
    assert _read_candidates(tmp_path / "absent.jsonl") == []
