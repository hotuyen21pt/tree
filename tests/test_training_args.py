"""TrainingArguments differs across transformers majors; the builder must adapt.

The Kaggle image moved to transformers 5, which dropped `warmup_ratio`, and
the run died with `TypeError: got an unexpected keyword argument`. These tests
stand in for both majors so that failure cannot come back.
"""
import inspect
import sys
import types

import pytest


def _stub_heavy_dependencies() -> None:
    """Let `training.py` import without torch/peft, which tests do not need."""
    if "torch" not in sys.modules:
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
        torch.tensor = lambda *a, **k: None
        sys.modules.update({"torch": torch, "torch.utils": utils, "torch.utils.data": data})
    for name, attrs in (
        ("peft", ("LoraConfig", "PeftModel", "TaskType", "get_peft_model")),
        ("transformers", ("AutoModelForCausalLM", "AutoTokenizer", "Trainer",
                          "TrainerCallback", "TrainingArguments")),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            for attr in attrs:
                setattr(module, attr, type(attr, (), {}))
            module.__version__ = "0"
            sys.modules[name] = module


_stub_heavy_dependencies()

from tree_cot_rt import training  # noqa: E402


def _training_arguments_class(accepted_fields):
    """A TrainingArguments that accepts exactly these fields, like a real one."""
    namespace: dict = {}
    params = ", ".join(f"{name}=None" for name in accepted_fields)
    exec(f"def __init__(self, {params}):\n    pass", namespace)
    signature = inspect.signature(namespace["__init__"])

    class TrainingArguments:
        __signature__ = signature

        def __init__(self, **kwargs):
            unexpected = set(kwargs) - set(accepted_fields)
            if unexpected:
                raise TypeError(
                    "TrainingArguments.__init__() got an unexpected keyword "
                    f"argument {sorted(unexpected)[0]!r}"
                )
            self.received = kwargs

    return TrainingArguments


@pytest.fixture
def with_transformers(monkeypatch):
    def apply(accepted_fields, version):
        cls = _training_arguments_class(accepted_fields)
        stub = types.ModuleType("transformers")
        stub.__version__ = version
        stub.TrainingArguments = cls
        monkeypatch.setitem(sys.modules, "transformers", stub)
        monkeypatch.setattr(training, "TrainingArguments", cls)

    return apply


MODERN = (
    "output_dir", "learning_rate", "max_steps",
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "lr_scheduler_type", "logging_steps", "report_to",
)
LEGACY = MODERN + ("warmup_ratio", "disable_tqdm", "eval_strategy")

DESIRED = dict(
    output_dir="/tmp/out",
    learning_rate=5e-5,
    max_steps=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    report_to="none",
)


def test_unsupported_fields_are_dropped_not_raised(with_transformers, capsys):
    with_transformers(MODERN, "5.16.1")
    args = training._training_arguments(**DESIRED)
    assert "warmup_ratio" not in args.received
    assert args.received["lr_scheduler_type"] == "cosine"
    assert args.received["max_steps"] == 10
    out = capsys.readouterr().out
    assert "warmup_ratio" in out and "5.16.1" in out


def test_everything_is_kept_when_the_library_accepts_it(with_transformers):
    with_transformers(LEGACY, "4.48.3")
    args = training._training_arguments(**DESIRED)
    assert args.received["warmup_ratio"] == 0.03
    assert set(args.received) == set(DESIRED)


def test_losing_an_essential_field_stops_the_run(with_transformers):
    # Training without the paper's schedule silently would be worse than failing.
    with_transformers(tuple(f for f in MODERN if f != "lr_scheduler_type"), "9.0.0")
    with pytest.raises(RuntimeError, match="lr_scheduler_type"):
        training._training_arguments(**DESIRED)
