"""A 7B in fp16 nearly fills a 16 GB card; the LoRA rank decides if a run is possible.

A Kaggle run died inside `Trainer.__init__` after four minutes with a CUDA
out-of-memory traceback that never named LoRA. These tests pin the arithmetic
that now catches it before any weights are loaded.
"""
import sys
import types

import pytest

from tests.test_training_args import _stub_heavy_dependencies  # noqa: F401

_stub_heavy_dependencies()

from tree_cot_rt import training  # noqa: E402

GIB = 2**30

#: Qwen2.5-7B-Instruct, the backbone these notebooks attach.
QWEN_7B = dict(
    num_hidden_layers=28,
    hidden_size=3584,
    intermediate_size=18944,
    num_attention_heads=28,
    num_key_value_heads=4,
    vocab_size=152064,
    tie_word_embeddings=False,
)


@pytest.fixture
def gpus(monkeypatch):
    """Pretend `count` devices of `gib` each, with the real Qwen config."""

    def apply(count, gib):
        config = types.SimpleNamespace(**QWEN_7B)
        stub = types.ModuleType("transformers")
        stub.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *a, **k: config)
        monkeypatch.setitem(sys.modules, "transformers", stub)
        monkeypatch.setattr(training.torch.cuda, "is_available", lambda: True, raising=False)
        monkeypatch.setattr(training.torch.cuda, "device_count", lambda: count, raising=False)
        monkeypatch.setattr(
            training.torch.cuda,
            "get_device_properties",
            lambda i: types.SimpleNamespace(total_memory=int(gib * GIB)),
            raising=False,
        )

    return apply


def test_parameter_count_matches_the_published_size(gpus):
    gpus(1, 14.56)
    params = training._model_parameter_count("Qwen/Qwen2.5-7B-Instruct")
    assert 7.3e9 < params < 7.9e9, params          # published: 7.62B


def test_lora_grows_linearly_with_rank(gpus):
    gpus(1, 14.56)
    small = training._lora_parameter_count("x", 8)
    large = training._lora_parameter_count("x", 32)
    assert large == 4 * small
    assert 70e6 < large < 90e6, large              # ~80.7M at rank 32


def test_rank_32_is_refused_on_one_16gb_card(gpus):
    # The actual Kaggle failure.
    gpus(1, 14.56)
    with pytest.raises(RuntimeError) as error:
        training.check_gpu_memory("Qwen/Qwen2.5-7B-Instruct", 32)
    message = str(error.value)
    assert "T4 x2" in message
    assert "LORA_RANK" in message


def test_rank_8_still_fits_one_card_but_warns(gpus, capsys):
    # This is the configuration of the run that trained for eight hours:
    # 0.15 GiB spare. Allowed, but the margin has to be said out loud.
    gpus(1, 14.56)
    training.check_gpu_memory("Qwen/Qwen2.5-7B-Instruct", 8)
    out = capsys.readouterr().out
    assert "GPU budget" in out
    assert "spare for activations" in out


def test_two_cards_allow_the_papers_rank(gpus):
    gpus(2, 14.56)
    training.check_gpu_memory("Qwen/Qwen2.5-7B-Instruct", 32)   # must not raise


def test_no_gpu_is_not_an_error(monkeypatch):
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False, raising=False)
    training.check_gpu_memory("Qwen/Qwen2.5-7B-Instruct", 32)
