"""A model accelerate spilled to CPU trains ~50x slower and looks like a hang.

A run reported `ETA 22:10:01` for a stage that should take half an hour, and
the only cause that fits that factor is CPU offload. These tests pin the two
defences: never ask for sharding on a single GPU, and refuse a device map that
put anything outside the GPUs.
"""
import types

import pytest

from tests.test_training_args import _stub_heavy_dependencies  # noqa: F401

_stub_heavy_dependencies()

from tree_cot_rt import training  # noqa: E402


@pytest.fixture
def gpu_count(monkeypatch):
    def apply(count):
        monkeypatch.setattr(
            training.torch.cuda, "is_available", lambda: count > 0, raising=False
        )
        monkeypatch.setattr(
            training.torch.cuda, "device_count", lambda: count, raising=False
        )

    return apply


def test_single_gpu_does_not_ask_for_sharding(gpu_count):
    # device_map="auto" on one card offloads instead of failing; that is the
    # 50x slowdown, so a single GPU must load the ordinary way.
    gpu_count(1)
    assert training._auto_device_map() is None


def test_several_gpus_shard(gpu_count):
    gpu_count(2)
    assert training._auto_device_map() == "auto"


def test_cpu_only_does_not_shard(gpu_count):
    gpu_count(0)
    assert training._auto_device_map() is None


def test_offloaded_layers_stop_the_run():
    model = types.SimpleNamespace(
        hf_device_map={
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.26": "cpu",
            "model.layers.27": "disk",
        }
    )
    with pytest.raises(RuntimeError) as error:
        training.assert_not_offloaded(model)
    message = str(error.value)
    assert "T4 x2" in message
    assert "model.layers.26" in message


def test_a_map_entirely_on_gpus_is_accepted(capsys):
    model = types.SimpleNamespace(
        hf_device_map={"model.layers.0": 0, "model.layers.27": 1}
    )
    training.assert_not_offloaded(model)
    assert "Model placement" in capsys.readouterr().out


def test_no_device_map_is_not_an_error():
    training.assert_not_offloaded(types.SimpleNamespace())
