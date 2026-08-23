import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import save_file

from llmcompressor.entrypoints.model_free.process import ModelFreePtqConverter


def _get_block_scheme() -> QuantizationScheme:
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="block",
            symmetric=True,
            dynamic=False,
            block_structure=[16, 16],
        ),
    )


@pytest.fixture
def mfptq():
    return ModelFreePtqConverter(
        scheme=_get_block_scheme(),
        scheme_name="config_group_0",
        ignore=[],
    )


def test_validate_raises_for_non_2d_linear_weight(mfptq, tmp_path):
    path = tmp_path / "bad_shape.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": torch.ones(128)}, str(path))

    tensors = {"model.layers.0.mlp.down_proj.weight": torch.ones(128)}
    with pytest.raises(ValueError, match="model.layers.0.mlp.down_proj.weight"):
        mfptq.validate(tensors)


def test_validate_does_not_raise_for_block_incompatible_shape(mfptq, tmp_path):
    tensors = {"model.layers.0.mlp.down_proj.weight": torch.ones(17, 16)}
    mfptq.validate(tensors)
