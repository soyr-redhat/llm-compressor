"""
Tests for inverse_weight_map approach that eliminates the
reindex_fused_weights preprocessing step for microscale schemes.
"""

import pytest
import torch
from compressed_tensors.entrypoints.convert import build_inverse_weight_maps
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import save_file

from llmcompressor.entrypoints.model_free.process import ModelFreePtqConverter


def _make_nvfp4_scheme():
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="float",
            strategy="tensor_group",
            group_size=16,
            symmetric=True,
            dynamic=False,
            scale_dtype=torch.float8_e4m3fn,
        ),
    )


def _rand_weight(*shape):
    return torch.randn(*shape, dtype=torch.float16)


@pytest.fixture
def mfptq():
    return ModelFreePtqConverter(
        scheme=_make_nvfp4_scheme(),
        scheme_name="config_group_0",
        ignore=[],
    )


class TestBuildInverseWeightMaps:
    def test_single_file(self, mfptq, tmp_path):
        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "shard-00001.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "shard-00001.safetensors",
            "model.layers.1.self_attn.k_proj.weight": "shard-00001.safetensors",
            "model.layers.1.self_attn.v_proj.weight": "shard-00001.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(tmp_path / "shard-00001.safetensors"),
        }
        inverse_weight_maps = build_inverse_weight_maps(
            weight_map, model_files, [mfptq]
        )
        inverse_weight_maps["shard-00001.safetensors"][
            str(tmp_path / "shard-00001.safetensors")
        ].sort()
        assert inverse_weight_maps == {
            "shard-00001.safetensors": {
                str(tmp_path / "shard-00001.safetensors"): [
                    "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.v_proj.weight",
                    "model.layers.1.self_attn.k_proj.weight",
                    "model.layers.1.self_attn.q_proj.weight",
                    "model.layers.1.self_attn.v_proj.weight",
                ],
            }
        }

    def test_missing_dependency(self, mfptq, tmp_path):
        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "shard-00001.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(tmp_path / "shard-00001.safetensors"),
        }
        with pytest.raises(ValueError):
            build_inverse_weight_maps(weight_map, model_files, [mfptq])

    def test_invalid_weight_map(self, mfptq, tmp_path):
        weight_map = {
            "tensor.a": "shard-00001.safetensors",
            "tensor.b": "shard-00002.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(tmp_path / "shard-00001.safetensors"),
        }
        with pytest.raises(KeyError):
            build_inverse_weight_maps(weight_map, model_files, [mfptq])

    def test_all_colocated(self, mfptq, tmp_path):
        """All fused weights in same shard — no cross-shard fetching needed."""
        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "shard-00001.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "shard-00002.safetensors",
            "model.layers.1.self_attn.k_proj.weight": "shard-00002.safetensors",
            "model.layers.1.self_attn.v_proj.weight": "shard-00002.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(tmp_path / "shard-00001.safetensors"),
            "shard-00002.safetensors": str(tmp_path / "shard-00002.safetensors"),
        }
        inverse_weight_maps = build_inverse_weight_maps(
            weight_map, model_files, [mfptq]
        )
        assert set(
            inverse_weight_maps["shard-00001.safetensors"][
                str(tmp_path / "shard-00001.safetensors")
            ]
        ) == {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        }
        assert set(
            inverse_weight_maps["shard-00002.safetensors"][
                str(tmp_path / "shard-00002.safetensors")
            ]
        ) == {
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
        }

    def test_cross_shard_partners_found(self, mfptq, tmp_path):
        """q_proj on shard1, k/v on shard2 — shard1 should fetch from shard2."""
        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "shard-00002.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "shard-00002.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "shard-00002.safetensors",
            "model.layers.1.self_attn.k_proj.weight": "shard-00001.safetensors",
            "model.layers.1.self_attn.v_proj.weight": "shard-00001.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(tmp_path / "shard-00001.safetensors"),
            "shard-00002.safetensors": str(tmp_path / "shard-00002.safetensors"),
        }
        inverse_weight_maps = build_inverse_weight_maps(
            weight_map, model_files, [mfptq]
        )
        assert set(
            inverse_weight_maps["shard-00001.safetensors"][
                str(tmp_path / "shard-00001.safetensors")
            ]
        ) == {
            "model.layers.0.self_attn.q_proj.weight",
        }
        assert set(
            inverse_weight_maps["shard-00001.safetensors"][
                str(tmp_path / "shard-00002.safetensors")
            ]
        ) == {
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
        }
        assert set(
            inverse_weight_maps["shard-00002.safetensors"][
                str(tmp_path / "shard-00001.safetensors")
            ]
        ) == {
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
        }
        assert set(
            inverse_weight_maps["shard-00002.safetensors"][
                str(tmp_path / "shard-00002.safetensors")
            ]
        ) == {
            "model.layers.1.self_attn.q_proj.weight",
        }


class TestModelFreePtqConverterProcessMicroscaleColocated:
    """Tests for co-located fused weights — standard case, no cross-shard needed."""

    @pytest.fixture
    def qkv_tensors(self):
        return {
            "model.layers.0.self_attn.q_proj.weight": _rand_weight(32, 32),
            "model.layers.0.self_attn.k_proj.weight": _rand_weight(32, 32),
            "model.layers.0.self_attn.v_proj.weight": _rand_weight(32, 32),
            "model.layers.0.mlp.down_proj.weight": _rand_weight(32, 32),
        }

    def test_colocated_fused_weights(self, mfptq, qkv_tensors):
        """Standard case: all fused weights in one shard."""
        result = mfptq.process(qkv_tensors)
        assert len(result) > 0
        assert not any(k.endswith(".weight") and "scale" not in k for k in result)


class TestModelFreePtqConverterProcessMicroscaleCrossShardInverseMap:
    """Tests for cross-shard fused weights using precomputed inverse_weight_map."""

    @pytest.fixture
    def split_shards(self, mfptq, tmp_path):
        """q_proj on shard-1, k_proj + v_proj + down_proj on shard-2."""
        shard1_tensors = {
            "model.layers.0.self_attn.q_proj.weight": _rand_weight(32, 32),
        }
        shard2_tensors = {
            "model.layers.0.self_attn.k_proj.weight": _rand_weight(32, 32),
            "model.layers.0.self_attn.v_proj.weight": _rand_weight(32, 32),
            "model.layers.0.mlp.down_proj.weight": _rand_weight(32, 32),
        }
        shard1_path = tmp_path / "shard-00001.safetensors"
        shard2_path = tmp_path / "shard-00002.safetensors"
        save_file(shard1_tensors, shard1_path)
        save_file(shard2_tensors, shard2_path)

        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "shard-00002.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "shard-00002.safetensors",
            "model.layers.0.mlp.down_proj.weight": "shard-00002.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(shard1_path),
            "shard-00002.safetensors": str(shard2_path),
        }
        inverse_weight_maps = build_inverse_weight_maps(
            weight_map, model_files, [mfptq]
        )
        return (
            shard1_path,
            shard2_path,
            inverse_weight_maps["shard-00001.safetensors"],
            inverse_weight_maps["shard-00002.safetensors"],
        )

    def test_shard1_produces_output(self, mfptq, split_shards, tmp_path):
        """Shard-1 (q_proj only) processes correctly using precomputed inverse map."""
        from compressed_tensors.utils.safetensors_load import (
            load_tensors_from_inverse_weight_map,
        )

        shard1_path, _, iwm1, _ = split_shards
        save_path = tmp_path / "out-00001.safetensors"

        tensors = load_tensors_from_inverse_weight_map(iwm1, "cpu")
        result = mfptq.process(tensors)

        from safetensors.torch import save_file as sf

        sf(result, save_path)
        assert save_path.exists()
        assert len(result) > 0

    def test_shard2_produces_output(self, mfptq, split_shards, tmp_path):
        """Shard-2 (k/v/down) processes correctly using precomputed inverse map."""
        from compressed_tensors.utils.safetensors_load import (
            load_tensors_from_inverse_weight_map,
        )

        _, shard2_path, _, iwm2 = split_shards
        save_path = tmp_path / "out-00002.safetensors"

        tensors = load_tensors_from_inverse_weight_map(iwm2, "cpu")
        result = mfptq.process(tensors)

        from safetensors.torch import save_file as sf

        sf(result, save_path)
        assert save_path.exists()
        assert len(result) > 0

    def test_both_shards_produce_same_keys_as_merged(
        self, mfptq, split_shards, tmp_path
    ):
        """Combined output keys from both shards should match merged single-shard keys."""
        from safetensors.torch import load_file
        from compressed_tensors.utils.safetensors_load import (
            load_tensors_from_inverse_weight_map,
        )

        shard1_path, shard2_path, iwm1, iwm2 = split_shards

        result1 = mfptq.process(load_tensors_from_inverse_weight_map(iwm1, "cpu"))
        result2 = mfptq.process(load_tensors_from_inverse_weight_map(iwm2, "cpu"))
        combined_keys = set(result1.keys()) | set(result2.keys())

        # Process merged shard as reference
        merged = {**load_file(shard1_path), **load_file(shard2_path)}
        result_merged = mfptq.process(merged)

        assert combined_keys == set(result_merged.keys()), (
            f"Key mismatch:\n"
            f"  split only: {sorted(combined_keys - set(result_merged.keys()))}\n"
            f"  merged only: {sorted(set(result_merged.keys()) - combined_keys)}"
        )
