import re
from collections import defaultdict
from typing import Iterable

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.utils import match_quantizable_tensors
from loguru import logger

from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_weight,
    initialize_quantized_linear,
    validate_weight_for_quantization,
)
from llmcompressor.entrypoints.model_free.microscale import (
    DEFAULT_FUSED_MAPPINGS,
    get_fused_names,
    is_microscale_scheme,
)
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe,
    update_qparams,
)
from llmcompressor.observers import FusionHandler

__all__ = ["ModelFreePtqConverter"]


class ModelFreePtqConverter:
    """
    Converter that performs model-free PTQ (quantize + compress) on safetensors
    checkpoints. Implements the compressed-tensors Converter protocol so it can
    be chained with other converters (e.g. a dequantizer) via convert_checkpoint.
    """

    def __init__(
        self,
        scheme: QuantizationScheme,
        scheme_name: str,
        ignore: Iterable[str],
    ):
        self.scheme = scheme
        self.scheme_name = scheme_name
        self.ignore = list(ignore)

    @property
    def _is_microscale(self) -> bool:
        return is_microscale_scheme(self.scheme)

    def get_dependencies(self, weight_name: str) -> set[str]:
        """
        For microscale schemes, return fused partner tensor names that must be
        loaded alongside weight_name to compute a shared global scale.
        Standard schemes have no cross-tensor dependencies.
        """
        if not self._is_microscale:
            return set()

        deps = set()
        for primary_pattern, partner_templates in DEFAULT_FUSED_MAPPINGS.items():
            match = re.match(primary_pattern, weight_name)
            if match is None:
                continue
            for template in partner_templates:
                deps.add(template.format(**match.groupdict()))
        return deps

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Validate that each quantizable tensor can be quantized under the scheme.
        Runs on meta tensors (no real computation). Returns tensors unchanged so
        downstream chained converters see the same tensor names/dtypes.
        """
        meta_tensors = {k: v.to("meta") for k, v in tensors.items()}
        meta_tensors = split_fused_moe_experts(meta_tensors)
        for _, name in match_quantizable_tensors(
            meta_tensors, self.ignore, self.scheme.targets
        ):
            validate_weight_for_quantization(meta_tensors[name], self.scheme, name)
        return meta_tensors

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Quantize and compress all tensors in a safetensors shard.
        The device is inferred from the tensors themselves.
        """
        tensors = split_fused_moe_experts(tensors)
        if self._is_microscale:
            return self._process_microscale(tensors)
        return self._process_standard(tensors)

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig:
        """
        Build (or merge into) the QuantizationConfig for this PTQ run.
        When chained after a dequantizer (config=None), creates from scratch.
        When chained after another quantizer, merges the new scheme in.
        """
        new_config = QuantizationConfig.model_validate(
            {
                "config_groups": {self.scheme_name: self.scheme},
                "ignore": self.ignore,
                "quantization_status": QuantizationStatus.COMPRESSED,
                "format": self.scheme.format,
            }
        )
        if config is None:
            return new_config
        config.merge(new_config)
        return config

    def _process_standard(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        device = _infer_device(tensors)
        for module_name, name in match_quantizable_tensors(
            tensors, self.ignore, self.scheme.targets
        ):
            validate_weight_for_quantization(tensors[name], self.scheme, name)

            module = initialize_quantized_linear(tensors[name], self.scheme, device)
            calibrate_weight(module)
            compress_module(module)

            del tensors[name]
            prefix = module_name + "."
            for key, value in module.state_dict(prefix=prefix).items():
                tensors[key] = value.to("cpu")

        return tensors

    def _process_microscale(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        device = _infer_device(tensors)

        fused_sets, _ = get_fused_names(list(tensors.keys()))
        fused_name_to_fused_index: dict[str, int] = {
            name: index
            for index, matched_set in enumerate(fused_sets)
            for name in matched_set.values()
            if name is not None
        }
        fused_modules: dict[int, dict[str, torch.nn.Module]] = defaultdict(dict)

        for module_name, name in match_quantizable_tensors(
            tensors, self.ignore, self.scheme.targets
        ):
            validate_weight_for_quantization(tensors[name], self.scheme, name)

            module = initialize_quantized_linear(tensors[name], self.scheme, device)

            if name in fused_name_to_fused_index:
                fused_index = fused_name_to_fused_index[name]
                fused_modules[fused_index][name] = module
                initialize_observer(module, "weight")
                apply_calibration_status(module)
                continue

            calibrate_weight(module)
            compress_module(module)

            del tensors[name]
            prefix = module_name + "."
            for key, value in module.state_dict(prefix=prefix).items():
                tensors[key] = value.to("cpu")

        for named_modules in fused_modules.values():
            FusionHandler.fuse(
                [(mod.weight_observer, mod) for mod in named_modules.values()]
            )
            observe(named_modules.values(), base_name="weight")
            update_qparams(named_modules.values(), base_name="weight")

            for name, module in named_modules.items():
                freeze_module_quantization(module)
                compress_module(module)

                del tensors[name]
                module_name, _ = name.rsplit(".", 1)
                prefix = module_name + "."
                for key, value in module.state_dict(prefix=prefix).items():
                    tensors[key] = value.to("cpu")

        return tensors


def _infer_device(tensors: dict[str, torch.Tensor]) -> torch.device:
    for t in tensors.values():
        if t.device.type != "meta":
            return t.device
    return torch.device("cpu")


def split_fused_moe_experts(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Find fused MoE experts (with gate_up_proj/down_proj).
    Split them from 3D tensors into individual 2D expert tensors.
    """
    split_tensors = {}

    params_to_split = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "down_proj": ["down_proj"],
    }

    for name, tensor in tensors.items():
        keys_to_split = [key for key in params_to_split if key in name]
        if len(keys_to_split) >= 2:
            raise ValueError(f"Found multiple keys matching {name}: {keys_to_split}")

        elif len(keys_to_split) == 1 and tensor.ndim == 3:
            unsplit_name = keys_to_split[0]
            split_names = params_to_split[unsplit_name]

            num_experts = tensor.shape[0]

            if tensor.shape[1] % len(split_names) != 0:
                raise ValueError(
                    f"{unsplit_name} expects a second dimension divisible by "
                    f"{len(split_names)} but got shape: {tensor.shape}"
                )

            intermediate_size = tensor.shape[1] // len(split_names)
            for expert_idx in range(num_experts):
                expert_tensor = tensor[expert_idx]
                split_layers = expert_tensor.split(intermediate_size, dim=0)
                for split_name, split_layer in zip(split_names, split_layers):
                    key = name.replace(unsplit_name, f"{expert_idx}.{split_name}")
                    if not key.endswith(".weight"):
                        key = f"{key}.weight"
                    split_tensors[key] = split_layer

            logger.info(f"Split {name} into {num_experts} experts")

        else:
            split_tensors[name] = tensor

    return split_tensors
