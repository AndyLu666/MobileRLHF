import argparse
import json
import os
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def remap_key(native_key: str) -> str:
    prefix = "layer."
    suffix_a = ".lora_A"
    suffix_b = ".lora_B"
    if not native_key.startswith(prefix):
        raise ValueError(f"unsupported native key: {native_key}")
    if native_key.endswith(suffix_a):
        ab = "A"
        body = native_key[len(prefix):-len(suffix_a)]
    elif native_key.endswith(suffix_b):
        ab = "B"
        body = native_key[len(prefix):-len(suffix_b)]
    else:
        raise ValueError(f"unsupported native key: {native_key}")

    layer_str, target = body.split(".", 1)
    layer_idx = int(layer_str)
    if target == "attn.qkv":
        hf_target = "attn.c_attn"
    elif target == "attn.proj":
        hf_target = "attn.c_proj"
    else:
        raise ValueError(f"unsupported native target for GPT-2 PEFT conversion: {target}")

    return f"base_model.model.transformer.h.{layer_idx}.{hf_target}.lora_{ab}.weight"


def build_adapter_config(base_model_dir: str, native_path: str, metadata: Dict[str, str]) -> Dict[str, object]:
    rank = int(metadata.get("rank", "8"))
    alpha = int(float(metadata.get("alpha", "16")))
    dropout = float(metadata.get("dropout", "0.0"))
    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_dir,
        "bias": "none",
        "corda_config": None,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": True,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": alpha,
        "lora_bias": False,
        "lora_dropout": dropout,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "qalora_group_size": 16,
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        # Use fully-qualified GPT-2 attention module suffixes so PEFT does not
        # also expect LoRA weights for MLP c_proj modules.
        "target_modules": ["attn.c_attn", "attn.c_proj"],
        "target_parameters": None,
        "task_type": "CAUSAL_LM",
        "trainable_token_indices": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert native GPT-2 LoRA safetensors to a PEFT adapter directory")
    parser.add_argument("--native_lora", required=True)
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tensors = {}
    metadata = {}
    with safe_open(args.native_lora, framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        for key in reader.keys():
            if key == "__metadata__":
                continue
            tensor = reader.get_tensor(key)
            mapped_key = remap_key(key)
            if key.endswith(".lora_A"):
                tensors[mapped_key] = tensor.transpose(0, 1).contiguous()
            else:
                tensors[mapped_key] = tensor.transpose(0, 1).contiguous()

    save_file(tensors, os.path.join(args.output_dir, "adapter_model.safetensors"))
    adapter_config = build_adapter_config(args.base_model_dir, args.native_lora, metadata)
    with open(os.path.join(args.output_dir, "adapter_config.json"), "w", encoding="utf-8") as fout:
        json.dump(adapter_config, fout, indent=2, sort_keys=True)
        fout.write("\n")

    report = {
        "status": "ok",
        "native_lora": args.native_lora,
        "output_dir": args.output_dir,
        "tensor_count": len(tensors),
        "rank": adapter_config["r"],
        "alpha": adapter_config["lora_alpha"],
        "dropout": adapter_config["lora_dropout"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
