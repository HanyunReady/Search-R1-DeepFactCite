#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into a base model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Load/merge dtype. Use float32 for training parents; bfloat16 merge is not numerically equivalent to PEFT forward.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["same", "float32", "bfloat16"],
        default="same",
        help="Optional dtype cast before saving. Use --torch-dtype float32 --save-dtype bfloat16 for a disk-efficient GRPO actor.",
    )
    args = parser.parse_args()

    disable_peft_torchao_dispatcher()

    from peft import PeftModel

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.torch_dtype]
    tokenizer_source = args.adapter if (Path(args.adapter) / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    if args.save_dtype != "same":
        save_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.save_dtype]
        model = model.to(save_dtype)
        model.config.torch_dtype = save_dtype
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)
    print(f"merged_model={output}")


def disable_peft_torchao_dispatcher() -> None:
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_import_utils.is_torchao_available = lambda: False
        peft_lora_torchao.is_torchao_available = lambda: False
    except Exception:
        pass


if __name__ == "__main__":
    main()
