"""Check actual framework preprocessing and optimizer before loading 32B weights."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .runtime import temp_address


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=8192)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    temporary = temp_address(root)
    os.environ.update(TMPDIR=str(temporary), HF_HOME=str(root / ".cache/huggingface"),
                      HF_DATASETS_CACHE=str(root / ".cache/datasets"),
                      TORCH_EXTENSIONS_DIR=str(root / ".cache/torch_extensions"),
                      TRITON_CACHE_DIR=str(root / ".cache/triton"), PYTHONNOUSERSITE="1")
    for name in ("huggingface", "datasets", "torch_extensions", "triton"):
        (root / ".cache" / name).mkdir(parents=True, exist_ok=True)
    import multiprocess
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from openrlhf.datasets.sft_dataset import SFTDataset
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    manager = multiprocess.Manager()
    manager.shutdown()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = load_dataset("json", data_files=str(args.data), split="train")
    strategy = SimpleNamespace(args=SimpleNamespace(input_key="messages", output_key=None,
                                apply_chat_template=True, tokenizer_chat_template=None))
    prepared = SFTDataset(rows, tokenizer, args.max_len, strategy, num_processors=8)
    if len(prepared) != len(rows):
        raise ValueError(f"Framework unexpectedly dropped SFT rows: {len(prepared)}/{len(rows)}")
    prepared[0]
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    optimizer = DeepSpeedCPUAdam([parameter], lr=2e-5)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    del optimizer
    result = {"status": "passed", "rows": len(prepared), "max_len": args.max_len,
              "tmp_address": str(temporary), "tmp_storage": str(temporary.resolve()),
              "cpu_adam_step": "passed", "num_processors": 8}
    (root / "runs/sft-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
