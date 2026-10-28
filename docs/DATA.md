# Data formats and preparation

Datasets are supplied separately. This repository includes the preparation code,
input manifests, and the three [paper case studies](case-studies/README.md).

## Evaluation tasks

The evaluator reads one JSON object per line. Each task needs a unique
`source_id`, its `unverified_code`, and a chat `prompt`:

```python
import json
from pathlib import Path
from topoverus.data import prompt

code = Path("program.rs").read_text()
task = {
    "source_id": "custom/program",
    "unverified_code": code,
    "prompt": prompt(code),
    "source_family_status": "external_or_unknown",
}
Path("data").mkdir(exist_ok=True)
Path("data/eval.jsonl").write_text(json.dumps(task) + "\n")
```

`prompt(code, errors)` also accepts existing verifier feedback. For obfuscated
benchmark IDs beginning with `Obfuscated_Verusbench/`, `source_family_status`
is used to group results. This label records source-family metadata.

Use the evaluation command in the [README](../README.md). Reference solutions
are not required for evaluating custom tasks.

## Training inputs

`configs/data-inputs.json` lists the raw JSONL filenames and hashes expected by
the preparation script. Place these inputs in `data/raw/`.

- Task rows contain `prompt_messages`, a reference `answer`, and `meta` source identifiers.
- SFT trajectory rows contain `messages` with the buggy program and structured assistant target.

Build the safety checker, then prepare the data:

```bash
python scripts/prepare_safety_tool.py
python scripts/prepare_data.py \
  --raw-dir data/raw --output data/clean-v4 \
  --verus /absolute/path/to/verus \
  --lynette .deps/lynette-safety-v4/target/release/lynette
```

Preparation checks input hashes, separates training and held-out source
identities, validates SFT targets, writes the retained JSONL files, converts RL
and held-out rows to Parquet, and constructs invariant dependency graphs.
`configs/data-output-sha256.json` pins the prepared files expected by this recipe.

The prepared directory contains `sft_train.jsonl`, `rl_train.jsonl`,
`heldout.jsonl`, `main_eval.jsonl`, `split.json`, `graphs.jsonl`, the Parquet
files, and validation metadata used by the launchers.

## Individual preparation commands

For task inputs with the schema above, the individual tools are:

```bash
python -m topoverus.data \
  --train data/raw/rl_verus_train.jsonl \
  --heldout data/raw/rl_verus_val.jsonl \
  --evaluation data/raw/main_472_eval.jsonl \
  --sft data/raw/qwq_sft_chat_train_v3_ctx_8k.jsonl \
  --output data/clean-v4 \
  --verus /absolute/path/to/verus \
  --lynette .deps/lynette-safety-v4/target/release/lynette

python -m topoverus.parquet \
  --input data/clean-v4/rl_train.jsonl --output data/clean-v4/rl_train.parquet
python -m topoverus.parquet \
  --input data/clean-v4/heldout.jsonl --output data/clean-v4/heldout.parquet
python -m topoverus.build_graphs \
  --tasks data/clean-v4/rl_train.jsonl --output data/clean-v4/graphs.jsonl \
  --verus /absolute/path/to/verus \
  --lynette .deps/lynette-safety-v4/target/release/lynette
```

Dataset and benchmark excerpts retain their original source terms; the code
license does not assign a license to the combined data.
