# Training

The training route is QwQ-32B → counterexample SFT → dependency-aware GRPO.
Run commands from the repository root. Use separate SFT and RL environments
with `PYTHONNOUSERSITE=1`.

| Component | SFT | RL / inference |
| --- | --- | --- |
| Python | 3.11 | 3.12 |
| PyTorch | 2.8.0 | 2.6.0 |
| Transformers | 4.57.3 | 4.52.1 |
| Trainer | OpenRLHF 0.9.0 / DeepSpeed 0.18.1 | VERL, reconstructed with the supplied patch |
| vLLM | — | 0.8.5.post1 |
| FlashAttention | 2.8.3 | 2.7.4.post1 |

The configurations target four L40S GPUs. Model weights and datasets are
obtained separately; see [data preparation](DATA.md).

## Environments

In the SFT environment, install:

```bash
python -m pip install -r configs/sft-requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install openrlhf==0.9.0 --no-deps
python -m pip install --no-deps -e .
```

The separate OpenRLHF installation preserves Transformers 4.57.3 from the
requirements file.

In the RL environment, install and reconstruct VERL:

```bash
python -m pip install -r configs/rl-requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python scripts/prepare_rl_framework.py
python -m pip install --no-deps -e .deps/verl
python -m pip install --no-deps -e .
```

The reconstruction applies the supplied patch to VERL v0.6.0 and checks its
source hashes. See [third-party dependencies](../third_party/README.md).
Rust/Cargo is required to build the safety checker:

```bash
python scripts/prepare_safety_tool.py
```

## Base model

Download the original QwQ-32B checkpoint in the RL environment:

```bash
python scripts/download_model.py --output data/models/QwQ-32B
python -m topoverus.checkpoint --model data/models/QwQ-32B \
  --official-manifest runs/model-download/manifest.json \
  --output runs/model-download/weight-validation.json
```

The downloader pins the original base revision. SFT uses these original weights;
GRPO initializes from the resulting SFT checkpoint.

## SFT

With the prepared data at `data/clean-v4`, activate the SFT environment:

```bash
bash scripts/train_sft.sh \
  --model data/models/QwQ-32B \
  --data data/clean-v4 \
  --run runs/sft \
  --checkpoint-hashes runs/model-download/weight-validation.json
```

The launcher uses `configs/sft.json`, checks data lengths and model identity,
and saves the final checkpoint under `runs/sft/checkpoints`.

## Dependency-aware GRPO

Activate the RL environment:

```bash
bash scripts/train_dep.sh \
  --model runs/sft/checkpoints \
  --data data/clean-v4 \
  --run runs/dep \
  --verus /absolute/path/to/verus \
  --lynette .deps/lynette-safety-v4/target/release/lynette
```

RL uses `configs/dep_rl.json` and `configs/reward.json`. Its reward combines
Verus feedback, response format, and matched invariant dependencies. Training
requires the safety checker. The final actor is saved under
`runs/dep/checkpoints/global_step_*/actor/huggingface/`.

Export its FP32 master weights to BF16 in place:

```bash
PYTHONNOUSERSITE=1 python -m topoverus.export_model --run runs/dep
```

Datasets must live under `data/` and run outputs under `runs/`. New runs require
new output directories. Each launcher stores its command, configuration,
source snapshot, and input hashes alongside the run.
