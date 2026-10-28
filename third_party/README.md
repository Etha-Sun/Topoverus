# Third-party dependencies

## VERL

`verl-source.json` pins the public VERL v0.6.0 archive and the source hashes used
by the training framework. `verl-v0.6.0-training.patch` reconstructs the required
framework, which reports version `0.5.0.dev0`.

```bash
python scripts/prepare_rl_framework.py
python -m pip install --no-deps -e .deps/verl
```

Install the RL requirements and matching FlashAttention build first, as described
in the [training guide](../docs/TRAINING.md). The preparation script verifies the
archive, applies the patch, and checks the resulting source hashes. Use
`--archive` to supply a previously downloaded upstream archive.

The upstream [Apache-2.0 license](VERL-LICENSE) is retained.

## Lynette

`lynette-source.json` pins the retained source archive, checker patch, and
resulting source hashes. Build with Rust/Cargo:

```bash
python scripts/prepare_safety_tool.py
```

The output is `.deps/lynette-safety-v4/target/release/lynette` and supports
`--topoverus-safety-version`. `--offline` uses cached Cargo dependencies.
The checker is optional for evaluation and required for training and data
preparation. See [evaluation modes](../docs/SAFETY.md).

The upstream Microsoft [MIT license](LYNETTE-LICENSE) is retained.

## QwQ-32B

The base model's [Apache-2.0 license and Alibaba Cloud attribution](QWQ-LICENSE)
are retained. Model weights are obtained from Hugging Face.
