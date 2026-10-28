# Model

[TopoVerus-32B](https://huggingface.co/YuechunSun/TopoVerus-32B) is a full-weight
BF16 model derived from Qwen/QwQ-32B through counterexample-guided supervised
fine-tuning and dependency-aware GRPO. The weights occupy approximately 61 GiB.

The [quick start](model-cards/dep-v2.md) shows how to load the model with
Transformers and generate a repair. The [README](../README.md) covers batch
repair with Verus feedback. Model weights are hosted on Hugging Face.

The base model is [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B), revision
`976055f8c83f394f35dbd3ab09a285a984907bd0`. The model uses Apache-2.0;
original TopoVerus code uses MIT. See [licensing](LICENSING.md).
