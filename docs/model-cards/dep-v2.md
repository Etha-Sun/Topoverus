---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/QwQ-32B
base_model_relation: finetune
language:
- en
tags:
- topoverus
- verus
- formal-verification
- grpo
---

# TopoVerus-32B

TopoVerus-32B repairs proofs in Verus programs. It is a full-weight BF16 model
derived from [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) through
counterexample-guided supervised fine-tuning and dependency-aware GRPO training.

## Quick start

Install Transformers, PyTorch, Accelerate and safetensors in a compatible GPU
environment. The example uses Transformers 4.52.1 and PyTorch 2.6.0.
The BF16 weights occupy approximately 61 GiB; generation needs additional
memory. Place the Verus program to repair in `program.rs`.

```python
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "YuechunSun/TopoVerus-32B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
code = Path("program.rs").read_text()
messages = [
    {
        "role": "system",
        "content": (
            "You are a Verus formal-verification expert. Repair only proof annotations. "
            "Preserve executable code, function contracts and specification functions. "
            "Respond with <think>...</think>, <counterexample> containing [State], "
            "[Execution Trace] and [Violation], <reasoning>...</reasoning>, then "
            "exactly one final Rust code fence. Do not use assume or admit."
        ),
    },
    {
        "role": "user",
        "content": f"Repair this Verus program.\n\n[Buggy Code]\n```rust\n{code}\n```",
    },
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=8192,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
print(tokenizer.decode(
    output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
))
```

The response includes reasoning and a final fenced Verus program. Extract the
final program and run Verus on it. For iterative repair, provide the previous
candidate and its verifier diagnostics in the next request.

## License and attribution

This model is licensed under Apache-2.0; see `LICENSE`. It is derived from
Qwen/QwQ-32B by Alibaba Cloud. Modifications include supervised fine-tuning,
dependency-aware reinforcement learning, and export of the final weights to BF16.
