"""Check actual four-rank packed attention before loading the full RL actor."""

import argparse
from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    faulthandler.enable()
    import torch
    import torch.distributed as dist
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils.model import update_model_config
    from verl.utils.ulysses import set_ulysses_sequence_parallel_group
    from vllm.device_allocator.cumem import CuMemAllocator

    configuration = json.loads(args.config.read_text())
    attention = configuration["+actor_rollout_ref.model.override_config._attn_implementation"]
    assert attention == "flash_attention_2"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    assert dist.get_world_size() == 4
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool(tag="preflight"):
        probe = torch.ones(1024, device="cuda")
    assert probe.sum().item() == 1024
    allocator.sleep(offload_tags=("preflight",))
    allocator.wake_up()
    assert probe.sum().item() == 1024
    del probe
    # Destroy the probe's pool before subsequent native-allocator operations.
    # Keep the pluggable allocator alive until its pool has been destroyed.
    torch.cuda.synchronize()
    allocators = [entry[1] for entry in allocator.allocator_and_pools.values()]
    allocator.allocator_and_pools.clear()
    torch.cuda.empty_cache()
    rank = dist.get_rank()
    architecture = json.loads((args.model / "config.json").read_text())
    assert architecture["model_type"] == "qwen2"
    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=32, hidden_size=architecture["num_attention_heads"] * 8,
                        intermediate_size=128, num_hidden_layers=1,
                        num_attention_heads=architecture["num_attention_heads"],
                        num_key_value_heads=architecture["num_key_value_heads"],
                        attn_implementation="sdpa")
    update_model_config(config, {"_attn_implementation": attention})
    model = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=4)
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device="cuda")
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
    set_ulysses_sequence_parallel_group(None)
    with torch.no_grad():
        reference = model(inputs, position_ids=positions, use_cache=False).logits.float()
    set_ulysses_sequence_parallel_group(dist.group.WORLD)

    def parallel(value):
        return model(value[:, rank * 2:(rank + 1) * 2],
                     position_ids=positions[:, rank * 2:(rank + 1) * 2], use_cache=False).logits

    def gather(value):
        pieces = [torch.empty_like(value) for _ in range(4)]
        dist.all_gather(pieces, value.detach())
        return torch.cat(pieces, dim=1).float()

    local = parallel(inputs)
    actual = gather(local)
    torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.002)
    changed = inputs.clone()
    changed[:, :4] = torch.tensor([9, 10, 11, 12], device="cuda")
    with torch.no_grad():
        altered = gather(parallel(changed))
    torch.testing.assert_close(actual[:, 4:], altered[:, 4:], rtol=0, atol=0)
    local.float().square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    if rank == 0:
        report = {"status": "passed", "ranks": 4, "attention": config._attn_implementation,
                  "sp4_vs_sp1_max_logit_delta": float((actual - reference).abs().max()),
                  "cross_sample_max_logit_delta": float((actual[:, 4:] - altered[:, 4:]).abs().max()),
                  "backward_finite": True,
                  "vllm_sleep_allocator": "allocation_and_sleep_wake_passed",
                  "scope": "tiny Qwen2 with QwQ head counts; actual VERL Ulysses forward/backward"}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
