"""Keep one HF result: convert final FP32 actor shards to BF16 in place."""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import shutil

from .checkpoint import file_hash, inspect


def cast_bf16(directory: Path) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint = inspect(directory)
    hashes, total_size, changed = {}, 0, 0
    for name in checkpoint["shards"]:
        path = directory / name
        with safe_open(path, framework="pt", device="cpu") as reader:
            tensors = {key: reader.get_tensor(key) for key in reader.keys()}
            attributes = reader.metadata()
        needs_cast = any(t.is_floating_point() and t.dtype != torch.bfloat16 for t in tensors.values())
        if needs_cast:
            tensors = {key: value.to(torch.bfloat16) if value.is_floating_point() else value
                       for key, value in tensors.items()}
            temporary = path.with_name("." + name + ".bf16.tmp")
            save_file(tensors, str(temporary), metadata=attributes)
            os.replace(temporary, path)
            changed += 1
        total_size += sum(value.numel() * value.element_size() for value in tensors.values())
        del tensors
        hashes[name] = file_hash(path)
        print(f"BF16 export {name}: complete", flush=True)
    index = directory / "model.safetensors.index.json"
    if index.exists():
        value = json.loads(index.read_text())
        value.setdefault("metadata", {})["total_size"] = total_size
        index.write_text(json.dumps(value, indent=2) + "\n")
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    # Different Transformers generations read different precision fields.
    config.update(dtype="bfloat16", torch_dtype="bfloat16")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    result = inspect(directory)
    result.update(status="validated", dtype="bfloat16", converted_shards=changed, sha256=hashes,
                  shard_stats={name: {"size": (directory / name).stat().st_size,
                                      "mtime_ns": (directory / name).stat().st_mtime_ns}
                               for name in result["shards"]})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    if not run.is_relative_to(root / "runs"):
        raise ValueError("Only export this release's RL runs")
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["stage"] != "dep" or manifest["status"] not in ("trained", "exporting"):
        raise ValueError("Export requires a completed Dep-v2 run")
    model = Path(manifest["final_checkpoint"]["directory"])
    if not model.resolve().is_relative_to(run / "checkpoints"):
        raise ValueError("Checkpoint does not belong to this run")
    previous = run / "config/trainer-manifest.json"
    if previous.exists():
        if manifest["status"] != "exporting":
            raise ValueError("This run has already been exported")
        if file_hash(run / "config/export_model.py") != file_hash(Path(__file__)):
            raise ValueError("Export implementation changed during an interrupted export")
    else:
        shutil.copyfile(path, previous)
        shutil.copyfile(Path(__file__), run / "config/export_model.py")
    manifest["status"] = "exporting"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    result = cast_bf16(model)
    result.update(trainer_manifest_sha256=file_hash(previous),
                  export_source_sha256=file_hash(Path(__file__)),
                  frameworks={name: metadata.version(name) for name in ("torch", "safetensors")})
    certificate = run / "checkpoint-validation.json"
    certificate.write_text(json.dumps(result, indent=2) + "\n")
    manifest.update(status="trained", final_checkpoint=inspect(model), final_checkpoint_sha256=result["sha256"],
                    export={"dtype": "bfloat16", "certificate": str(certificate),
                            "certificate_sha256": file_hash(certificate)})
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
