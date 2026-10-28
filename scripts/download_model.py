"""Download the official original model with a pinned source revision."""

import argparse
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="976055f8c83f394f35dbd3ab09a285a984907bd0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = root / "runs/model-download"
    record.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root / ".cache/huggingface")
    info = HfApi().model_info("Qwen/QwQ-32B", revision=args.revision, files_metadata=True)
    patterns = ["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md", "LICENSE"]
    size = sum(item.size or 0 for item in info.siblings if item.rfilename.endswith(".safetensors"))
    free = shutil.disk_usage(output).free
    if free < size + 10 * 1024**3:
        raise RuntimeError(f"Insufficient space: weights={size}, free={free}")
    manifest = {"repository": "Qwen/QwQ-32B", "revision": info.sha,
                "output": str(output), "weight_bytes": size, "free_bytes_before": free,
                "files": [{"name": item.rfilename, "bytes": item.size,
                           "lfs": item.lfs} for item in info.siblings], "status": "downloading"}
    (record / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"Downloading Qwen/QwQ-32B revision={info.sha} bytes={size} to {output}", flush=True)
    snapshot_download("Qwen/QwQ-32B", revision=info.sha, local_dir=output,
                      allow_patterns=patterns, max_workers=4)
    manifest["status"] = "downloaded"
    (record / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (output / "topoverus_source.json").write_text(json.dumps({
        "repository": manifest["repository"], "revision": info.sha}, indent=2) + "\n")
    print("Official original QwQ-32B download complete", flush=True)


if __name__ == "__main__":
    main()
