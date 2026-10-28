"""Check shard/index integrity and hash the weights used for initialization."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inspect(directory: Path) -> dict:
    index = directory / "model.safetensors.index.json"
    if index.exists():
        weights = json.loads(index.read_text())["weight_map"]
        files = sorted(set(weights.values()))
    else:
        files, weights = ["model.safetensors"], None
    headers = {}
    for name in files:
        path = directory / name
        if Path(name).name != name:
            raise ValueError("Invalid shard filename in model index")
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 64 * 1024 * 1024:
                raise ValueError("Invalid safetensors header")
            header = json.loads(stream.read(length))
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        if path.stat().st_size != 8 + length + max(value["data_offsets"][1] for value in tensors.values()):
            raise ValueError(f"Truncated/invalid shard: {name}")
        headers[name] = tensors
    if weights is not None:
        actual = {tensor: shard for shard, tensors in headers.items() for tensor in tensors}
        if actual != weights:
            raise ValueError("Model index does not match tensor headers")
    return {"directory": str(directory.resolve()), "shards": files,
            "tensors": sum(len(header) for header in headers.values()),
            "weight_bytes": sum((directory / name).stat().st_size for name in files)}


def weight_hashes(directory: Path, certificate: Path | None = None) -> dict:
    checkpoint = inspect(directory)
    if certificate is None:
        return {name: file_hash(directory / name) for name in checkpoint["shards"]}
    record = json.loads(certificate.read_text())
    if record.get("status") != "validated" or record["directory"] != str(directory.resolve()):
        raise ValueError("Hash certificate belongs to a different checkpoint")
    stats = {name: {"size": (directory / name).stat().st_size,
                    "mtime_ns": (directory / name).stat().st_mtime_ns}
             for name in checkpoint["shards"]}
    if record["shard_stats"] != stats or set(record["sha256"]) != set(checkpoint["shards"]):
        raise ValueError("Checkpoint changed since hashing")
    return record["sha256"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path)
    args = parser.parse_args()
    result = inspect(args.model)
    with ThreadPoolExecutor(max_workers=2) as pool:
        hashes = list(pool.map(lambda name: file_hash(args.model / name), result["shards"]))
    result["sha256"] = dict(zip(result["shards"], hashes))
    result["shard_stats"] = {name: {"size": (args.model / name).stat().st_size,
                                    "mtime_ns": (args.model / name).stat().st_mtime_ns}
                             for name in result["shards"]}
    if args.official_manifest:
        manifest = json.loads(args.official_manifest.read_text())
        expected = {record["name"]: record["lfs"]["sha256"] for record in manifest["files"]
                    if record["name"].endswith(".safetensors")}
        if result["sha256"] != expected:
            raise ValueError("Downloaded weights do not match official LFS hashes")
        result["official_revision"] = manifest["revision"]
    result["status"] = "validated"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated {len(result['shards'])} shards / {result['tensors']} tensors; SHA256 checked", flush=True)


if __name__ == "__main__":
    main()
