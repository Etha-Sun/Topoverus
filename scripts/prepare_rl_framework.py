"""Rebuild the exact training framework from a public archive and local patch."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="Use an already downloaded upstream archive")
    args = parser.parse_args()
    record = json.loads((ROOT / "third_party/verl-source.json").read_text())
    patch = ROOT / "third_party" / record["patch"]
    if sha256(patch) != record["patch_sha256"]:
        raise ValueError("Framework patch changed")
    source = ROOT / ".deps/verl"
    if source.exists():
        raise FileExistsError(f"Refusing to overwrite {source}")
    archive = args.archive or ROOT / ".cache/verl-v0.6.0.tar.gz"
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(record["upstream_archive_url"], timeout=60) as response:
            archive.write_bytes(response.read())
    if sha256(archive) != record["upstream_archive_sha256"]:
        raise ValueError("Framework archive does not match the pinned SHA256")
    source.mkdir(parents=True)
    with tarfile.open(archive) as contents:
        prefix = contents.getmembers()[0].name.split("/")[0]
        for member in contents.getmembers():
            relative = Path(member.name).relative_to(prefix)
            if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
            if member.isfile():
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents.extractfile(member).read())
    subprocess.run(["patch", "--strip=1", "--batch", "--forward", "--input", str(patch)],
                   cwd=source, check=True, stdout=subprocess.DEVNULL)
    for name, expected in record["source_files_sha256"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Reconstructed source differs: {name}")
    if any((source / name).exists() for name in record["removed_paths"]):
        raise ValueError("Unexpected upstream source remains")
    print(f"Reconstructed {len(record['source_files_sha256'])} validated source files at {source}")
    print("Install in the RL environment: python -m pip install --no-deps -e .deps/verl")


if __name__ == "__main__":
    main()
