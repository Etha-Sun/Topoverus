"""Build the pinned Lynette safety-checker candidate without legacy paths."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', help='Use cached Cargo dependencies')
    args = parser.parse_args()
    third_party = ROOT / 'third_party'
    record = json.loads((third_party / 'lynette-source.json').read_text())
    archive = third_party / record['archive']
    patch = third_party / record['patch']
    for path, expected in [(archive, record['archive_sha256']), (patch, record['patch_sha256'])]:
        if sha256(path) != expected:
            raise ValueError(f'Pinned checker input changed: {path.name}')
    source = ROOT / '.deps/lynette-safety-v4'
    if source.exists():
        raise FileExistsError(f'Refusing to overwrite {source}')
    with tarfile.open(archive) as contents:
        members = contents.getmembers()
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or not member.isfile():
                raise ValueError(f'Unexpected checker archive member: {member.name}')
        if {m.name for m in members} != set(record['source_sha256']):
            raise ValueError('Checker archive file list differs')
        source.mkdir(parents=True)
        for member in members:
            target = source / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents.extractfile(member).read())
            if sha256(target) != record['source_sha256'][member.name]:
                raise ValueError(f'Original checker source differs: {member.name}')
    subprocess.run(['patch', '--strip=1', '--batch', '--forward', '--input', str(patch)],
                   cwd=source, check=True)
    for name, expected in record['source_files_sha256'].items():
        if sha256(source / name) != expected:
            raise ValueError(f'Patched checker source differs: {name}')
    command = ['cargo', 'build', '--release', '--locked', '-j', '4']
    if args.offline:
        command.append('--offline')
    subprocess.run(command, cwd=source, check=True)
    binary = source / 'target/release/lynette'
    version = subprocess.check_output([str(binary), '--topoverus-safety-version'], text=True).strip()
    if version != record['safety_version']:
        raise ValueError('Checker capability version differs')
    (source / 'build-provenance.json').write_text(json.dumps({
        'safety_version': version, 'binary_sha256': sha256(binary),
        'source_files_sha256': record['source_files_sha256'], 'command': command,
    }, indent=2) + '\n')
    print(f'Built {version}: {binary}')


if __name__ == '__main__':
    main()
