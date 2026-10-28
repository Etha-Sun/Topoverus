"""Rebuild the release dataset from hash-pinned raw inputs and checked tools."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/clean-v4')
    parser.add_argument('--verus', required=True)
    parser.add_argument('--lynette', required=True)
    args = parser.parse_args()
    inputs = json.loads((ROOT / 'configs/data-inputs.json').read_text())
    command = [sys.executable, '-m', 'topoverus.data']
    for role, record in inputs.items():
        path = args.raw_dir.resolve() / record['filename']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Raw input differs from the retained run: {record["filename"]}')
        command.extend(['--' + role, str(path)])
    output = args.output.resolve()
    command.extend(['--output', str(output), '--verus', args.verus, '--lynette', args.lynette])
    subprocess.run(command, cwd=ROOT, check=True)
    for name, expected in json.loads((ROOT / 'configs/data-output-sha256.json').read_text()).items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Prepared dataset differs from the retained run: {name}')
    for name in ['rl_train', 'heldout']:
        subprocess.run([sys.executable, '-m', 'topoverus.parquet', '--input', str(output / (name + '.jsonl')),
                        '--output', str(output / (name + '.parquet'))], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'topoverus.build_graphs', '--tasks', str(output / 'rl_train.jsonl'),
                    '--output', str(output / 'graphs.jsonl'), '--verus', args.verus,
                    '--lynette', args.lynette], cwd=ROOT, check=True)
    print(f'Rebuilt and validated the release dataset at {output}')


if __name__ == '__main__':
    main()
