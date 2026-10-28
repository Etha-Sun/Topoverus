"""Export code, paper results, model links and case studies without a commit."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('topoverus', 'scripts', 'configs', 'tests', 'third_party', 'docs/case-studies', 'assets')
PUBLIC_FILES = (
    'README.md', 'pyproject.toml', '.gitignore', 'LICENSE',
    'docs/TRAINING.md', 'docs/DATA.md', 'docs/MODELS.md', 'docs/SAFETY.md',
    'docs/LICENSING.md', 'docs/model-cards/dep-v2.md',
    'results/README.md', 'results/paper-results.csv',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use runs/ or a separate directory for the publication snapshot')
    files = [ROOT / name for name in PUBLIC_FILES]
    files.extend(path for name in DIRECTORIES for path in (ROOT / name).rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.pyc', '.pyo'])
    files.extend(path for name in ['NOTICE', 'CITATION.cff'] if (path := ROOT / name).exists())
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sorted(files):
        relative = path.relative_to(ROOT)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    record = {'files': len(hashes), 'sha256': hashes,
              'project_license_present': (output / 'LICENSE').exists(),
              'scope': 'code, configs, tests, dependencies, usage docs, paper results, model links and case studies; no model weights, datasets, local audits, runs or credentials'}
    output.with_name(output.name + '-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Prepared {len(hashes)} publication files in {output}')


if __name__ == '__main__':
    main()
