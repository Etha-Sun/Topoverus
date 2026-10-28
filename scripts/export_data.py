"""Prepare a hash-checked processed data bundle for local publication review."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from topoverus.graph import load_graphs


ROOT = Path(__file__).resolve().parents[1]
GRAPH_FIELDS = ('source_id', 'source_sha256', 'reference_sha256', 'builder_version',
                'invariants', 'dependency_edges', 'status')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.data.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('Review bundles must stay under release runs/')
    expected = json.loads((ROOT / 'configs/data-output-sha256.json').read_text())
    for name, sha in expected.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Data differs from the retained training inputs: {name}')
    graphs = load_graphs(source / 'graphs.jsonl')
    tasks = [json.loads(line) for line in (source / 'rl_train.jsonl').read_text().splitlines()]
    if set(graphs) != {row['source_id'] for row in tasks}:
        raise ValueError('Graph and training identities differ')
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / 'clean-v4'
    bundle.mkdir()
    for name in [*expected, 'audit.json', 'provenance.json']:
        shutil.copy2(source / name, bundle / name)
    with (bundle / 'graphs.jsonl').open('w', encoding='utf-8') as stream:
        for record in graphs.values():
            stream.write(json.dumps({key: record[key] for key in GRAPH_FIELDS}, ensure_ascii=False) + '\n')
    manifest = {'status': 'prepared_for_review_not_published',
                'prepared_files': {path.name: {'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(bundle.iterdir())},
                'sft_validation_policy': json.loads((bundle / 'provenance.json').read_text())['sft_safety_policy'],
                'graph_export': 'Preserves all training business fields; omits temporary diagnostics and paths',
                'graph_records': len(graphs), 'graph_fields': list(GRAPH_FIELDS),
                'code_license_does_not_cover_dataset': True,
                'distribution_status': 'Dataset access and upstream/derivative terms remain unresolved',
                'parquet': 'Convert rl_train.jsonl and heldout.jsonl after extraction; retain audit/provenance'}
    with tarfile.open(output / 'topoverus-clean-v4.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.iterdir()):
            archive.add(path, arcname='clean-v4/' + path.name)
    manifest['archive_sha256'] = hashlib.sha256((output / 'topoverus-clean-v4.tar.gz').read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {len(manifest["prepared_files"])} checked files in {output}; no upload performed')


if __name__ == '__main__':
    main()
