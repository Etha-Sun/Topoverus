"""Export matched Verus-only or strict-safe results and their input provenance."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

from topoverus.data import read_jsonl
from topoverus.evaluate import metrics


def benchmark_summary(tasks, records, samples, iterations, strict=True):
    groups = {}
    attempts = {}
    for record in records:
        attempts.setdefault(record['source_id'], []).append(record)
    for task in tasks:
        parts = task['source_id'].split('/')
        label = '/'.join(parts[:2]) if parts[0] == 'cleaned-verusbench' else parts[0]
        groups.setdefault(label, []).append(task)
    return {label: metrics(subset, [record for task in subset for record in attempts.get(task['source_id'], [])],
                           samples, iterations, [1, 3], strict=strict)['groups']['overall']
            for label, subset in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft', type=Path, required=True, help='Completed SFT evaluation run')
    parser.add_argument('--rl', type=Path, required=True, help='Completed Dep-v2 evaluation run')
    parser.add_argument('--dep', type=Path, help='Completed historical Dep evaluation run')
    parser.add_argument('--tasks', type=Path, help='Exact evaluation tasks; export separate benchmark breakdown')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, benchmark_rows, provenance = [], [], {}
    runs = [('clean_ctx_sft', args.sft), ('dep_v2', args.rl)]
    if args.dep:
        runs.append(('historical_dep', args.dep))
    for model, run in runs:
        summary_path = run / 'metrics/summary.json'
        summary = json.loads(summary_path.read_text())
        manifest = json.loads((run / 'manifest.json').read_text())
        if manifest['status'] != 'evaluated':
            raise ValueError(f'Evaluation is incomplete: {run}')
        strict = manifest.get('safety_policy') != 'verus_only'
        metric = 'strict_safe_pass' if strict else 'verified_pass'
        prefix = 'strict' if strict else 'verus'
        provenance[model] = {'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                             'safety_policy': manifest.get('safety_policy', 'strict_safe_v1_trust_counts'),
                             'tasks_sha256': manifest['tasks_sha256'], 'tools_sha256': manifest['tools_sha256'],
                             'source_sha256': manifest['source_sha256'],
                             'samples_per_task': summary['samples_per_task'], 'iterations': summary['iterations']}
        for key in ['seed', 'max_prompt_tokens', 'max_response_tokens', 'inference_dtype', 'frameworks']:
            provenance[model][key] = manifest[key]
        for key in ['checkpoint', 'checkpoint_hash_certificate_sha256', 'inference_provenance', 'adaptive_protocol_certificate']:
            if key in manifest:
                provenance[model][key] = manifest[key]
        if 'checkpoint' in provenance[model]:
            provenance[model]['checkpoint'] = {key: value for key, value in provenance[model]['checkpoint'].items()
                                               if key != 'directory'}
        if 'inference_provenance' in provenance[model] and 'checkpoint' in provenance[model]['inference_provenance']:
            provenance[model]['inference_provenance']['checkpoint'] = {
                key: value for key, value in provenance[model]['inference_provenance']['checkpoint'].items()
                if key != 'directory'}
        with tarfile.open(run / 'config/source.tar.gz') as archive:
            provenance[model]['guard_sha256'] = {
                name: hashlib.sha256(archive.extractfile('topoverus/' + name).read()).hexdigest()
                for name in ['trust.py', 'tools.py', 'evaluate.py']}
        groups = [(rows, summary['groups'])]
        if args.tasks:
            if hashlib.sha256(args.tasks.read_bytes()).hexdigest() != manifest['tasks_sha256']:
                raise ValueError('Benchmark task file differs from the evaluated input')
            breakdown = benchmark_summary(read_jsonl(args.tasks), read_jsonl(run / 'outputs/attempts.jsonl'),
                                          summary['samples_per_task'], summary['iterations'], strict=strict)
            groups.append((benchmark_rows, breakdown))
        for destination, group_values in groups:
            for group, values in sorted(group_values.items()):
                destination.append({'model': model, 'safety_policy': provenance[model]['safety_policy'], 'group': group, 'tasks': values['tasks'],
                                    f'{prefix}_pass1_estimate': values[f'{metric}:unbiased_pass@1'],
                                    f'{prefix}_first_chain_success': values[f'{metric}:first_1_success'],
                                    f'{prefix}_pass3': values[f'{metric}:unbiased_pass@3'],
                                    'solved_tasks': round(values[f'{metric}:unbiased_pass@3'] * values['tasks'])})
    for comparison in list(provenance.values())[1:]:
        for key in ['tasks_sha256', 'tools_sha256', 'samples_per_task', 'iterations', 'safety_policy',
                    'seed', 'max_prompt_tokens', 'max_response_tokens', 'inference_dtype', 'frameworks', 'guard_sha256']:
            if provenance['clean_ctx_sft'][key] != comparison[key]:
                raise ValueError(f'Evaluation protocols differ: {key}')
    for filename, output_rows in [('evaluation.csv', rows), ('benchmarks.csv', benchmark_rows)]:
        if not output_rows:
            continue
        with (args.output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    (args.output / 'evaluation-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
