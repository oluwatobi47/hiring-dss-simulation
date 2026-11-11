#!/usr/bin/env python3
"""Judge each selected configuration separately; no inference servers required."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from model_comparison_registry import CONFIGURATIONS, TERRA_MODEL, comparison_manifest

ROOT = Path(__file__).resolve().parent


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/simulation/output/comparison",
                        help="Parent of experiment directories, not the experiment directory itself.")
    parser.add_argument("--configuration", action="append", choices=list(CONFIGURATIONS))
    parser.add_argument("--model", default=TERRA_MODEL, help="Judge model, not the model being evaluated.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=600)
    parser.add_argument("--verbose-openai", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def preflight(runs, *, allow_missing=False, overwrite=False):
    """Check every selected input before importing the API adapter or making calls."""
    plans, errors = [], []
    for spec in runs:
        path, destination = spec['results'], spec['evaluation']
        files = sorted(path.glob('TC_OUTPUT_*.json'))
        plan = dict(spec, files=files, status='pending')
        plans.append(plan)
        if not files:
            if allow_missing:
                plan['status'] = 'skipped_missing'
            else:
                errors.append(f"{spec['configuration']}: no direct TC_OUTPUT_*.json files in {path}")
            continue
        if destination.exists() and not destination.is_dir():
            errors.append(f"{destination} is not a directory")
            continue
        existing = list(destination.iterdir()) if destination.exists() else []
        if existing and not overwrite:
            errors.append(f"{destination} is nonempty; use --overwrite explicitly")
        expected_names = {f.name.replace('TC_OUTPUT_', 'TC_EVAL_', 1) for f in files}
        stale = [f.name for f in destination.glob('TC_EVAL_*.json') if f.name not in expected_names]
        if stale:
            errors.append(f"{destination}: move stale evaluation files elsewhere first: {stale}")
        for file in files:
            try:
                rows = json.loads(file.read_text(encoding='utf-8'))
                if not isinstance(rows, list) or not rows:
                    raise ValueError('expected a nonempty record list')
                seen = set()
                case = file.stem.removeprefix('TC_OUTPUT_')
                for row in rows:
                    required = {'question_id', 'prompt', 'expected_output', 'model_response'}
                    if not isinstance(row, dict) or not required.issubset(row) or row['question_id'] is None:
                        raise ValueError('record missing required simulation fields')
                    if row.get('batch_id', case) != case:
                        raise ValueError('batch_id does not match test-case filename')
                    key = str(row['question_id'])
                    if key in seen:
                        raise ValueError('duplicate question ID within test case')
                    seen.add(key)
            except (ValueError, OSError) as exc:
                errors.append(f"{file}: {exc}")
    if errors:
        raise ValueError('Preflight failed:\n' + '\n'.join(errors))
    return plans


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.temperature < 0 or args.max_output_tokens <= 0:
        parser.error('Temperature must be nonnegative and token budget positive.')
    try:
        runs = comparison_manifest(args.output_root.resolve(), args.experiment)
        selected = set(args.configuration or CONFIGURATIONS)
        plans = preflight([r for r in runs if r['configuration'] in selected],
                          allow_missing=args.allow_missing, overwrite=args.overwrite)
    except ValueError as exc:
        parser.exit(2, str(exc) + '\n')
    for plan in plans:
        print(f"{plan['configuration']}: {plan['status']}; {len(plan['files'])} test-case files -> {plan['evaluation']}")
    print('Partial runs are evaluated as supplied; completeness is audited in the notebook.')
    if args.dry_run:
        return 0

    pending = [p for p in plans if p['status'] == 'pending']
    if pending and not os.environ.get('OPENAI_API_KEY'):
        parser.exit(2, 'Set OPENAI_API_KEY before evaluation.\n')
    # Lazy import allows dry-run without inference packages or API credentials.
    started = datetime.now(timezone.utc)
    report = dict(experiment=args.experiment, judge_model=args.model, temperature=args.temperature,
                  max_output_tokens=args.max_output_tokens, started_at_utc=started.isoformat(),
                  configurations=[dict(configuration=p['configuration'], status=p['status'],
                                       results_dir=str(p['results']), evaluation_dir=str(p['evaluation']),
                                       test_case_files=len(p['files'])) for p in plans])
    report_path = args.output_root.resolve() / args.experiment / ('BATCH_EVAL_' + started.strftime('%Y%m%dT%H%M%S%fZ') + '.json')

    def save_report():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    save_report()
    for index, plan in enumerate(plans):
        if plan['status'] != 'pending':
            continue
        state = report['configurations'][index]
        try:
            from evaluate_simulation_results import OpenAI, process_output_file, summarize, write_json
            llm = OpenAI(model=args.model, temperature=args.temperature,
                         max_tokens=args.max_output_tokens, verbose=args.verbose_openai)
            records, per_file = [], {}
            for file in plan['files']:
                print(f"Judging {plan['configuration']}: {file.name}", flush=True)
                rows = process_output_file(file, llm)
                destination = plan['evaluation'] / file.name.replace('TC_OUTPUT_', 'TC_EVAL_', 1)
                write_json(destination, rows)
                per_file[file.name] = dict(summarize(rows), evaluation_file=str(destination))
                records.extend(rows)
            write_json(plan['evaluation'] / 'EVAL_SUMMARY.json',
                       dict(results_dir=str(plan['results']), evaluation_dir=str(plan['evaluation']),
                            judge_model=args.model, files=per_file, overall=summarize(records)))
            state.update(status='completed', evaluated_rows=len(records))
            save_report()
        except (Exception, KeyboardInterrupt) as exc:
            # Exception messages may contain answer text or provider credentials.
            state.update(status='failed', error_type=type(exc).__name__)
            for later in report['configurations'][index + 1:]:
                if later['status'] == 'pending':
                    later['status'] = 'not_run_after_failure'
            report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            save_report()
            print(f"Evaluation stopped ({type(exc).__name__}). Report: {report_path}")
            return 1
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    save_report()
    print(f"Batch report: {report_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
