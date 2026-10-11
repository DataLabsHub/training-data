#!/usr/bin/env python3
"""Reproducible EdgeBench source inventory for this integrated baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PRODUCTION_ROOTS = (
    'celery',
    'components/kombu/kombu',
    'components/billiard/billiard',
    'components/vine/vine',
)
CRITICAL_ROOTS = {
    'celery-app': ('celery/app',),
    'celery-worker': ('celery/worker',),
    'celery-backends': ('celery/backends',),
    'celery-canvas-result-scheduler': (
        'celery/canvas.py', 'celery/result.py', 'celery/beat.py',
        'celery/schedules.py',
    ),
    'celery-events': ('celery/events',),
    'kombu-runtime': ('components/kombu/kombu',),
    'billiard-runtime': ('components/billiard/billiard',),
}


def text_lines(path):
    try:
        return len(path.read_text(encoding='utf-8').splitlines())
    except (UnicodeDecodeError, OSError):
        return 0


def python_files(root, item):
    path = root / item
    if path.is_file():
        return [path] if path.suffix == '.py' else []
    return sorted(candidate for candidate in path.rglob('*.py')
                  if '__pycache__' not in candidate.parts)


def inventory(root, items):
    files = []
    for item in items:
        files.extend(python_files(root, item))
    unique = sorted(set(files))
    return {'files': len(unique), 'lines': sum(text_lines(path) for path in unique)}


def excluded(root):
    groups = {
        'tests': ('t', 'components/kombu/t', 'components/billiard/t',
                  'components/vine/t'),
        'fixtures': ('t/data', 'components/kombu/t/data',
                     'components/billiard/t/data', 'components/vine/t/data'),
        'vendor': ('vendor', 'third_party', 'node_modules'),
        'generated': ('build', 'dist', 'coverage', '.tox', '.pytest_cache'),
    }
    result = {}
    for name, paths in groups.items():
        files = []
        for value in paths:
            path = root / value
            if path.exists():
                files.extend(candidate for candidate in path.rglob('*')
                             if candidate.is_file())
        result[name] = {
            'files': len(set(files)),
            'lines': sum(text_lines(path) for path in set(files)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repo', type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    all_files = [path for path in root.rglob('*')
                 if path.is_file() and '.git' not in path.parts]
    report = {
        'counter': 'metrics/count_source.py version 1 (Python line counter)',
        'repo': str(root),
        'raw': {
            'files': len(all_files),
            'lines': sum(text_lines(path) for path in all_files),
        },
        'effective_production': inventory(root, PRODUCTION_ROOTS),
        'production_by_source': {
            'celery': inventory(root, ('celery',)),
            'kombu': inventory(root, ('components/kombu/kombu',)),
            'billiard': inventory(root, ('components/billiard/billiard',)),
            'vine': inventory(root, ('components/vine/vine',)),
        },
        'critical_path': {
            name: inventory(root, paths)
            for name, paths in CRITICAL_ROOTS.items()
        },
        'excluded': excluded(root),
        'rules': {
            'included': list(PRODUCTION_ROOTS),
            'extensions': ['.py'],
            'excluded_concepts': [
                'tests', 'fixtures', 'docs', 'examples', 'benchmarks',
                'vendor', 'generated output', 'build output', 'lock files',
            ],
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
