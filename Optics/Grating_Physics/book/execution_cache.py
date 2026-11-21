"""Reuse successful execution only when all non-cosmetic inputs match."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

import yaml
from public_content import source_files

ROOT = Path(__file__).resolve().parents[1]
COSMETIC = {'assets/textbook.css', 'assets/favicon.svg', 'assets/favicon.ico'}


def notebooks(root):
    def walk(entries):
        for entry in entries:
            if str(entry.get('file', '')).endswith('.ipynb'):
                yield entry['file']
            yield from walk(entry.get('children', []))
    config = yaml.safe_load((root/'myst.yml').read_text(encoding='utf8'))
    return list(walk(config['project']['toc']))


def fingerprint(root=ROOT, private_root=None):
    root = Path(root)
    inputs = {}
    for path in source_files(root):
        name = path.relative_to(root).as_posix()
        if name in COSMETIC:
            continue
        data = path.read_bytes()
        if name == 'myst.yml':
            config = yaml.safe_load(data)
            config.get('site', {}).get('options', {}).pop('favicon', None)
            data = json.dumps(config, sort_keys=True).encode()
        elif path.suffix in {'.py', '.md', '.json', '.ipynb', '.txt', '.yml', '.svg', '.css'}:
            data = data.replace(b'\r\n', b'\n')
        inputs[name] = hashlib.sha256(data).hexdigest()
    # Binder is outside the book on CI/public checkouts, inside publishing templates privately.
    binder = root/'book/publishing/.binder/environment.yml'
    if not binder.exists():
        binder = root.parents[1]/'.binder/environment.yml'
    if binder.exists():
        inputs['binder'] = hashlib.sha256(binder.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    workflow = root/'book/publishing/.github/workflows/grating-book.yml'
    if not workflow.exists():
        workflow = root.parents[1]/'.github/workflows/grating-book.yml'
    if workflow.exists():
        inputs['deployment-workflow'] = hashlib.sha256(workflow.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if private_root is not None:
        private_root = Path(private_root)
        private_files = [private_root/'book/workflow.py']
        private_files += sorted((private_root/'supplement/tests').rglob('*.py'))
        for path in private_files:
            if path.is_file():
                inputs['private/'+path.relative_to(private_root).as_posix()] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    requirements = root/'book/requirements-tested.txt'
    versions = {}
    if requirements.exists():
        for line in requirements.read_text().splitlines():
            if '==' in line:
                name = line.split('==')[0]
                versions[name] = metadata.version(name)
    installed = sorted({(dist.metadata['Name'].lower().replace('_', '-'), dist.version)
                        for dist in metadata.distributions() if dist.metadata.get('Name')})
    record = dict(inputs=inputs, packages=versions, installed=installed, python=sys.version,
                  platform=sys.platform, machine=platform.machine())
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def receipt(root):
    return Path(root)/'_build/executed/validation-receipt.json'


def output_hashes(root):
    outputs = {}
    for name in notebooks(Path(root)):
        path = Path(root)/'_build/executed'/name
        if path.is_symlink():
            raise ValueError('Symlink in executed notebook cache')
        outputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return outputs


def valid(root=ROOT, private_root=None):
    try:
        saved = json.loads(receipt(root).read_text())
        return saved == dict(fingerprint=fingerprint(root, private_root), outputs=output_hashes(root))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def save(root=ROOT, private_root=None):
    path = receipt(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(fingerprint=fingerprint(root, private_root), outputs=output_hashes(root)), indent=2), encoding='utf8')
    temporary.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('key', 'check', 'save'))
    command = parser.parse_args().command
    if command == 'key':
        print(fingerprint())
    elif command == 'check':
        print('true' if valid() else 'false')
    else:
        save()
