"""Check pilot scope, parsed dropdown maths, and public credential isolation."""
import json
from pathlib import Path
import zipfile

import yaml
from pilot import ROOT, BUILD, PILOT, scan_public


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


config = yaml.safe_load((ROOT / 'myst.yml').read_text(encoding='utf-8'))
assert [entry['file'] for entry in config['project']['toc']] == list(PILOT)
assert 'server' not in config['project']['jupyter']
assert config['project']['jupyter']['binder'] == {
    'repo': 'ahariharan-hub/grating-physics-book', 'ref': 'main'}
assert config['project']['jupyter']['kernelName'] == 'python3'
content = BUILD / 'public-pilot/_build/site/content'
records = []
for path in content.glob('*.json'):
    page = json.loads(path.read_text(encoding='utf-8'))
    if 'mdast' not in page:
        continue
    details = [node for node in walk(page['mdast']) if node.get('type') == 'details']
    counts = [sum(node.get('type') in ('math', 'inlineMath') for node in walk(detail)) for detail in details]
    assert counts and all(count > 0 for count in counts), (path.name, counts)
    records.append(dict(page=path.name, dropdowns=len(details), dropdown_math_nodes=counts))
assert len(records) == len(PILOT), records
assert sum(item['dropdowns'] for item in records) == 8
archives = list((BUILD / 'public-pilot/_build/html/_public').glob('*.zip'))
assert len(archives) == 2, [path.name for path in archives]
for path in archives:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, path.name
scan_public(BUILD / 'public-pilot/_build/html')
output = dict(pilot_scope=f'{len(PILOT)} pages', dropdown_math=records, public_credentials='absent',
              downloadable_archives=len(archives),
              browser_visual_check='unverified: computer-use URL policy block')
(ROOT / 'book/build-validation.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
print(json.dumps(output, indent=2))
