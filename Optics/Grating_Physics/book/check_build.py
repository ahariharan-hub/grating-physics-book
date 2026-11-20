"""Check pilot scope, parsed dropdown maths, and public credential isolation."""
import json
from pathlib import Path
import re
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
    records.append(dict(page=path.name, dropdowns=len(details), dropdown_math_nodes=counts))
assert len(records) == len(PILOT), records
source_dropdowns = []
for name in PILOT:
    notebook = json.loads((ROOT / name).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            assert '<details' not in ''.join(cell['source']).lower(), (
                name, 'Use native MyST dropdowns so Answer mathematics is parsed.')
            source_dropdowns.extend(re.findall(r'^:::\{dropdown\}[^\n]*\n(.*?)^:::\s*$',
                                               ''.join(cell['source']), re.M | re.S))
assert sum(item['dropdowns'] for item in records) == len(source_dropdowns)
assert sum(count > 0 for item in records for count in item['dropdown_math_nodes']) >= sum(
    '$' in source for source in source_dropdowns), 'A source math dropdown lost its parsed math.'
archives = list((BUILD / 'public-pilot/_build/html/_public').glob('*.zip'))
assert len(archives) == 2, [path.name for path in archives]
for path in archives:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, path.name
scan_public(BUILD / 'public-pilot/_build/html')
output = dict(pilot_scope=f'{len(PILOT)} pages', dropdown_math=records, public_credentials='absent',
              downloadable_archives=len(archives),
              browser_visual_check='not covered by structural build validation')
(ROOT / 'book/build-validation.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
print(json.dumps(output, indent=2))
