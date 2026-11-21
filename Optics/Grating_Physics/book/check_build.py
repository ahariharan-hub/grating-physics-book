"""Check pilot scope, parsed dropdown maths, and public credential isolation."""
import json
from pathlib import Path
import re
import zipfile

import yaml
from pilot import ROOT, BUILD, PILOT, PAGES, toc_files, scan_public
from public_content import check_readers, is_public_source


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


config = yaml.safe_load((ROOT / 'myst.yml').read_text(encoding='utf-8'))
check_readers(ROOT)
for path in (BUILD/'public-pilot').rglob('*'):
    if path.is_file() and '_build' not in path.relative_to(BUILD/'public-pilot').parts:
        assert is_public_source(path.relative_to(BUILD/'public-pilot')), ('Non-public staged source', path)
assert toc_files(config['project']['toc']) == PAGES
assert tuple(name for name in PAGES if Path(name).suffix == '.ipynb') == PILOT
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
assert len(records) == len(PAGES), records
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
layout = {}
style = config.get('site', {}).get('options', {}).get('style')
if style:
    html_root = BUILD / 'public-pilot/_build/html'
    css = (ROOT / style).read_bytes().replace(b'\r\n', b'\n')
    assert any(p.read_bytes().replace(b'\r\n', b'\n') == css for p in html_root.rglob('*.css')), 'Custom stylesheet missing from the built site.'
    homepage = (html_root / 'index.html').read_text(encoding='utf-8')
    assert 'textbook-hero' in homepage and 'Ashwin Hariharan' in homepage
    assert 'Fields and wave equations' in homepage and 'Physical gratings and orders' in homepage
    assert '0.7 Ideal masks and physical gratings' in homepage
    assert (html_root / 'foundations-of-scalar-optics/index.html').is_file(), 'Notebook 0.0 must remain accessible from the new homepage.'
    layout = dict(homepage='passed', nested_navigation='passed', custom_stylesheet='bundled')
scan_public(BUILD / 'public-pilot/_build/html')
output = dict(pilot_scope=f'{len(PAGES)} pages; {len(PILOT)} executed notebooks', dropdown_math=records, public_credentials='absent',
              downloadable_archives=len(archives), layout=layout,
              browser_visual_check='not covered by structural build validation')
(ROOT / 'book/build-validation.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
print(json.dumps(output, indent=2))
