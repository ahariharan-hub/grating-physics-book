"""Coverage, notebook references and independent execution of worked scripts."""
import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import nbformat
import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/'Wave_solver_py_scripts/scripts'
CATALOGUE=json.loads((SCRIPTS/'worked_examples.json').read_text(encoding='utf-8'))


def test_four_distinct_implementations_per_teaching_notebook():
    counts=Counter(item['notebook'] for item in CATALOGUE)
    assert len(CATALOGUE)==140 and len(counts)==35
    assert set(counts.values())=={4}
    assert len({item['file'] for item in CATALOGUE})==140
    sections=('INPUT PARAMETERS','SETUP','NUMERICAL CALCULATION','CHECK AND INTERPRET','GRAPHICS','NUMERICAL SUMMARY')
    for item in CATALOGUE:
        path=SCRIPTS/item['file'];source=path.read_text(encoding='utf-8')
        assert path.name.startswith(item['notebook'].replace('.','_')+f'_{item["number"]:02d}_')
        for section in sections:assert '# %% '+section in source
        tree=ast.parse(source)
        assert any(isinstance(node,ast.Assert) for node in ast.walk(tree))
        assert not any(isinstance(node,ast.ImportFrom) and ('_chapter' in (node.module or '') or '_playground' in (node.module or '')) for node in ast.walk(tree))
        assert 'exec(' not in source and 'runpy' not in source
    assert not list(SCRIPTS.glob('[0-9]_*.py')) or all(p.name.startswith('8_3') for p in SCRIPTS.glob('[0-9]_*.py'))


def test_notebooks_link_their_own_four_scripts_and_keep_valid_notebook_structure():
    for book in sorted({item['notebook'] for item in CATALOGUE}):
        path=next(ROOT.glob(book.replace('.','_')+'_*.ipynb'))
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        cells=[cell for cell in notebook.cells if cell.get('id')=='worked-examples-'+book.replace('.','-')]
        assert len(cells)==1
        assert cells[0].cell_type=='markdown' and '## Worked-out examples' in cells[0].source
        expected=[item for item in CATALOGUE if item['notebook']==book]
        links=re.findall(r'\]\((Wave_solver_py_scripts/scripts/[^)]+\.py)\)',cells[0].source)
        assert set(links)=={'Wave_solver_py_scripts/scripts/'+item['file'] for item in expected}
        assert all((ROOT/link).is_file() for link in links)
    index=(SCRIPTS/'INDEX.md').read_text(encoding='utf-8')
    for item in CATALOGUE:assert ']('+item['file']+')' in index
    for chapter in range(10):
        notebook=nbformat.read(next(ROOT.glob(f'{chapter}_0_*.ipynb')),as_version=4)
        assert any(cell.get('id')==f'worked-examples-index-{chapter}' for cell in notebook.cells)


@pytest.fixture(scope='module')
def extracted(tmp_path_factory):
    destination=tmp_path_factory.mktemp('worked_download')
    with zipfile.ZipFile(ROOT/'downloads/grating_physics_scripts.zip') as archive:
        archive.extractall(destination)
    return destination


@pytest.mark.parametrize('chapter',range(10))
def test_every_worked_script_calculates_checks_and_plots_in_the_extracted_bundle(chapter,extracted):
    folder=extracted/'Grating_Physics/Wave_solver_py_scripts/scripts'/f'chapter_{chapter}'
    assert len(list(folder.glob('*.py')))==sum(int(item['notebook'].split('.')[0])==chapter for item in CATALOGUE)
    command='''import os,runpy,sys
from pathlib import Path
for path in sorted(Path(sys.argv[1]).glob('*.py')):
    print('RUN',path.name,flush=True)
    sys.argv=[str(path),'--no-show']
    runpy.run_path(str(path),run_name='__main__')
print('CHAPTER PASSED',flush=True)
'''
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','MPLBACKEND':'Agg'}
    env.pop('PYTHONPATH',None)
    completed=subprocess.run([sys.executable,'-c',command,str(folder)],cwd=extracted,
                             capture_output=True,text=True,env=env,timeout=120)
    assert completed.returncode==0,completed.stdout[-2500:]+'\n'+completed.stderr
    assert completed.stdout.rstrip().endswith('CHAPTER PASSED')


def test_rs_example_supports_explicit_figure_export(extracted,tmp_path):
    path=extracted/'Grating_Physics/Wave_solver_py_scripts/scripts/chapter_8/8_2_01_rectangular_rs_integral.py'
    output=tmp_path/'figures'
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','MPLBACKEND':'Agg'}
    env.pop('PYTHONPATH',None)
    completed=subprocess.run([sys.executable,str(path),'--no-show','--save-dir',str(output)],
                             cwd=tmp_path,capture_output=True,text=True,env=env,timeout=60)
    assert completed.returncode==0,completed.stderr
    assert (output/'8_2_01_rectangular_rs_integral.png').stat().st_size>1000
