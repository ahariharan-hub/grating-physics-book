"""Standalone plain scripts, portable ZIPs, and Matplotlib chapter controls."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from numpy.testing import assert_allclose
import pytest

ROOT=Path(__file__).resolve().parents[2]
COLLECTIONS=ROOT/'Wave_solver_py_scripts'
NAMES=[
    '0_scalar_foundations','1_plane_waves','2_grating_orders','3_phase_matching',
    '4_evanescent_orders','5_diffraction_coordinates','6_field_to_intensity',
    '7_interference','8_field_propagation','8_3a_scalar_asm_propagation',
    '8_3b_talbot_propagation','A1_1_scalar_fmm_grating','A1_2_absorbing_fmm_grating',
    'A1_3_fmm_convergence','A1_4_grating_parameter_sweep','A2_1_finite_grating_propagation',
]


CATALOGUE=json.loads((COLLECTIONS/'scripts/worked_examples.json').read_text(encoding='utf-8'))
SPECIALIST_NAMES=[name for name in NAMES if name.startswith(('A1_', 'A2_', '8_3'))]
PLAIN_FILES=[item['file'] for item in CATALOGUE]+[name+'.py' for name in SPECIALIST_NAMES]


def load_plain(name):
    sys.path.insert(0,str(COLLECTIONS/'scripts'))
    spec=importlib.util.spec_from_file_location('plain_'+name,COLLECTIONS/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def extracted(tmp_path_factory):
    root=tmp_path_factory.mktemp('download_bundles')
    for collection in ('numerical_explorer','scripts'):
        with zipfile.ZipFile(ROOT/'downloads'/f'grating_physics_{collection}.zip') as archive:
            archive.extractall(root/collection)
    return root


@pytest.mark.parametrize('name',SPECIALIST_NAMES)
def test_all_plain_scripts_run_from_their_own_download_without_numerical_explorers(name,extracted):
    course=extracted/'scripts/Grating_Physics'
    assert not (course/'Wave_solver_py_scripts/numerical_explorer').exists()
    path=course/'Wave_solver_py_scripts/scripts'/f'{name}.py'
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8',
         'OPENBLAS_NUM_THREADS':'1','MPLBACKEND':'Agg'}
    env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(path),'--no-show'],cwd=extracted,
                          capture_output=True,text=True,env=env,timeout=60)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()


def test_numerical_download_runs_all_chapter_8_examples_independently(extracted):
    course=extracted/'numerical_explorer/Grating_Physics'
    assert not (course/'Wave_solver_py_scripts/scripts').exists()
    path=course/'Wave_solver_py_scripts/numerical_explorer/8_field_propagation.py'
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1'}
    env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(path),'--headless','--all'],cwd=extracted,
                          capture_output=True,text=True,env=env,timeout=60)
    assert result.returncode==0,result.stderr
    assert 'Weyl' in result.stdout and 'Selected |H_ASM|' in result.stdout


def test_plain_script_supports_ide_execution_with_an_unrelated_working_directory(extracted):
    path=extracted/'scripts/Grating_Physics/Wave_solver_py_scripts/scripts/chapter_1/1_1_01_time_oscillation.py'
    command=("from pathlib import Path; import sys; p=Path(sys.argv[1]); "
             "sys.argv=[str(p),'--no-show']; "
             "exec(compile(p.read_text(encoding='utf-8'),str(p),'exec'), "
             "dict(__name__='__main__',__file__=str(p)))")
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8','MPLBACKEND':'Agg'}
    env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,'-c',command,str(path)],cwd=extracted,
                          capture_output=True,text=True,env=env,timeout=30)
    assert result.returncode==0,result.stderr
    assert '1.1 -' in result.stdout


def test_archive_manifest_and_numbering_match_current_sources():
    manifest=json.loads((ROOT/'downloads/manifest.json').read_text())
    for record in manifest:
        path=ROOT/'downloads'/record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        expected=PLAIN_FILES if record['file']=='grating_physics_scripts.zip' else [name+'.py' for name in NAMES]
        assert set(record['entrypoints'])==set(expected)
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            for entry in archive.namelist():
                assert entry.startswith('Grating_Physics/') and '..' not in Path(entry).parts
                assert not any(part in {'.venv','__pycache__','tmp'} for part in Path(entry).parts)
                relative=Path(entry).relative_to('Grating_Physics')
                if relative.parts[0] in {'wave_optics','Wave_solver_py_scripts'}:
                    assert archive.read(entry)==(ROOT/relative).read_bytes(),entry
    page=(ROOT/'Downloads.md').read_text(encoding='utf-8')
    for record in manifest:assert 'downloads/'+record['file'] in page


def test_plain_sources_have_no_explorer_or_widget_imports():
    for filename in PLAIN_FILES:
        module=ast.parse((COLLECTIONS/'scripts'/filename).read_text(encoding='utf-8'))
        imports=[]
        for node in ast.walk(module):
            if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
            elif isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
        assert not any('_chapter' in value or '_playground' in value or 'matplotlib.widgets' in value for value in imports)


def test_plain_uniform_material_has_unit_transmission():
    app=load_plain('A1_1_scalar_fmm_grating')
    app.n_ridge=app.n_groove=app.n_incident=app.n_output=1.
    solution=app.compute()['solution']
    assert_allclose(solution.power_balance,1,atol=1e-12)
    assert_allclose(solution.transmission_efficiency[solution.orders==0],1,atol=1e-12)


def test_plain_absorber_balances_power():
    app=load_plain('A1_2_absorbing_fmm_grating')
    assert_allclose(app.compute()['solution'].energy_balance,1,atol=1e-8)


def test_plain_asm_identity_and_fresnel_talbot_recurrence():
    app=load_plain('8_3a_scalar_asm_propagation');app.distance=0
    result=app.compute()
    assert_allclose(result['asm'],result['boundary'],atol=1e-12)
    app=load_plain('8_3b_talbot_propagation');app.maximum_talbot_periods=1.;app.depth_samples=3
    result=app.compute();carrier=np.exp(2j*np.pi*result['talbot']/(app.wavelength/app.medium_index))
    assert_allclose(result['fields']['fresnel'][-1]/carrier,result['boundary'],atol=1e-10)


@pytest.mark.parametrize('chapter',range(9))
def test_chapter_sliders_apply_reset_and_keep_control_axes(chapter):
    import matplotlib
    matplotlib.use('Agg')
    sys.path.insert(0,str(COLLECTIONS/'numerical_explorer'))
    from _chapter_examples import LESSONS,parameters,CONTROLS
    from _chapter_models import compute,draw
    from _chapter_playground import controls_for
    from _playground import Explorer
    spec=LESSONS[chapter][0];p=parameters(spec);n=len(compute(spec,p)['panels'])
    key=spec.guided[0];control=CONTROLS[key]
    def calculation(settings):
        result=compute(spec,settings);result['caption']=spec.title;return result
    explorer=Explorer(p,controls_for(spec)[:1],calculation,draw,((n+1)//2,min(n,2)),spec.title)
    try:
        population=len(explorer.figure.axes)
        slider_axis=explorer.sliders[key].ax
        changed=p[key]+control.step
        explorer.sliders[key].set_val(changed)
        assert explorer.parameters[key]==p[key]
        explorer.update();assert explorer.parameters[key]==changed
        explorer.restore();assert explorer.parameters[key]==p[key]
        assert slider_axis in explorer.figure.axes
        assert len(explorer.figure.axes)==population
        explorer.figure.canvas.draw()
    finally:explorer.close()
