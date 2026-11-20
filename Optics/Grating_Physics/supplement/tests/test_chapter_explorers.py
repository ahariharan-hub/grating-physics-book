"""Physical checks for the chapter numerical playgrounds."""
import importlib.util
import os
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Wave_solver_py_scripts/numerical_explorer'))
from _chapter_examples import LESSONS, parameters
from _chapter_models import compute, draw

CASES = [(chapter, index, spec) for chapter, specs in enumerate(LESSONS)
         for index, spec in enumerate(specs)]


@pytest.mark.parametrize('chapter,index,spec', CASES,
                         ids=[f'{c}-{i}-{s.model}' for c,i,s in CASES])
def test_every_lesson_computes_and_renders(chapter,index,spec):
    result=compute(spec,parameters(spec))
    assert result['status'] and result['panels']
    figure=Figure(figsize=(12,7),layout='constrained');FigureCanvasAgg(figure)
    draw(result,figure);figure.canvas.draw()
    assert figure.axes


def test_helmholtz_accepts_both_directions_and_rejects_wrong_wavenumber():
    spec=LESSONS[0][0];p=parameters(spec)
    for ratio in (-1,1):
        p['ratio']=ratio
        assert_allclose(compute(spec,p)['values']['residual'],0,atol=1e-12)
    p['ratio']=.5
    assert np.max(abs(compute(spec,p)['values']['residual']))>1


@pytest.mark.parametrize('family',range(8))
def test_solution_families_render_finite_fields(family):
    spec=LESSONS[0][1];p=parameters(spec);p['family']=family
    result=compute(spec,p)
    for panel in result['panels']: assert np.all(np.isfinite(panel['values']))


def test_material_reference_and_reconstruction():
    spec=LESSONS[0][5];p=parameters(spec)
    first=compute(spec,p)['values'];p['distance']+=.4
    second=compute(spec,p)['values'];sol=first['solution']
    prop=sol.output_kinds=='propagating'
    assert_allclose(abs(first['coefficients'][prop]),abs(second['coefficients'][prop]),atol=1e-12)
    ev=sol.output_kinds=='evanescent'
    assert np.all(abs(second['coefficients'][ev])<=abs(first['coefficients'][ev])+1e-14)
    assert abs(sol.energy_balance-1)<1e-8


def test_evanescent_amplitude_decays_without_normal_flux():
    spec=LESSONS[4][0];v=compute(spec,parameters(spec))['values']
    assert v['status']=='evanescent' and v['q'].real==0
    assert_allclose(abs(v['field'][0]),1,atol=1e-12)
    assert np.max(abs(v['field'][-1]))<1e-10


def test_detector_uniform_intensity_despite_spatial_phase():
    spec=LESSONS[6][1];v=compute(spec,parameters(spec))['values']['field']
    assert np.std(v.real)>.1
    assert_allclose(abs(v)**2,1,atol=1e-12)


def test_integer_exposure_averages_to_half_amplitude_squared_for_every_phase():
    spec=LESSONS[6][0];p=parameters(spec);p['cycles']=2.;p['amplitude']=1.7
    for phase in (-130,0,73):
        p['phase']=phase
        assert_allclose(compute(spec,p)['values']['mean'][-1],1.7**2/2,atol=1e-12)


def test_coherence_and_order_spacing_have_distinct_effects():
    spec=LESSONS[7][1];p=parameters(spec);p['coherence']=0
    v=compute(spec,p)['values']
    assert_allclose(v['intensity'],2,atol=1e-12)
    p.update(coherence=1,order=2)
    v=compute(spec,p)['values']
    assert_allclose(v['fringe_period'],p['period']/2)
    assert_allclose(v['visibility'],1)


def test_rs_patch_sum_matches_on_axis_detector_value():
    spec=LESSONS[8][1];v=compute(spec,parameters(spec))['values']
    assert_allclose(v['field'][len(v['y'])//2],v['on_axis'],atol=1e-12)


def test_weyl_reconstructs_complex_spherical_field():
    spec=LESSONS[8][2];p=parameters(spec);p['weyl_order']=48
    assert compute(spec,p)['values']['error']<1e-6


def test_asm_and_fresnel_treat_evanescent_components_differently():
    spec=LESSONS[8][4];v=compute(spec,parameters(spec))['values']
    assert abs(v['asm'][-1])<1e-10
    assert_allclose(abs(v['fresnel']),1,atol=1e-12)


def test_bad_sampling_reports_actionable_error():
    spec=LESSONS[8][-1];p=parameters(spec);p['samples']=128;p['window']=48
    with pytest.raises(ValueError,match='FFT samples'): compute(spec,p)


@pytest.mark.parametrize('filename',[
    '0_scalar_foundations','1_plane_waves','2_grating_orders',
    '3_phase_matching','4_evanescent_orders','5_diffraction_coordinates',
    '6_field_to_intensity','7_interference','8_field_propagation'])
def test_launchers_have_no_import_side_effects(filename):
    spec=importlib.util.spec_from_file_location(filename,ROOT/'Wave_solver_py_scripts/numerical_explorer'/f'{filename}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert callable(module.compute_example) and not hasattr(module,'explorer')


def test_headless_launcher_from_another_working_directory(tmp_path):
    completed=subprocess.run([sys.executable,str(ROOT/'Wave_solver_py_scripts/numerical_explorer'/'8_field_propagation.py'),
                              '--headless','--example','5'],cwd=tmp_path,
                             capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    assert 'Selected |H_ASM|' in completed.stdout


