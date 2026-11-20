"""Physical benchmarks and controls for six standalone solver experiments."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
from unittest.mock import Mock,patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pytest

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'Wave_solver_py_scripts/numerical_explorer'
NAMES=('finite_grating_propagation','scalar_asm_propagation','absorbing_fmm_grating',
       'fmm_convergence','grating_parameter_sweep','talbot_propagation')
FILENAMES={'finite_grating_propagation':'A2_1_finite_grating_propagation',
 'scalar_asm_propagation':'8_3a_scalar_asm_propagation', 'absorbing_fmm_grating':'A1_2_absorbing_fmm_grating',
 'fmm_convergence':'A1_3_fmm_convergence', 'grating_parameter_sweep':'A1_4_grating_parameter_sweep',
 'talbot_propagation':'8_3b_talbot_propagation'}
MODULES={}
for name in NAMES:
    spec=importlib.util.spec_from_file_location('standalone_'+name,FOLDER/(FILENAMES[name]+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MODULES[name]=module
from _playground import Explorer


def small_parameters(name):
    p=dict(MODULES[name].parameters)
    if name=='finite_grating_propagation':
        p.update(fft_window=32.,display_half_width=6.,samples=4096,maximum_depth=2.,depth_samples=9,openings=3)
    elif name=='scalar_asm_propagation':
        p.update(window=16.,display_half_width=6.,samples=128,width=2.,height=3.,separation=4.,distance=3.)
    elif name=='absorbing_fmm_grating':p.update(M=6)
    elif name=='fmm_convergence':p.update(cutoffs=(1,3,5),reference_M=8,target_order=1)
    elif name=='grating_parameter_sweep':p.update(M=6,points=5,sweep_min=.02,sweep_max=.1)
    elif name=='talbot_propagation':p.update(samples=128,depth_samples=17)
    return p


def test_finite_array_has_requested_openings_and_reconstructs_its_boundary():
    app=MODULES['finite_grating_propagation'];p=small_parameters('finite_grating_propagation')
    p['illumination']='uniform'
    data=app.compute(p)
    assert np.count_nonzero(np.diff(np.r_[0,data['mask'],0])>0)==p['openings']
    assert_allclose(data['fields'][0],data['boundary'],atol=2e-13)
    assert abs(sum(data['mask'])*data['dx']-p['openings']*p['opening_width'])<=p['openings']*data['dx']
    assert not np.any(abs(data['orders'])==3)  # Zero of the 1/3-open ideal harmonic.


def test_2d_asm_boundary_identity_and_uniform_phase_invariance():
    app=MODULES['scalar_asm_propagation'];p=small_parameters('scalar_asm_propagation')
    p.update(aperture='gaussian',phase_shift=.4,distance=0.)
    data=app.compute(p)
    assert_allclose(data['asm'],data['boundary'],atol=3e-13)
    assert_allclose(data['fresnel'],data['boundary'],atol=3e-13)
    p['distance']=4.
    first=app.compute(p)
    p['phase_shift']+=np.pi/3
    second=app.compute(p)
    assert_allclose(second['asm'],first['asm']*np.exp(1j*np.pi/3),atol=3e-13)
    assert_allclose(abs(second['asm'])**2,abs(first['asm'])**2,atol=3e-13)


def test_absorber_air_limit_and_depth_only_reuses_scattering_solution():
    app=MODULES['absorbing_fmm_grating'];p=small_parameters('absorbing_fmm_grating')
    p.update(absorber_index=1.,kappa=0.)
    app.solve_cached.cache_clear()
    with patch.object(app,'solve_absorbing_grating',wraps=app.solve_absorbing_grating) as solve:
        first=app.compute(p)
        p['output_depth']+=.1
        second=app.compute(p)
        assert solve.call_count==1
    s=first['solution']
    assert_allclose([s.reflectance,s.transmittance,s.absorption,s.energy_balance],[0,1,0,1],atol=3e-12)
    assert first['solution'] is second['solution']
    assert_allclose(abs(second['profile']),1,atol=3e-12)


def test_absorber_volume_loss_closes_power_balance():
    app=MODULES['absorbing_fmm_grating'];s=app.compute(small_parameters('absorbing_fmm_grating'))['solution']
    assert s.absorption>0
    assert_allclose(s.reflectance+s.transmittance+s.absorption,1,atol=2e-8)


def test_convergence_detects_errors_that_flux_balance_misses():
    app=MODULES['fmm_convergence'];p=small_parameters('fmm_convergence')
    p.update(cutoffs=(3,5,8),reference_M=20)
    data=app.compute(p)
    assert data['errors'][0]>1e-3
    assert data['field_errors'][0]>1e-3
    assert max(data['balances'])<1e-10
    assert data['errors'][-1]==0 and data['field_errors'][-1]==0
    assert data['errors'][-2]<data['errors'][0]


def test_uniform_grating_sweep_has_unit_transmission_and_undefined_zero_order_phases():
    app=MODULES['grating_parameter_sweep'];p=small_parameters('grating_parameter_sweep')
    p.update(n_ridge=1.,n_groove=1.,orders=(0,1))
    data=app.compute(p)
    assert_allclose(data['coefficients'][:,0],1,atol=3e-12)
    assert_allclose(data['totals'],np.tile([0,1,1],(p['points'],1)),atol=3e-12)
    assert np.all(np.isnan(data['phases'][:,1]))
    p.update(sweep_min=-.1,sweep_max=.1)
    gaps=app.compute(p)
    assert len(gaps['failures'])==3
    assert np.all(np.isnan(gaps['coefficients'][:3]))


def test_talbot_fresnel_recurrence_and_half_period_shift():
    app=MODULES['talbot_propagation'];p=small_parameters('talbot_propagation')
    p.update(depth_samples=3,maximum_talbot_periods=1.,observation_fraction=1.)
    data=app.compute(p)
    lam=p['wavelength']/p['medium_index']
    carrier=np.exp(2j*np.pi/lam*data['talbot'])
    assert_allclose(data['selected_fresnel']/carrier,data['mask'],atol=2e-10)
    half_carrier=np.exp(1j*np.pi/lam*data['talbot'])
    assert_allclose(data['fresnel'][1]/half_carrier,np.roll(data['mask'],p['samples']//2),atol=2e-10)
    assert data['error_asm'][-1]>.01  # Sharp binary mask is not perfectly paraxial.


@pytest.mark.parametrize('name',NAMES)
def test_apply_reset_and_repeated_render_do_not_accumulate_axes_or_compute_while_dragging(name):
    app=MODULES[name];p=small_parameters(name)
    key,change={'finite_grating_propagation':('waist',.2),'scalar_asm_propagation':('phase_shift',.2),
                'absorbing_fmm_grating':('output_depth',.03),'fmm_convergence':('output_depth',.01),
                'grating_parameter_sweep':('sweep_max',.02),'talbot_propagation':('observation_fraction',.1)}[name]
    compute=Mock(wraps=app.compute)
    shape=(2,3) if name in ('scalar_asm_propagation','absorbing_fmm_grating') else (2,2)
    explorer=Explorer(p,[(key,key,0.,max(2.,p[key]+1),None)],compute,app.draw,shape,name)
    try:
        population=len(explorer.figure.axes)
        explorer.sliders[key].set_val(p[key]+change)
        assert compute.call_count==1
        explorer.update();assert compute.call_count==2
        explorer.update();assert compute.call_count==2
        assert len(explorer.figure.axes)==population
        explorer.restore();assert compute.call_count==3
        assert explorer.parameters==p
        assert len(explorer.figure.axes)==population
    finally:
        explorer.close();explorer.close()


@pytest.mark.parametrize('name',NAMES)
def test_standalone_headless_execution_from_another_directory_creates_no_output_files(name,tmp_path):
    command=("from pathlib import Path; import sys; p=Path(sys.argv[1]); "
             "source=p.read_text(encoding='utf-8').replace('interactive=True','interactive=False'); "
             "exec(compile(source,str(p),'exec'),dict(__name__='__main__',__file__=str(p)))")
    result=subprocess.run([sys.executable,'-c',command,str(FOLDER/(FILENAMES[name]+'.py'))],cwd=tmp_path,
        text=True,capture_output=True,timeout=120,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','MPLBACKEND':'Agg'})
    assert result.returncode==0,result.stderr
    assert list(tmp_path.iterdir())==[]
