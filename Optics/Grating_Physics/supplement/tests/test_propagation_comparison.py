"""Complex normalization, full-2D propagation, approximations and live controls."""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import pytest
from numpy.testing import assert_allclose
from PIL import Image
from ipywidgets.widgets.widget import _instances

from wave_optics.propagation_comparison import (
    PERIOD, boundary_field, boundary_spectrum, approximate_field, asm_cut,
    comparison, comparison_explorer, phase_difference,
)
from wave_optics.scalar_propagation import propagate_field


def test_analytic_fresnel_matches_existing_2d_fft_in_complex_field():
    x = (np.arange(512)-256)*.25; xx, yy = np.meshgrid(x, x)
    initial = boundary_field(xx, yy, 6.)
    actual = propagate_field(initial, .25, .25, 1., 60., 'fresnel')['field']
    expected = approximate_field(xx, yy, width=6., distance=60.)
    assert_allclose(actual, expected, atol=2e-14, rtol=0)
    # The explicitly 2D prefactor matters; no 1/sqrt(s) slit substitution.
    assert np.ptp(abs(expected[256])) > .2


def test_spectrum_and_fraunhofer_keep_absolute_complex_prefactors():
    # Independent 2D spatial quadrature of the prescribed boundary.
    x = np.linspace(-24, 24, 961); xx, yy = np.meshgrid(x, x); dx = x[1]-x[0]
    for kx, ky in [(0., 0.), (.6, -.3)]:
        integral = np.sum(boundary_field(xx, yy, 3.)*np.exp(-1j*(kx*xx+ky*yy)))*dx*dx
        assert_allclose(boundary_spectrum(kx, ky, 3.), integral, atol=2e-12, rtol=0)
    p = np.array([0., 2., 5.]); s = 80.; k = 2*np.pi
    expected = np.exp(1j*k*s+1j*k*p*p/(2*s))/(1j*s)*boundary_spectrum(k*p/s, 0., 3.)
    assert_allclose(approximate_field(p, width=3., distance=s, method='fraunhofer'), expected, atol=1e-14)


def test_full_asm_matches_independent_rs_integral():
    # Integrate the full 3D RS I kernel over both source coordinates.
    w, lam, s = 6., 1., 80.; dx = .125
    q = np.arange(-5*w+dx/2, 5*w, dx); xi, eta = np.meshgrid(q, q)
    initial = boundary_field(xi, eta, w); k = 2*np.pi/lam
    points = np.array([0., 8., 19.])
    values = []
    for x in points:
        r = np.sqrt((x-xi)**2+eta**2+s*s)
        kernel = s*np.exp(1j*k*r)*(1-1j*k*r)/(2*np.pi*r**3)
        values.append(np.sum(initial*kernel)*dx*dx)
    assert_allclose(asm_cut(points, width=w, wavelength=lam, distance=s), values, atol=2e-11, rtol=0)


@pytest.mark.parametrize('lam,w,s',[(1.,6.,120.),(.5,3.,800.),(1.5,2.,800.),(.5,12.,20.)])
def test_fourier_quadrature_padding_and_cutoff_converge(lam,w,s):
    x = np.array([0., s*lam/PERIOD, .5*s])
    normal = asm_cut(x, wavelength=lam, width=w, distance=s)
    refined = asm_cut(x, wavelength=lam, width=w, distance=s, window=4096.)
    assert_allclose(normal, refined, atol=2e-7, rtol=2e-4)
    # Raise the spectral cutoff without changing Fourier-node spacing.
    cutoff = asm_cut(x, wavelength=lam, width=w, distance=s, spacing=.25)
    assert_allclose(normal, cutoff, atol=2e-12, rtol=0)


def test_far_field_does_not_remove_paraxial_error():
    paraxial = comparison(.5, 3., 800.)
    wide = comparison(1.5, 2., 800.)
    assert max(paraxial['errors']) < .12
    assert wide['errors'][0] > .1
    # The far-field approximation can closely follow Fresnel while both miss ASM.
    from wave_optics.propagation_comparison import relative_cut_error
    assert relative_cut_error(wide['fields'][2], wide['fields'][1], wide['x']) < .03
    assert not wide['fields'].flags.writeable
    assert np.all(np.isfinite(wide['fields']))


def test_live_controls_change_fields_and_view_reuses_calculation():
    baseline = set(_instances); panel = comparison_explorer()
    try:
        controls = panel.controls; old = panel.data
        before = bytes(panel.image.value); controls['view'].value = 'Phase difference'
        assert panel.data is old and bytes(panel.image.value) != before
        for name, value in [('distance',80.),('width',4.),('wavelength',1.2)]:
            old = panel.data; before = bytes(panel.image.value); controls[name].value = value
            assert panel.data is not old and bytes(panel.image.value) != before
            assert not np.allclose(panel.data['errors'], old['errors'])
        controls['distance'].value = 800.
        assert np.isfinite(panel.data['fields']).all()
        assert '$' not in panel.equation.value and '\\' not in panel.equation.value
        Image.open(BytesIO(panel.image.value)).verify()
    finally: panel.close(); panel.close()
    assert set(_instances) == baseline


def test_phase_difference_omits_dark_points_and_preserves_piston():
    ref = np.array([0j,1.,1.,1.]); u = ref*np.exp(.4j)
    p = phase_difference(u,ref)
    assert np.isnan(p[0]); assert_allclose(p[1:],.4)
    assert np.isnan(phase_difference(np.zeros(4,complex),ref)).all()


def test_notebook_rerun_and_failed_initial_render_release_owned_widgets():
    root = Path(__file__).resolve().parents[2]
    book = nbformat.read(root/'8_3_ASM_Fresnel_and_Fraunhofer_Propagation.ipynb',4)
    source = next(c.source for c in book.cells if c.id == '8-3-comparison-controls')
    baseline = set(_instances); ns = {}
    try:
        with patch('IPython.display.display'):
            exec(source, ns); old = ns['_comparison_panel']; count = len(_instances)
            exec(source, ns)
        assert old._closed and len(_instances) == count
    finally: ns['_comparison_panel'].close()
    assert set(_instances) == baseline
    with patch('wave_optics.propagation_comparison.comparison_figure', side_effect=RuntimeError('render failed')):
        with pytest.raises(RuntimeError): comparison_explorer()
    assert set(_instances) == baseline


def test_invalid_inputs():
    for args in [dict(wavelength=0),dict(width=1),dict(distance=0),dict(width=np.nan),dict(wavelength=True)]:
        with pytest.raises(ValueError): comparison(**args)
    with pytest.raises(ValueError): asm_cut([1j])
    with pytest.raises(ValueError): approximate_field(0.,method='unknown')


def test_output_interpolation_matches_direct_complex_angular_sum():
    from wave_optics.propagation_comparison import _asm_weights
    kx, weights = _asm_weights(1.5, 2., 20.)
    x = np.array([-.713, .021, 2.917, 11.412])
    expected = np.exp(1j*x[:,None]*kx)@weights
    assert_allclose(asm_cut(x,wavelength=1.5,width=2.,distance=20.),expected,atol=8e-11,rtol=0)
