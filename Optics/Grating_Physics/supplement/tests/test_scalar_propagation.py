"""Physics checks for angular-spectrum geometry and propagation."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.scalar_propagation import (
    fresnel_phase_error, propagate_field, spectrum_geometry, transfer_function,
)


class ScalarPropagationTests(unittest.TestCase):
    def test_geometry_obeys_dispersion_with_real_and_complex_directions(self):
        # Frequencies correspond to normal, 3-4-5, grazing, and evanescent cases.
        geometry = spectrum_geometry([0, .3, 1, 1.25], [0, .4, 0, 0], 1)
        k = 2*np.pi
        assert_allclose(geometry['kx']**2+geometry['ky']**2+geometry['kz']**2, k*k)
        assert_allclose(geometry['alpha']**2+geometry['beta']**2+geometry['gamma']**2, 1)
        assert_allclose(geometry['theta_deg'][:3], [0, 30, 90])
        self.assertAlmostEqual(geometry['azimuth_deg'][1], np.rad2deg(np.arctan2(4, 3)))
        self.assertTrue(np.all(geometry['propagating'] == [True, True, True, False]))
        self.assertTrue(np.isnan(geometry['theta_deg'][-1]))
        self.assertTrue(np.isnan(geometry['azimuth_deg'][-1]))
        self.assertEqual(geometry['kz'][-1], .75j*k)
        self.assertTrue(np.isnan(geometry['azimuth_deg'][0]))
        grid = spectrum_geometry(np.array([0, .3])[None, :], np.array([0, .4])[:, None], 1)
        self.assertEqual(grid['kz'].shape, (2, 2))
        self.assertEqual(grid['fx'].shape, (2, 2))

    def test_single_fourier_mode_advances_without_changing_its_shape(self):
        nx, ny, dx, dy = 80, 64, .2, .3
        wavelength, distance = .7, 4.2
        x, y = np.arange(nx)*dx, np.arange(ny)*dy
        fx, fy = 3/(nx*dx), -5/(ny*dy)
        initial = .7*np.exp(1j*(2*np.pi*(fx*x[None, :]+fy*y[:, None])+.23))
        theta = np.arcsin(wavelength*np.hypot(fx, fy))
        expected_phase = 2*np.pi/wavelength*np.cos(theta)*distance
        result = propagate_field(initial, dx, dy, wavelength, distance)
        assert_allclose(result['field'], initial*np.exp(1j*expected_phase), atol=5e-15)
        assert_allclose(abs(result['field']), .7, atol=5e-15)
        self.assertEqual(result['fx'].shape, (nx,))
        self.assertEqual(result['fy'].shape, (ny,))

    def test_evanescent_fourier_mode_decays_with_known_decay_length(self):
        # fx=1.25/lambda implies alpha=0.75*k, so z=1/alpha reduces amplitude to 1/e.
        wavelength, nx, ny, dx, dy = 1., 80, 12, .1, .25
        x = np.arange(nx)*dx
        initial = np.broadcast_to(np.exp(2j*np.pi*1.25*x), (ny, nx))
        decay_length = 1/(.75*2*np.pi)
        result = propagate_field(initial, dx, dy, wavelength, decay_length)
        assert_allclose(result['field'], initial/np.e, atol=4e-15)
        # Grazing is retained, and no spectral component grows in forward ASM.
        h = transfer_function([0, .5, 1, 1.25], 0, wavelength, decay_length)
        assert_allclose(abs(h), [1, 1, 1, 1/np.e], atol=1e-15)

    def test_zero_distance_identity_and_consecutive_propagation(self):
        rng = np.random.default_rng(203)
        initial = rng.normal(size=(15, 18))+1j*rng.normal(size=(15, 18))
        for method in ('asm', 'fresnel'):
            with self.subTest(method=method):
                identity = propagate_field(initial, .2, .3, .8, 0, method)
                assert_allclose(identity['field'], initial, atol=2e-15)
                assert_allclose(identity['transfer'], 1, atol=0, rtol=0)
                first = propagate_field(initial, .2, .3, .8, .4, method)['field']
                second = propagate_field(first, .2, .3, .8, .7, method)['field']
                direct = propagate_field(initial, .2, .3, .8, 1.1, method)['field']
                assert_allclose(second, direct, atol=7e-15)

    def test_paraxial_gaussian_matches_analytic_beam_in_wide_window(self):
        wavelength, waist, size, spacing = .5, 3., 256, .2
        coordinates = (np.arange(size)-size//2)*spacing
        radius_squared = coordinates[None, :]**2+coordinates[:, None]**2
        initial = np.exp(-radius_squared/waist**2)
        rayleigh = np.pi*waist**2/wavelength
        distance = .8*rayleigh
        # Exact solution of the paraxial equation, including curvature and Gouy phase.
        denominator = 1+1j*distance/rayleigh
        expected = np.exp(2j*np.pi*distance/wavelength)/denominator
        expected = expected*np.exp(-radius_squared/(waist**2*denominator))
        result = propagate_field(initial, spacing, spacing, wavelength, distance, 'fresnel')
        assert_allclose(result['field'], expected, atol=4e-14, rtol=1e-12)
        self.assertAlmostEqual(np.sum(abs(result['field'])**2), np.sum(abs(initial)**2), places=10)

    def test_phase_error_is_negative_and_agrees_with_independent_phase_difference(self):
        wavelength, distance = .8, 13.
        q = np.array([0, .03, .2, .7, 1, 1.2])
        fx = np.sqrt(q)/wavelength
        error = fresnel_phase_error(fx, 0, wavelength, distance)
        exact_phase = 2*np.pi*distance/wavelength*np.sqrt(1-q[:5])
        fresnel_phase = 2*np.pi*distance/wavelength*(1-q[:5]/2)
        assert_allclose(error['exact'][:5], exact_phase-fresnel_phase, atol=2e-14)
        self.assertTrue(np.all(error['exact'][1:5] < 0))
        self.assertTrue(np.all(abs(error['exact'][1:5]) >= abs(error['leading'][1:5])))
        self.assertTrue(np.isnan(error['exact'][-1]))
        self.assertTrue(np.isnan(error['leading'][-1]))
        tiny = fresnel_phase_error(np.sqrt(1e-10)/wavelength, 0, wavelength, distance)
        self.assertLess(abs(tiny['exact']/tiny['leading']-1), 6e-11)
        self.assertLess(tiny['exact'], 0)  # Stable even when subtracting phases would lose precision.

    def test_invalid_physical_parameters_are_rejected(self):
        cases = [
            lambda: spectrum_geometry(0, 0, 0),
            lambda: spectrum_geometry(1j, 0, 1),
            lambda: spectrum_geometry([np.nan], 0, 1),
            lambda: spectrum_geometry(np.zeros(2), np.zeros(3), 1),
            lambda: transfer_function(0, 0, 1, -1),
            lambda: transfer_function(0, 0, 1, 1, 'unknown'),
            lambda: fresnel_phase_error(0, 0, 1, -1),
            lambda: propagate_field(np.ones((3, 4)), 0, 1, 1, 1),
            lambda: propagate_field(np.ones((1, 4)), 1, 1, 1, 1),
            lambda: propagate_field(np.ones(4), 1, 1, 1, 1),
            lambda: propagate_field([[1, 2], [3, np.inf]], 1, 1, 1, 1),
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    case()


if __name__ == '__main__':
    unittest.main()
