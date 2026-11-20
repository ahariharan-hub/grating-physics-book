"""Independent numerical acceptance checks for milestone 0_5C.1."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from wave_optics.scalar_grating import ScalarGratingConfiguration, solve_scalar_grating
from wave_optics.scalar_propagation import propagate_field, transfer_function
from wave_optics.thin_grating import (
    ThinPhaseGrating, compare_grating_models, thin_mask_coefficients, thin_mask_exit_field,
)


def integrated_coefficient(c, m, shift=0.):
    """Quadrature of three constant-index intervals, independent of sinc formula."""
    edge1, edge2 = shift-c.duty_cycle*c.period/2, shift+c.duty_cycle*c.period/2
    intervals = ((-c.period/2, edge1, c.n_groove),
                 (edge1, edge2, c.n_ridge), (edge2, c.period/2, c.n_groove))
    value = 0j
    for left, right, index in intervals:
        def integrand(x):
            return np.exp(2j*np.pi*(index*c.thickness/c.wavelength-m*x/c.period))
        value += quad(lambda x: integrand(x).real, left, right, epsabs=1e-12)[0]
        value += 1j*quad(lambda x: integrand(x).imag, left, right, epsabs=1e-12)[0]
    return value/c.period


class ThinGratingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ScalarGratingConfiguration(M=20)
        cls.solution = solve_scalar_grating(cls.config)
        cls.thin = ThinPhaseGrating(cls.config)

    def test_analytical_coefficients_match_independent_piecewise_quadrature(self):
        orders = np.array([-20, -7, -2, -1, 0, 1, 2, 7, 20])
        for f in (.17, .5, .83):
            c = replace(self.config, duty_cycle=f)
            expected = np.array([integrated_coefficient(c, m) for m in orders])
            assert_allclose(thin_mask_coefficients(c, orders), expected, atol=2e-14, rtol=0)

    def test_centred_origin_has_even_complex_coefficients_and_correct_translation_phase(self):
        c = replace(self.config, duty_cycle=.37)
        m = np.arange(-5, 6)
        coefficients = thin_mask_coefficients(c, m)
        assert_allclose(coefficients, coefficients[::-1], atol=1e-15)
        # Even complex masks give C[-m]=C[m], not generally conjugate symmetry.
        self.assertGreater(np.max(abs(coefficients-coefficients[::-1].conj())), .1)
        shift = .12*c.period
        translated = [integrated_coefficient(c, order, shift) for order in m]
        assert_allclose(translated, coefficients*np.exp(-2j*np.pi*m*shift/c.period), atol=2e-14)

    def test_exit_field_includes_absolute_uniform_phase_and_geometry_edges(self):
        c = self.config
        x = c.period*np.array([-.5, -.25, 0., .25, .5, 1.])
        expected_indices = np.array([c.n_groove, c.n_ridge, c.n_ridge,
                                     c.n_groove, c.n_groove, c.n_ridge])
        expected = np.exp(2j*np.pi*expected_indices*c.thickness/c.wavelength)
        assert_allclose(thin_mask_exit_field(c, x), expected, atol=1e-15)
        uniform_phase = np.exp(2j*np.pi*c.n_output*c.thickness/c.wavelength)
        relative_mask = np.exp(2j*np.pi*(expected_indices-c.n_output)*c.thickness/c.wavelength)
        assert_allclose(expected, uniform_phase*relative_mask, atol=1e-15)
        self.assertGreater(abs(uniform_phase-1), .1)  # A missing common phase is detectable.

    def test_uniform_medium_agrees_with_exact_plane_wave_and_hbvp(self):
        c = replace(self.config, n_ridge=1.23, n_groove=1.23, n_incident=1.23, n_output=1.23)
        thin, hbvp = ThinPhaseGrating(c), solve_scalar_grating(c)
        x, z = np.array([-.37, 0., .29]), np.array([-c.thickness, -.57, -1.4])
        expected = np.broadcast_to(np.exp(-2j*np.pi*c.n_output*z[:, None]/c.wavelength), (3, 3))
        assert_allclose(thin.reconstruct(x, z), expected, atol=3e-14)
        assert_allclose(hbvp.reconstruct(x, z), expected, atol=3e-14)
        assert_allclose(thin.transmission[thin.orders != 0], 0, atol=0)
        self.assertAlmostEqual(np.sum(thin.transmission_efficiency), 1., places=14)

    def test_uniform_slab_is_not_mistaken_for_uniform_medium(self):
        c = replace(self.config, n_groove=1.5, n_ridge=1.5)
        thin, hbvp = ThinPhaseGrating(c), solve_scalar_grating(c)
        # A mismatched uniform layer still reflects in the BVP; the mask omits this.
        self.assertGreater(np.sum(hbvp.reflection_efficiency), .01)
        self.assertGreater(abs(hbvp.transmission[c.M]-thin.transmission[c.M]), .01)
        assert_allclose(thin.transmission[thin.orders != 0], 0, atol=0)

    def test_zero_and_full_duty_are_single_phase_factors(self):
        for duty, index in ((0., self.config.n_groove), (1., self.config.n_ridge)):
            c = replace(self.config, duty_cycle=duty)
            coefficients = thin_mask_coefficients(c)
            assert_allclose(coefficients[c.M], np.exp(2j*np.pi*index*c.thickness/c.wavelength))
            assert_allclose(coefficients[np.arange(2*c.M+1) != c.M], 0, atol=0)

    def test_negative_z_propagation_matches_hbvp_branch_and_asm(self):
        thin, c = self.thin, self.config
        distance = .137
        assert_allclose(thin.beta_output, self.solution.beta_output, atol=0)
        self.assertTrue(np.array_equal(thin.output_kinds, self.solution.output_kinds))
        expected = thin.transmission*transfer_function(thin.orders/c.period, 0.,
                                                       c.wavelength/c.n_output, distance)
        assert_allclose(thin.at_reference_plane(-c.thickness-distance), expected, atol=5e-15)
        propagating = thin.output_kinds == 'propagating'
        near, far = thin.transmission[propagating], expected[propagating]
        assert_allclose(abs(far), abs(near), atol=1e-15)
        meaningful = abs(near) > 1e-6
        assert_allclose(np.angle(far[meaningful]/near[meaningful]),
                        np.angle(np.exp(1j*thin.beta_output[propagating][meaningful]*distance)), atol=1e-14)

    def test_reconstructed_propagation_matches_existing_fft_asm(self):
        thin, c = self.thin, self.config
        x = np.arange(128)*c.period/128
        initial = np.tile(thin.reconstruct(x, -c.thickness), (4, 1))
        propagated = propagate_field(initial, c.period/128, 1., c.wavelength/c.n_output, .17)['field']
        expected = np.tile(thin.reconstruct(x, -c.thickness-.17), (4, 1))
        assert_allclose(propagated, expected, atol=8e-15)

    def test_evanescent_decay_length_and_grazing_zero_flux(self):
        thin = self.thin
        i = self.config.M+3
        distance = 1/thin.beta_output[i].imag
        self.assertGreater(abs(thin.transmission[i]), .01)
        actual = thin.at_reference_plane(-self.config.thickness-distance)[i]
        assert_allclose(actual, thin.transmission[i]/np.e, atol=3e-16)
        grazing = ThinPhaseGrating(replace(self.config, wavelength=1., period=1., M=4))
        select = abs(grazing.orders) == 1
        self.assertTrue(np.all(grazing.output_kinds[select] == 'grazing'))
        assert_allclose(grazing.at_reference_plane(-20.)[select], grazing.transmission[select], atol=0)
        assert_allclose(grazing.transmission_efficiency[select], 0, atol=0)
        assert_allclose(grazing.transmission_efficiency[grazing.output_kinds == 'evanescent'], 0, atol=0)

    def test_retained_reconstruction_converges_away_from_discontinuities(self):
        c = replace(self.config, duty_cycle=.37)
        x = np.linspace(-.5*c.period, .5*c.period, 1001, endpoint=False)
        away = np.minimum(abs(x-c.duty_cycle*c.period/2), abs(x+c.duty_cycle*c.period/2)) > .08*c.period
        exact = thin_mask_exit_field(c, x[away])
        errors = []
        for M in (32, 128, 512):
            thin = ThinPhaseGrating(replace(c, M=M))
            errors.append(np.max(abs(thin.reconstruct(x[away], -c.thickness)[0]-exact)))
        self.assertLess(errors[-1], .005)
        self.assertLess(errors[-1], errors[0]/8)
        # At an ideal step the Fourier series approaches the average, not either side.
        edge = c.duty_cycle*c.period/2
        average = .5*(np.exp(2j*np.pi*c.n_ridge*c.thickness/c.wavelength)
                      + np.exp(2j*np.pi*c.n_groove*c.thickness/c.wavelength))
        assert_allclose(thin.reconstruct(edge, -c.thickness)[0, 0], average, atol=.002)

    def test_common_plane_differences_preserve_phase_reference_without_resolve(self):
        c, s, thin = self.config, self.solution, self.thin
        with patch('wave_optics.scalar_grating.solve_scalar_grating', side_effect=AssertionError('BVP rerun')):
            near = compare_grating_models(s, -c.thickness, thin_model=thin)
            for distance in (.0, .15, .7, 1.5):
                plane = compare_grating_models(s, -c.thickness-distance, thin_model=thin)
                factor = np.exp(1j*s.beta_output*distance)
                assert_allclose(plane.hbvp, s.at_reference_plane(plane.z_ref), atol=0)
                assert_allclose(plane.thin, thin.transmission*factor, atol=1e-15)
                assert_allclose(plane.difference, plane.hbvp-plane.thin, atol=0)
                assert_allclose(plane.difference, near.difference*factor, atol=2e-15)
                meaningful = np.isfinite(plane.phase_difference)
                assert_allclose(plane.phase_difference[meaningful], near.phase_difference[meaningful], atol=2e-14)

    def test_phases_are_missing_when_either_coefficient_is_negligible(self):
        near = compare_grating_models(self.solution, -.3)
        zero_thin = (near.orders != 0) & (near.orders % 2 == 0)
        self.assertTrue(np.all(np.isnan(near.thin_phase[zero_thin])))
        self.assertTrue(np.all(np.isnan(near.phase_difference[zero_thin])))
        i = self.config.M+1
        self.assertAlmostEqual(near.phase_difference[i], np.angle(near.hbvp[i]*near.thin[i].conjugate()))
        far = compare_grating_models(self.solution, -20.)
        self.assertTrue(np.all(np.isnan(far.thin_phase[far.kinds == 'evanescent'])))
        self.assertTrue(np.all(np.isnan(far.hbvp_phase[far.kinds == 'evanescent'])))

    def test_longitudinal_flux_weighting_and_model_power_balance(self):
        c, thin, s = self.config, self.thin, self.solution
        # Verify the modal power independently by integrating -Im(U* dU/dz).
        x = np.arange(256)*c.period/256
        basis = np.exp(1j*thin.kx[:, None]*x)
        coefficients = thin.at_reference_plane(-.6)
        u = coefficients @ basis
        derivative = (-1j*thin.beta_output*coefficients) @ basis
        flux = -np.imag(np.mean(u.conj()*derivative))/(2*np.pi*c.n_incident/c.wavelength)
        self.assertAlmostEqual(flux, np.sum(thin.transmission_efficiency), places=13)
        self.assertLess(np.sum(thin.transmission_efficiency), .95)
        self.assertNotAlmostEqual(np.sum(abs(thin.transmission)**2), flux, places=3)
        comparison = compare_grating_models(s, -.8)
        self.assertAlmostEqual(comparison.hbvp_power_balance, 1., places=12)
        assert_allclose(comparison.hbvp_efficiency, s.transmission_efficiency, atol=0)
        self.assertAlmostEqual(comparison.thin_transmittance, flux, places=13)

    def test_length_unit_rescaling_preserves_complex_fields(self):
        c = self.config
        scaled = ThinPhaseGrating(replace(c, wavelength=c.wavelength*1e-6,
                                         period=c.period*1e-6, thickness=c.thickness*1e-6))
        assert_allclose(scaled.reconstruct([0., .2e-6], [-.3e-6, -.7e-6]),
                        self.thin.reconstruct([0., .2], [-.3, -.7]), atol=1e-14)

    def test_validation_shapes_and_read_only_data(self):
        with self.assertRaisesRegex(ValueError, 'equal exterior'):
            ThinPhaseGrating(replace(self.config, n_output=1.4))
        with self.assertRaises(TypeError):
            thin_mask_coefficients({})
        for bad in ([], [1.5], [True], [[1, 2]]):
            with self.assertRaises(ValueError):
                thin_mask_coefficients(self.config, bad)
        for bad in (0., -.29, np.nan, [np.nan], [[-.4]], [1j]):
            with self.assertRaises(ValueError):
                self.thin.harmonics(bad)
        with self.assertRaises(ValueError):
            self.thin.at_reference_plane([-.4, -.5])
        for bad in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                compare_grating_models(self.solution, -.4, phase_floor=bad)
        with self.assertRaises(ValueError):
            compare_grating_models(self.solution, -.4, thin_model=ThinPhaseGrating(replace(self.config, M=2)))
        with self.assertRaises(ValueError):
            self.thin.transmission[0] = 1
        single = ThinPhaseGrating(replace(self.config, M=0))
        self.assertEqual(single.reconstruct(0., -.4).shape, (1, 1))
        self.assertEqual(single.at_reference_plane(-.4).shape, (1,))
        with self.assertRaises(ValueError):
            compare_grating_models(self.solution, -.4).difference[0] = 1


if __name__ == '__main__':
    unittest.main()
