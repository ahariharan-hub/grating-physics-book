"""Notebook 7 V2 scalar correlation and resolved-intensity identities."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics import mutual_coherence as mc
from wave_optics.phase_exposure import phase_step_data
from wave_optics.superposition import two_wave_cycle


class MutualCoherenceTests(unittest.TestCase):
    def test_complex_superposition_and_cross_term(self):
        u1 = np.array([1+2j, -1j, 3.])
        u2 = np.array([2-1j, 2j, -2.])
        w = np.array([1., 2., 3.]) / 6
        result = mc.mutual_statistics(u1, u2, weights=w)
        assert_allclose(result['correlation'], np.sum(w*u1.conjugate()*u2))
        assert_allclose(result['intensity'], result['intensity1']+result['intensity2']+result['cross_term'])
        for beta in (0, 90, 180, 277):
            cycle = two_wave_cycle(1., 2., beta)
            assert_allclose(cycle['relative_intensity'], mc.averaged_intensity(1, 4, 1, np.deg2rad(beta)))

    def test_normalization_is_complex_and_independent_of_positive_attenuation(self):
        first = np.array([1+1j, -1., 2j])
        factor = 2*np.exp(.7j)
        for attenuation in (1., .2):
            result = mc.mutual_statistics(first, attenuation*factor*first)
            assert_allclose(result['gamma'], np.exp(.7j), atol=2e-15)
        result = mc.mutual_statistics(np.ones(4), np.array([1, 1j, -1, -1j]))
        assert_allclose(result['gamma'], 0, atol=1e-15)

    def test_prescribed_magnitudes_and_visibility_are_independent_controls(self):
        phase = np.linspace(0, 2*np.pi, 1001)
        for i1, i2, g in ((1, 1, 1), (1, 1, .5), (1, 1, 0), (1, 4, 1), (.1, 3, .3)):
            intensity = mc.averaged_intensity(i1, i2, g, phase)
            measured = (intensity.max()-intensity.min())/(intensity.max()+intensity.min())
            assert_allclose(measured, mc.fringe_visibility(i1, i2, g), atol=1e-14)
        self.assertEqual(mc.fringe_visibility(1, 4, 1), .8)
        self.assertEqual(mc.fringe_visibility(1, 1, .8), .8)

    def test_fixed_eo_phase_rotates_correlation_without_reducing_magnitude(self):
        first = np.array([1, 2j, -3, 1-1j])
        second = np.array([1j, 1, -1j, 2])
        initial = mc.mutual_statistics(first, second)
        for beta in (0, 73, 180, 360):
            shifted = mc.mutual_statistics(first, second*np.exp(1j*np.deg2rad(beta)))
            assert_allclose(shifted['gamma'], initial['gamma']*np.exp(1j*np.deg2rad(beta)), atol=1e-15)
            assert_allclose(abs(shifted['gamma']), abs(initial['gamma']))

    def test_specified_exposure_weights_average_intensity_not_field(self):
        phase = np.linspace(-4*np.pi, 4*np.pi, 201)
        for beta, p in ((180, .5), (90, .5), (180, .2), (180, 0), (180, 1)):
            gamma = mc.phase_average([0, beta], weights=[p, 1-p])
            result = mc.averaged_intensity(1, 1, gamma, phase)
            expected = p*abs(1+np.exp(1j*phase))**2 + (1-p)*abs(1+np.exp(1j*(phase+np.deg2rad(beta))))**2
            assert_allclose(result, expected, atol=3e-15)
            assert_allclose(abs(gamma), abs(p+(1-p)*np.exp(1j*np.deg2rad(beta))))
        assert_allclose(mc.phase_average([0, 180]), 0, atol=1e-15)
        # Reuse the original Notebook 7 EO computation as an integration oracle.
        u1, u2 = np.ones_like(phase, complex), np.exp(1j*phase)
        source = dict(fields=dict(field_1=u1, field_2=u2, intensity=abs(u1+u2)**2),
                      point=dict(intensity_1=1., intensity_2=1.), coordinates=phase, intensity_limit=4.)
        old = phase_step_data(source, 90.)
        assert_allclose(old['average'], mc.averaged_intensity(1, 1, mc.phase_average([0, 90]), phase))

    def test_phase_moves_maxima_by_the_predicted_signed_distance(self):
        beta = np.deg2rad(90)
        q = 2*np.pi
        x = np.linspace(-2, 2, 401)
        assert_allclose(mc.averaged_intensity(1, 1, 1, q*x+beta),
                        mc.averaged_intensity(1, 1, 1, q*(x+beta/q)), atol=1e-14)
        assert_allclose(mc.averaged_intensity(1, 1, 1, q*(-beta/q)+beta), 4)

    def test_two_orders_have_equal_wavenumber_and_preserve_coefficient_phase(self):
        basis = mc.two_order_basis(np.linspace(-8, 8, 401))
        assert_allclose(np.linalg.norm(basis['vectors'], axis=1), np.pi)
        assert_allclose(basis['vectors'][:, 0], [0, np.pi/2])
        self.assertTrue(np.all(basis['vectors'][:, 2] < 0))
        for ratio, beta in ((1, 0), (.3, 63), (2, 180), (0, 90)):
            expected = abs(basis['field1']+ratio*np.exp(1j*np.deg2rad(beta))*basis['field2'])**2
            assert_allclose(mc.two_order_intensity(basis, ratio, beta), expected, atol=4e-16)
        assert_allclose(mc.two_order_intensity(basis, 1, 0, 0), .32)
        assert_allclose(mc.two_order_basis(basis['x']+4)['field2']/basis['field2'], 1, atol=3e-15)

    def test_dark_arms_do_not_fabricate_normalized_coherence(self):
        for a1, a2 in ((0, 1), (1, 0), (0, 0)):
            result = mc.mutual_statistics([a1]*2, [a2]*2)
            self.assertIsNone(result['gamma'])
            self.assertEqual(result['cross_term'], 0)
            assert_allclose(mc.averaged_intensity(a1, a2, None, [0, 1]), a1+a2)
        self.assertEqual(mc.fringe_visibility(0, 1), 0)
        self.assertIsNone(mc.fringe_visibility(0, 0))

    def test_invalid_sampling_and_correlations_are_rejected(self):
        for weights in ([0, 0], [1, -1], [1], [np.nan, 1]):
            with self.assertRaises(ValueError):
                mc.phase_average([0, 180], weights=weights)
        basis = mc.two_order_basis([0., 1.])
        for magnitude in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                mc.two_order_intensity(basis, magnitude=magnitude)
        for gamma in (1.1, np.nan, None):
            with self.assertRaises(ValueError):
                mc.averaged_intensity(1, 1, gamma)


if __name__ == '__main__':
    unittest.main()
