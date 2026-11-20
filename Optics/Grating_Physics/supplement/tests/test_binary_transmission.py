"""Analytical Fourier coefficients and normalization of both ideal binary masks."""

from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from wave_optics.binary_transmission import (
    binary_transmission_coefficients, binary_transmission_field, dielectric_phase_levels,
)
from wave_optics.scalar_grating import ScalarGratingConfiguration
from wave_optics.thin_grating import thin_mask_coefficients, thin_mask_exit_field


class BinaryTransmissionTests(unittest.TestCase):
    def test_both_models_match_independent_period_average_integrals(self):
        orders = np.array([-12, -3, -1, 0, 1, 3, 12])
        for period in (.8, 3.1):
            for f in (.21, .5, .77):
                for tr, tg in ((1., 0.), (np.exp(1.3j), np.exp(-.4j))):
                    expected = []
                    for m in orders:
                        def integral(left, right, level):
                            fun = lambda x: level*np.exp(-2j*np.pi*m*x/period)
                            return (quad(lambda x: fun(x).real, left, right, epsabs=1e-12)[0]
                                    + 1j*quad(lambda x: fun(x).imag, left, right, epsabs=1e-12)[0])
                        edge = f*period/2
                        expected.append((integral(-period/2, -edge, tg)+integral(-edge, edge, tr)
                                         + integral(edge, period/2, tg))/period)
                    assert_allclose(binary_transmission_coefficients(orders, f, tr, tg), expected, atol=3e-14)

    def test_half_duty_amplitude_mask_has_expected_normalization_and_signed_orders(self):
        m = np.arange(-6, 7)
        values = binary_transmission_coefficients(m, .5, 1, 0)
        self.assertEqual(values[6], .5)
        assert_allclose(values[6+np.array([1, 3, 5])], [1/np.pi, -1/(3*np.pi), 1/(5*np.pi)], atol=1e-16)
        assert_allclose(values[(m != 0) & (m % 2 == 0)], 0, atol=1e-16)
        assert_allclose(values, values[::-1], atol=0)
        self.assertEqual(abs(values[6])**2, .25)  # This is not the total transmitted mean square f=0.5.

    def test_unit_amplitude_phase_mask_diffracts_and_can_suppress_zero_order(self):
        m = np.arange(-5, 6)
        values = binary_transmission_coefficients(m, .5, -1, 1)
        self.assertEqual(values[5], 0)
        assert_allclose(values[6], -2/np.pi, atol=1e-16)
        field = binary_transmission_field(np.linspace(-1., 1., 201), 1., .5, -1, 1)
        assert_allclose(abs(field), 1, atol=0)
        self.assertGreater(abs(values[6]), .6)
        uniform = binary_transmission_coefficients(m, .37, np.exp(.3j), np.exp(.3j))
        assert_allclose(uniform[m != 0], 0, atol=0)
        assert_allclose(uniform[m == 0], np.exp(.3j), atol=0)

    def test_parseval_infinite_sum_limit_for_both_masks(self):
        M = 2000
        m = np.arange(-M, M+1)
        for f in (.19, .5, .83):
            for tr, tg in ((1, 0), (np.exp(1.9j), np.exp(-.7j))):
                coefficients = binary_transmission_coefficients(m, f, tr, tg)
                norm = np.sum(abs(coefficients)**2)
                exact = f*abs(tr)**2+(1-f)*abs(tg)**2
                # Analytic bound on omitted +/-m tails, using |sin(pi*m*f)| <= 1.
                bound = 2*abs(tr-tg)**2/(np.pi**2*M)
                self.assertGreaterEqual(exact-norm, -1e-14)
                self.assertLess(exact-norm, bound)

    def test_finite_field_parseval_uses_same_fourier_normalization(self):
        m = np.arange(-20, 21)
        x = np.arange(256)/256
        for tr, tg in ((1, 0), (np.exp(1.9j), np.exp(-.7j))):
            coefficients = binary_transmission_coefficients(m, .37, tr, tg)
            field = coefficients @ np.exp(2j*np.pi*m[:, None]*x)
            self.assertAlmostEqual(np.mean(abs(field)**2), np.sum(abs(coefficients)**2), places=14)
            self.assertLess(np.mean(abs(field)**2), .37*abs(tr)**2+.63*abs(tg)**2)

    def test_exact_local_field_has_centred_edges_periodicity_and_two_levels(self):
        x = np.array([-.5, -.25, 0., .25, .5])
        for tr, tg in ((1, 0), (np.exp(.7j), np.exp(-1.3j))):
            expected = np.array([tg, tr, tr, tg, tg])
            assert_allclose(binary_transmission_field(x, 1., .5, tr, tg), expected, atol=0)
            assert_allclose(binary_transmission_field(x+3, 1., .5, tr, tg), expected, atol=0)
        assert_allclose(binary_transmission_field(x, 1., 0., 1., 0.), 0, atol=0)
        assert_allclose(binary_transmission_field(x, 1., 1., 1., 0.), 1, atol=0)

    def test_uniform_endpoint_duties_and_common_phase_rotation(self):
        m = np.arange(-7, 8)
        tr, tg = np.exp(.4j), np.exp(1.7j)
        for f, level in ((0., tg), (1., tr)):
            values = binary_transmission_coefficients(m, f, tr, tg)
            assert_allclose(values, (m == 0)*level, atol=0)
        base = binary_transmission_coefficients(m, .37, tr, tg)
        rotation = np.exp(.61j)
        rotated = binary_transmission_coefficients(m, .37, tr*rotation, tg*rotation)
        assert_allclose(rotated, base*rotation, atol=2e-16)
        assert_allclose(abs(rotated), abs(base), atol=2e-16)

    def test_dielectric_levels_retain_negative_z_absolute_phase_and_legacy_results(self):
        c = ScalarGratingConfiguration(M=20)
        tr, tg = dielectric_phase_levels(c)
        assert_allclose([tr, tg], np.exp(2j*np.pi*c.thickness/c.wavelength*np.array([c.n_ridge, c.n_groove])), atol=1e-15)
        m = np.arange(-c.M, c.M+1)
        assert_allclose(binary_transmission_coefficients(m, c.duty_cycle, tr, tg), thin_mask_coefficients(c), atol=0)
        x = np.linspace(-c.period, c.period, 101)
        assert_allclose(binary_transmission_field(x, c.period, c.duty_cycle, tr, tg), thin_mask_exit_field(c, x), atol=0)
        uniform = replace(c, n_ridge=1., n_groove=1.)
        values = binary_transmission_coefficients(m, uniform.duty_cycle, *dielectric_phase_levels(uniform))
        assert_allclose(values[m == 0], np.exp(2j*np.pi*c.thickness/c.wavelength), atol=1e-15)

    def test_parameter_validation_and_scalar_order(self):
        self.assertEqual(binary_transmission_coefficients(0, .5, 1, 0).shape, ())
        for orders in ([], [1.2], [[1]], [True]):
            with self.assertRaises(ValueError):
                binary_transmission_coefficients(orders, .5, 1, 0)
        for duty in (-.1, 1.1, np.nan, [0.5]):
            with self.assertRaises(ValueError):
                binary_transmission_coefficients([0], duty, 1, 0)
        for level in (np.nan, np.inf, [1, 2], True):
            with self.assertRaises(ValueError):
                binary_transmission_coefficients([0], .5, level, 0)
        with self.assertRaises(ValueError):
            binary_transmission_field([0], 0., .5, 1, 0)
        with self.assertRaises(ValueError):
            binary_transmission_field([np.nan], 1., .5, 1, 0)
        with self.assertRaises(TypeError):
            dielectric_phase_levels({})


if __name__ == '__main__':
    unittest.main()
