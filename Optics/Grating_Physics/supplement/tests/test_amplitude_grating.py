"""Analytical aperture coefficients, normalization, and common-plane phase."""

from dataclasses import replace
import unittest
import numpy as np
from numpy.testing import assert_allclose

from wave_optics.absorbing_grating import AbsorbingGratingConfiguration, solve_absorbing_grating
from wave_optics.amplitude_grating import (
    IdealAmplitudeGrating, ideal_amplitude_coefficients, ideal_amplitude_exit_field, AbsorbingComparisonData,
    field_fourier_coefficients, FIELD_FFT_ATOL,
)
from wave_optics.thin_grating import compare_grating_models


class AmplitudeGratingTests(unittest.TestCase):
    def setUp(self):
        self.c = AbsorbingGratingConfiguration()

    def test_analytic_aperture_formula_and_sampled_projection(self):
        # Independently integrate the centred or shifted rectangular aperture.
        for fraction, shift in ((.5, 0.), (.37, .13), (0., 0.), (1., 0.)):
            c = replace(self.c, clear_width=fraction*self.c.period, opening_shift=shift)
            m = np.arange(-12, 13)
            benchmark = np.zeros(m.size, dtype=complex)
            benchmark[m == 0] = fraction
            nonzero = m != 0
            benchmark[nonzero] = np.sin(np.pi*m[nonzero]*fraction)/(np.pi*m[nonzero])
            benchmark *= np.exp(-2j*np.pi*m*shift/c.period)
            phase = np.exp(2j*np.pi*c.thickness/c.wavelength)
            assert_allclose(ideal_amplitude_coefficients(c, m), phase*benchmark, atol=2e-15)
            # Project over an exact period; the tolerance allows edge sampling.
            count = 65536
            x = (np.arange(count)-count/2)*c.period/count
            field = ideal_amplitude_exit_field(c, x)
            projected = np.array([np.mean(field*np.exp(-2j*np.pi*order*x/c.period)) for order in m])
            assert_allclose(projected, phase*benchmark, atol=3/count)

    def test_parseval_normalization_and_zero_phase(self):
        m = np.arange(-4000, 4001)
        coefficients = ideal_amplitude_coefficients(self.c, m)
        self.assertLess(abs(np.sum(abs(coefficients)**2)-self.c.open_fraction), 6e-5)
        ideal = IdealAmplitudeGrating(self.c)
        self.assertAlmostEqual(abs(ideal.transmission[self.c.M]), .5)
        assert_allclose(ideal.transmission[abs(ideal.orders) == 1],
                        np.exp(2j*np.pi*self.c.thickness/self.c.wavelength)/np.pi, atol=2e-15)
        physical = solve_absorbing_grating(self.c)
        plane = compare_grating_models(physical, -.3, thin_model=ideal)
        self.assertTrue(np.isnan(plane.thin_phase[self.c.M+2]))
        self.assertTrue(np.isnan(plane.phase_difference[self.c.M+2]))

    def test_complex_field_fft_normalization_origin_and_signed_orders(self):
        m = np.array([0, -4, 3, -1, 5])  # Deliberately not FFT bin order.
        expected = np.array([.7+.2j, -.13+.8j, .25-.1j, -.2-.3j, .1j])
        for count in (127, 128):
            for origin in (-self.c.period/2, 0., .173*self.c.period):
                x = origin+np.arange(count)*self.c.period/count
                u = expected @ np.exp(2j*np.pi*m[:, None]*x/self.c.period)
                actual = field_fourier_coefficients(u, x, self.c.period, m)
                assert_allclose(actual, expected, atol=3e-15)
                self.assertAlmostEqual(np.sum(abs(actual)**2), np.mean(abs(u)**2), places=13)
                # Discarding field phase cannot recover these complex coefficients.
                wrong = field_fourier_coefficients(abs(u), x, self.c.period, m)
                self.assertGreater(np.max(abs(wrong-expected)), .1)

    def test_sampled_hbvp_and_ideal_fields_preserve_common_plane_phase(self):
        c = replace(self.c, M=12, opening_shift=.13)
        physical, ideal = solve_absorbing_grating(c), IdealAmplitudeGrating(c)
        x = np.linspace(-c.period/2, c.period/2, 512, endpoint=False)
        for model in (physical, ideal):
            for z in (-c.thickness, -.57, -1.3):
                u = model.reconstruct(x, z)[0]
                actual = field_fourier_coefficients(u, x, c.period, model.orders)
                self.assertLess(np.max(abs(actual-model.at_reference_plane(z))), FIELD_FFT_ATOL)

    def test_field_fft_rejects_bad_sampling_and_unresolved_orders(self):
        x = np.linspace(-.5, .5, 16, endpoint=False)
        u = np.ones(16, dtype=complex)
        for samples in (x[::-1], np.roll(x, 1), np.linspace(-.5, .5, 16), x*1.1):
            with self.assertRaises(ValueError):
                field_fourier_coefficients(u, samples, 1., np.array([0]))
        for values, samples, orders in ((u[:-1], x, [0]), (u[:, None], x, [0]),
                (u*np.nan, x, [0]), (u, x, [8]), (u, x, [-8]), (u, x, [.5])):
            with self.assertRaises(ValueError):
                field_fourier_coefficients(values, samples, 1., np.array(orders))

    def test_identical_phase_origin_and_outgoing_propagation(self):
        c = replace(self.c, clear_width=self.c.period)
        ideal, physical = IdealAmplitudeGrating(c), solve_absorbing_grating(c)
        for z in (-c.thickness, -.6, -1.1):
            assert_allclose(ideal.at_reference_plane(z), physical.at_reference_plane(z), atol=3e-14)
            self.assertAlmostEqual(ideal.at_reference_plane(z)[c.M], np.exp(-2j*np.pi*z/c.wavelength))
        ideal = IdealAmplitudeGrating(self.c)
        expected = ideal.transmission*np.exp(1j*ideal.beta_output*.17)
        assert_allclose(ideal.at_reference_plane(-.47), expected, atol=2e-15)
        mask = ideal.output_kinds == 'evanescent'
        self.assertTrue(np.all(abs(expected[mask]) <= abs(ideal.transmission[mask])))
        assert_allclose(ideal.transmission_efficiency,
                        ideal.beta_output.real/(2*np.pi/self.c.wavelength)*abs(ideal.transmission)**2)
        with self.assertRaises(ValueError):
            ideal.at_reference_plane(0.)

    def test_thickness_sweep_solves_distinct_layers_at_same_output_distance(self):
        data = AbsorbingComparisonData(replace(self.c, M=8), [.08, .3, .7])
        planes = data.planes(.2)
        for s, ideal, p in zip(data.solutions, data.ideal_models, planes):
            self.assertAlmostEqual(p.z_ref, -s.configuration.thickness-.2)
            assert_allclose(p.hbvp, s.at_reference_plane(p.z_ref))
            assert_allclose(p.thin, ideal.at_reference_plane(p.z_ref))
            self.assertAlmostEqual(s.energy_balance, 1, places=10)
        self.assertGreater(np.ptp([abs(p.hbvp[8]) for p in planes]), .01)
        with self.assertRaises(ValueError):
            data.planes(-1)
        with self.assertRaises(ValueError):
            AbsorbingComparisonData(self.c, [-.1])


if __name__ == '__main__':
    unittest.main()
