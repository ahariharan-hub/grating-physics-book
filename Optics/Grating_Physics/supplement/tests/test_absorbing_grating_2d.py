"""Scalar absorption, aperture-reference and FFT2 regression checks."""
import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.absorbing_grating import AbsorbingGratingConfiguration, solve_absorbing_grating
from wave_optics.scalar_grating_2d import ScalarGrating2DConfiguration as Config, solve_scalar_grating_2d as solve
from wave_optics.amplitude_grating_2d import (
    absorbing_screen_configuration, AbsorbingComparison2DData, IdealAmplitudeGrating2D,
    field_fourier_coefficients_2d,
)


class Absorbing2DTests(unittest.TestCase):
    def test_analytic_absorbing_oblique_slab(self):
        index = 1.4+.3j
        c = Config(n_inclusion=index, n_background=index, n_output=1.2,
                   kx_incident=.9, ky_incident=-.6)
        s = solve(c)
        k0 = 2*np.pi/c.wavelength
        kt2 = c.kx_incident**2+c.ky_incident**2
        b0, b1, b2 = [np.sqrt((k0*n)**2-kt2) for n in (c.n_incident, index, c.n_output)]
        r01, r12 = (b0-b1)/(b0+b1), (b1-b2)/(b1+b2)
        phase = np.exp(1j*b1*c.thickness)
        r = (r01+r12*phase**2)/(1+r01*r12*phase**2)
        t = (2*b0/(b0+b1))*(2*b1/(b1+b2))*phase/(1+r01*r12*phase**2)
        assert_allclose(s.reflection[s.incident_index], r, atol=2e-13)
        assert_allclose(s.transmission[s.incident_index], t, atol=2e-13)
        self.assertAlmostEqual(s.absorption, 1-abs(r)**2-b2/b0*abs(t)**2, places=12)
        self.assertLess(s.absorption_quadrature_error, 1e-9)

    def test_absorbing_1d_limit(self):
        c = absorbing_screen_configuration(Mx=4, My=1, fill_y=1.)
        s = solve(c)
        old = solve_absorbing_grating(AbsorbingGratingConfiguration(M=4))
        keep = s.orders[:, 1] == 0
        assert_allclose(s.transmission[keep], old.transmission, atol=3e-12)
        assert_allclose(s.reflection[keep], old.reflection, atol=3e-12)
        self.assertAlmostEqual(s.absorption, old.absorption, places=11)
        assert_allclose(s.transmission[~keep], 0., atol=3e-12)

    def test_patterned_absorption_flux_and_interfaces(self):
        c = absorbing_screen_configuration(fill_y=.35, kx_incident=.7, ky_incident=-.4, Mx=2, My=2)
        s = solve(c)
        self.assertGreater(s.absorption, 0)
        self.assertAlmostEqual(s.energy_balance, 1, places=11)
        flux = s.downward_flux([0, -c.thickness])
        assert_allclose(flux, [1-s.reflectance, s.transmittance], atol=2e-12)
        self.assertAlmostEqual(flux[0]-flux[1], s.absorption, places=11)
        for z in (0, -c.thickness):
            for derivative in (0, 1):
                assert_allclose(s.harmonics(z+1e-9, derivative=derivative),
                                s.harmonics(z-1e-9, derivative=derivative), atol=3e-6)
        self.assertLess(s.eigen_residual, 1e-12)

    def test_zero_and_full_openings(self):
        for fx, fy in ((0., .5), (.5, 0.), (1., 1.)):
            c = absorbing_screen_configuration(fill_x=fx, fill_y=fy, Mx=1, My=1)
            s = solve(c)
            self.assertAlmostEqual(s.energy_balance, 1., places=11)
            ideal = IdealAmplitudeGrating2D(c)
            if fx == fy == 1:
                self.assertEqual(s.absorption, 0.)
                assert_allclose(s.transmission, ideal.transmission, atol=2e-12)
                assert_allclose(ideal.exact_exit_field([-.75, 0], [-.75, 0]),
                                ideal.incident_reference, atol=2e-12)
            else:
                assert_allclose(ideal.transmission, 0.)

    def test_fft2_and_ideal_absolute_reference(self):
        c = absorbing_screen_configuration(kx_incident=.9, ky_incident=-.3, period_y=1.2, Mx=2, My=2)
        data = AbsorbingComparison2DData(c)
        x = .13+np.arange(40)*c.period_x/40
        y = -.27+np.arange(32)*c.period_y/32
        for distance in (0, .4):
            z = -c.thickness-distance
            values = data.solution.reconstruct(x, y, [z])[0]
            extracted = field_fourier_coefficients_2d(values, x, y, c)
            actual, ideal = data.coefficients(distance)
            assert_allclose(extracted, actual, atol=1e-12)
            ideal_values = data.ideal.reconstruct(x, y, [z])[0]
            assert_allclose(field_fourier_coefficients_2d(ideal_values, x, y, c), ideal, atol=1e-12)
        origin_field = data.ideal.exact_exit_field([0.], [0.])[0,0]
        self.assertAlmostEqual(origin_field, data.ideal.incident_reference)
        assert_allclose(data.ideal.transmission/data.ideal.incident_reference,
                        c.fill_x*c.fill_y*np.sinc(data.solution.orders[:,0]*c.fill_x)
                        *np.sinc(data.solution.orders[:,1]*c.fill_y), atol=1e-15)
        with self.assertRaises(ValueError):
            data.coefficients(-.1)
        with self.assertRaises(ValueError):
            field_fourier_coefficients_2d(values, np.linspace(0,c.period_x,40), y, c)

    def test_passive_indices(self):
        for bad in (1-.1j, -1+.1j, complex('nan'), [1.5], True):
            with self.assertRaises(ValueError):
                Config(n_background=bad)
        with self.assertRaises(ValueError):
            Config(n_incident=1+.1j)
        with self.assertRaises(ValueError):
            AbsorbingComparison2DData(Config())  # inclusion is not a clear exterior-index opening


if __name__ == '__main__':
    unittest.main()
