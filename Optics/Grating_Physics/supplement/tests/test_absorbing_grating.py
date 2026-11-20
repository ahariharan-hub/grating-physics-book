"""Independent absorbing scalar benchmarks; run before the interactive lesson."""

from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from wave_optics.absorbing_grating import AbsorbingGratingConfiguration, solve_absorbing_grating
from wave_optics.scalar_grating import ScalarGratingConfiguration, solve_scalar_grating, _layer_basis


class AbsorbingGratingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = AbsorbingGratingConfiguration()
        cls.s = solve_absorbing_grating(cls.c)

    def test_uniform_absorbing_travelling_wave_branch_and_loss(self):
        # Local homogeneous-medium benchmark. Exterior scattering tests below
        # retain air; a globally absorbing incoming half-space is out of scope.
        n, h, k0 = 1.5+.8j, 2., 2*np.pi/.633
        depth = np.linspace(0, h, 31)
        f, g, df, dg = _layer_basis(np.array([n]), k0*depth, k0*h)
        assert_allclose(f[:, 0], np.exp(1j*k0*n*depth), atol=2e-15)
        assert_allclose(abs(f[:, 0]), np.exp(-k0*n.imag*depth), atol=2e-15)
        assert_allclose(df[:, 0]*k0, 1j*k0*n*f[:, 0], atol=2e-14)
        self.assertTrue(np.all(abs(g) <= 1+1e-15))
        loss = quad(lambda d: k0*np.imag(n*n)*np.exp(-2*k0*n.imag*d), 0, h)[0]
        self.assertAlmostEqual(loss, n.real*(1-np.exp(-2*k0*n.imag*h)), places=12)

    def test_complex_slab_matches_independent_fabry_perot_and_internal_field(self):
        for kappa in (.001, .8, 4.):
            for h in (.02, .3, 3.):
                c = replace(self.c, clear_width=0, kappa=kappa, thickness=h, M=4)
                s = solve_absorbing_grating(c)
                n = c.absorber_index
                r01 = (1-n)/(1+n)
                e = np.exp(2j*np.pi*n*h/c.wavelength)
                denominator = 1-r01*r01*e*e
                r = r01*(1-e*e)/denominator
                t = 4*n/(1+n)**2*e/denominator
                assert_allclose([s.reflection[c.M], s.transmission[c.M]], [r, t], atol=3e-14)
                assert_allclose(s.transmission[s.orders != 0], 0, atol=3e-14)
                # Independently anchored internal forward/backward waves.
                down = 2/(1+n)/denominator
                up_at_exit = -r01*down*e
                z = -h*np.array([.1, .4, .9])
                expected = down*np.exp(-2j*np.pi*n*z/c.wavelength)+up_at_exit*np.exp(2j*np.pi*n*(z+h)/c.wavelength)
                assert_allclose(s.harmonics(z)[:, c.M], expected, atol=3e-14)
                self.assertAlmostEqual(s.absorption, 1-abs(r)**2-abs(t)**2, places=10)

    def test_zero_absorption_matches_existing_lossless_geometry(self):
        c = replace(self.c, kappa=0, clear_width=.57)
        # Old ridge means the centred region. Here that region is the air opening.
        old = ScalarGratingConfiguration(wavelength=c.wavelength, period=c.period,
            thickness=c.thickness, duty_cycle=c.open_fraction, n_ridge=1., n_groove=c.n_absorber, M=c.M)
        a, b = solve_absorbing_grating(c), solve_scalar_grating(old)
        assert_allclose(a.transmission, b.transmission, atol=2e-14)
        assert_allclose(a.reflection, b.reflection, atol=2e-14)
        near = solve_absorbing_grating(replace(c, kappa=1e-9))
        assert_allclose(near.transmission, b.transmission, atol=5e-9)
        self.assertEqual(a.absorption, 0.)

    def test_all_air_and_complex_material_fourier_projection(self):
        c = replace(self.c, clear_width=self.c.period)
        s = solve_absorbing_grating(c)
        assert_allclose(s.reflection, 0, atol=1e-13)
        self.assertAlmostEqual(s.transmission[c.M], np.exp(2j*np.pi*c.thickness/c.wavelength))
        self.assertEqual(s.absorption, 0)
        c = replace(self.c, clear_width=.57)
        for m in (-8, -3, 0, 1, 6):
            def fun(x):
                return c.refractive_index(x, -c.thickness/2)[0, 0]**2*np.exp(-2j*np.pi*m*x/c.period)/c.period
            numerical = quad(lambda x: fun(x).real, -c.period/2, c.period/2, points=[-c.clear_width/2, c.clear_width/2])[0]
            numerical += 1j*quad(lambda x: fun(x).imag, -c.period/2, c.period/2, points=[-c.clear_width/2, c.clear_width/2])[0]
            assert_allclose(c.permittivity_coefficients(m), numerical, atol=3e-13)

    def test_volume_absorption_and_exterior_flux_balance(self):
        for c in (self.c, replace(self.c, opening_shift=.19), replace(self.c, kappa=2., thickness=.7)):
            s = solve_absorbing_grating(c)
            self.assertGreaterEqual(s.absorption, 0)
            self.assertLessEqual(s.power_balance, 1+1e-12)
            self.assertAlmostEqual(s.energy_balance, 1, places=10)
            assert_allclose(s.downward_flux([.1, 0]), 1-s.reflectance, atol=1e-12)
            assert_allclose(s.downward_flux([-c.thickness, -c.thickness-.2]), s.transmittance, atol=1e-12)
            self.assertTrue(np.all(np.diff(s.downward_flux(-np.linspace(0, c.thickness, 31))) <= 1e-11))
            self.assertLess(s.absorption_quadrature_error, 1e-8)
            assert_allclose(s.transmission_efficiency[s.output_kinds != 'propagating'], 0.)

    def test_interface_continuity_and_coupled_equation(self):
        s, c = self.s, self.c
        dz = 1e-7
        for boundary, step in ((0, dz), (-c.thickness, -dz)):
            u = s.harmonics(boundary-step)[0]
            du = s.harmonics(boundary-step, derivative=1)[0]
            assert_allclose(u+step*du, s.harmonics(boundary)[0], atol=4e-11)
            assert_allclose(du-step*s.coupling_matrix@u, s.harmonics(boundary, derivative=1)[0], atol=2e-8)
        z = np.array([-.03, -.15, -.28])
        dd = (s.harmonics(z+dz, derivative=1)-s.harmonics(z-dz, derivative=1))/(2*dz)
        assert_allclose(dd, -s.harmonics(z)@s.coupling_matrix.T, atol=2e-7)
        self.assertGreater(np.linalg.norm(s.coupling_matrix-s.coupling_matrix.conj().T), 1.)
        self.assertLess(s.eigen_residual, 1e-12)

    def test_complex_order_convergence(self):
        reference = solve_absorbing_grating(replace(self.c, M=72))
        orders = np.arange(-4, 5)
        errors = []
        for m in (8, 16, 24):
            s = solve_absorbing_grating(replace(self.c, M=m))
            errors.append(np.max(abs(s.transmission[orders+m]-reference.transmission[orders+72])))
        self.assertTrue(np.all(np.diff(errors) < 0))
        self.assertLess(errors[-1], 3e-5)
        # A substantially stronger contrast requires additional harmonics.
        reference = solve_absorbing_grating(replace(self.c, kappa=4, M=96))
        errors = []
        for m in (24, 48):
            s = solve_absorbing_grating(replace(self.c, kappa=4, M=m))
            errors.append(np.max(abs(s.transmission[orders+m]-reference.transmission[orders+96])))
        self.assertLess(errors[1], errors[0]/2)
        self.assertLess(errors[1], 2e-5)

    def test_entire_interactive_thickness_range_is_order_converged(self):
        orders = np.arange(-4, 5)
        for h in (.03, .08, .15, .3, .5, .75, 1.):
            s = solve_absorbing_grating(replace(self.c, thickness=h))
            reference = solve_absorbing_grating(replace(self.c, thickness=h, M=72))
            self.assertLess(np.max(abs(s.transmission[orders+24]-reference.transmission[orders+72])), 5e-5)

    def test_thickness_and_strong_absorption_stability(self):
        transmissions = []
        for h in (.03, .15, .3, .6, 1.):
            s = solve_absorbing_grating(replace(self.c, thickness=h))
            self.assertAlmostEqual(s.energy_balance, 1, places=10)
            transmissions.append(s.transmittance)
        self.assertGreater(np.ptp(transmissions), .1)
        s = solve_absorbing_grating(replace(self.c, kappa=8., thickness=30.))
        self.assertGreater(np.max(s._q.imag)*s._k0*30, 709)
        self.assertTrue(np.all(np.isfinite(s.reconstruct([-.5, 0, .5], [-30, -15, -.01]))))
        self.assertAlmostEqual(s.energy_balance, 1, places=10)

    def test_shift_covariance_and_validation(self):
        shifted = solve_absorbing_grating(replace(self.c, opening_shift=.19))
        factor = np.exp(-1j*self.s.kx*.19)
        assert_allclose(shifted.transmission, self.s.transmission*factor, atol=2e-13)
        assert_allclose(shifted.reflection, self.s.reflection*factor, atol=2e-13)
        for key, value in [('kappa', -.1), ('n_absorber', 0), ('clear_width', 2), ('M', True), ('thickness', 0)]:
            with self.assertRaises(ValueError):
                replace(self.c, **{key: value})
        with self.assertRaises(TypeError):
            solve_absorbing_grating({})


if __name__ == '__main__':
    unittest.main()
