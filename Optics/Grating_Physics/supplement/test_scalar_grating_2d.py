"""Physical regression checks; run python -m unittest discover -s supplement -p test_scalar_grating_2d.py."""
import unittest
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose
from wave_optics.scalar_grating import ScalarGratingConfiguration, solve_scalar_grating
from wave_optics.scalar_grating_2d import ScalarGrating2DConfiguration as Config, solve_scalar_grating_2d as solve


class Scalar2DTests(unittest.TestCase):
    def test_uniform_oblique_layer(self):
        c = Config(n_inclusion=1.4, n_background=1.4, n_output=1.2,
                   kx_incident=.9, ky_incident=-.6)
        s = solve(c)
        k0 = 2*np.pi/c.wavelength
        kt2 = c.kx_incident**2+c.ky_incident**2
        b0, b1, b2 = [np.sqrt((k0*n)**2-kt2) for n in (c.n_incident, 1.4, c.n_output)]
        r01, r12 = (b0-b1)/(b0+b1), (b1-b2)/(b1+b2)
        phase = np.exp(1j*b1*c.thickness)
        r = (r01+r12*phase**2)/(1+r01*r12*phase**2)
        t = (2*b0/(b0+b1))*(2*b1/(b1+b2))*phase/(1+r01*r12*phase**2)
        expected_r, expected_t = np.zeros(len(s.orders), complex), np.zeros(len(s.orders), complex)
        expected_r[s.incident_index], expected_t[s.incident_index] = r, t
        assert_allclose(s.reflection, expected_r, atol=2e-13)
        assert_allclose(s.transmission, expected_t, atol=2e-13)
        self.assertAlmostEqual(s.power_balance, 1., places=12)

    def test_one_dimensional_limit(self):
        c = Config(Mx=4, My=2, fill_y=1.)
        s = solve(c)
        old = solve_scalar_grating(ScalarGratingConfiguration(M=c.Mx))
        keep = s.orders[:, 1] == 0
        assert_allclose(s.transmission[keep], old.transmission, atol=2e-12)
        assert_allclose(s.reflection[keep], old.reflection, atol=2e-12)
        assert_allclose(s.transmission[~keep], 0., atol=2e-12)
        assert_allclose(s.reconstruct([.1,.3], [.2], [-.1,-.3])[:,0,:],
                        old.reconstruct([.1,.3], [-.1,-.3]), atol=2e-12)

    def test_interfaces_bloch_symmetry_and_flux(self):
        c = Config(period_y=1.2, fill_y=.35, kx_incident=.7, ky_incident=-.4)
        s = solve(c)
        self.assertAlmostEqual(s.power_balance, 1., places=11)
        for z in (0., -c.thickness):
            for d in (0, 1):
                assert_allclose(s.harmonics(z+1e-9, derivative=d),
                                s.harmonics(z-1e-9, derivative=d), atol=3e-6)
        assert_allclose(s.reconstruct([.1+c.period_x], [.2], [-.5]),
                        s.reconstruct([.1], [.2], [-.5])*np.exp(1j*c.kx_incident*c.period_x), atol=1e-12)
        swapped = solve(replace(c, period_x=c.period_y, period_y=c.period_x,
                               fill_x=c.fill_y, fill_y=c.fill_x,
                               kx_incident=c.ky_incident, ky_incident=c.kx_incident))
        assert_allclose(s.transmission.reshape(5,5), swapped.transmission.reshape(5,5).T, atol=2e-12)
        self.assertFalse(s.transmission.flags.writeable)

    def test_cutoff_mode_and_evanescent_decay(self):
        c = Config(wavelength=1., period_x=1., period_y=1.3, n_inclusion=1., n_background=1.)
        # A uniform medium at exact grazing has a nonunique homogeneous mode.
        with self.assertRaises(np.linalg.LinAlgError):
            solve(c)
        s = solve(replace(c, wavelength=.999))
        self.assertAlmostEqual(s.power_balance, 1., places=12)
        s = solve(Config())
        ev = s.output_kinds == 'evanescent'
        self.assertTrue(np.any(ev))
        self.assertTrue(np.all(abs(s.harmonics(-1.)[0,ev]) <= abs(s.harmonics(-.3)[0,ev])+1e-14))

    def test_validation(self):
        for kwargs in ({'Mx':-1}, {'My':True}, {'fill_x':1.1}, {'kx_incident':100.}, {'period_y':0.}):
            with self.assertRaises(ValueError):
                Config(**kwargs)
        with self.assertRaises(TypeError):
            solve(None)


if __name__ == '__main__':
    unittest.main()
