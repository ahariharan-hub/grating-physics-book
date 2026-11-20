"""Physical benchmarks for the scalar scattering BVP (no notebook or plots)."""

from dataclasses import replace
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from wave_optics.scalar_grating import ScalarGratingConfiguration, solve_scalar_grating


def slab_coefficients(wavelength, thickness, ni, nl, no):
    """Independent scalar Fabry-Perot benchmark, t referenced to the bottom."""
    r01, r12 = (ni-nl)/(ni+nl), (nl-no)/(nl+no)
    phase = np.exp(2j*np.pi*nl*thickness/wavelength)
    denominator = 1+r01*r12*phase**2
    return ((r01+r12*phase**2)/denominator,
            (2*ni/(ni+nl))*(2*nl/(nl+no))*phase/denominator)


class ScalarGratingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ScalarGratingConfiguration()
        cls.solution = solve_scalar_grating(cls.config)

    def test_analytic_permittivity_matches_material_integrals(self):
        c = replace(self.config, duty_cycle=.37, n_groove=1.2)
        edge = c.duty_cycle*c.period/2
        for p in (-7, -2, -1, 0, 1, 2, 7):
            def integrand(x):
                n = c.refractive_index(x, -c.thickness/2)[0, 0]
                return n*n*np.cos(2*np.pi*p*x/c.period)/c.period
            numerical = quad(integrand, -c.period/2, c.period/2,
                             points=[-edge, edge], epsabs=1e-12)[0]
            assert_allclose(c.permittivity_coefficients(p), numerical, atol=1e-13)
        s = solve_scalar_grating(c)
        assert_allclose(s.coupling_matrix, s.coupling_matrix.conj().T, atol=1e-14)
        # A Galerkin convolution uses epsilon_(m-n), including p=+/-2M.
        self.assertAlmostEqual(s.coupling_matrix[0, -1].real,
                               (2*np.pi/c.wavelength)**2*c.permittivity_coefficients(-2*c.M).real)

    def test_geometry_is_periodic_and_has_explicit_interface_conventions(self):
        c = replace(self.config, n_incident=1.1, n_output=1.3)
        x = np.array([-.6, -.2, 0, .2, .6])
        z = [0., -.15, -.3]
        expected = [[1.1]*5, [1., 1.5, 1.5, 1.5, 1.], [1.3]*5]
        assert_allclose(c.refractive_index(x, z), expected)
        assert_allclose(c.refractive_index(x+3*c.period, z), expected)
        for duty, index in ((0., c.n_groove), (1., c.n_ridge)):
            uniform = replace(c, duty_cycle=duty)
            assert_allclose(uniform.refractive_index(x, -.15), index)
            assert_allclose(uniform.permittivity_coefficients(np.arange(-2, 3)),
                            [0, 0, index**2, 0, 0])

    def test_uniform_medium_has_only_forward_wave_with_correct_phase(self):
        for n in (1., 1.23):
            c = replace(self.config, n_incident=n, n_output=n, n_ridge=n, n_groove=n)
            s = solve_scalar_grating(c)
            k = 2*np.pi*n/c.wavelength
            assert_allclose(s.reflection, 0, atol=2e-14)
            expected = np.zeros(2*c.M+1, dtype=complex)
            expected[c.M] = np.exp(1j*k*c.thickness)
            assert_allclose(s.transmission, expected, atol=2e-14)
            z = np.array([1., 0., -.07, -.2, -.3, -.8])
            assert_allclose(s.reconstruct([-.4, .4], z),
                            np.repeat(np.exp(-1j*k*z)[:, None], 2, axis=1), atol=2e-14)

    def test_duty_endpoints_match_complex_analytic_slab(self):
        for duty, layer_n in ((0., 1.2), (1., 1.7)):
            for height in (.03, .3, 1.7):
                c = replace(self.config, duty_cycle=duty, n_groove=1.2, n_ridge=1.7,
                            n_incident=1.1, n_output=1.4, thickness=height)
                with self.subTest(duty=duty, thickness=height):
                    s = solve_scalar_grating(c)
                    r, t = slab_coefficients(c.wavelength, height, 1.1, layer_n, 1.4)
                    assert_allclose([s.reflection[c.M], s.transmission[c.M]], [r, t], atol=2e-14)
                    assert_allclose(s.reflection[s.orders != 0], 0, atol=1e-14)
                    assert_allclose(s.transmission[s.orders != 0], 0, atol=1e-14)
                    self.assertAlmostEqual(s.transmission_efficiency[c.M], 1.4/1.1*abs(t)**2, places=13)
                    # Inside slab: down/up amplitudes at z=0 from field and derivative.
                    down = ((1+r)+1.1/layer_n*(1-r))/2
                    up = ((1+r)-1.1/layer_n*(1-r))/2
                    z = np.linspace(-.9*height, -.1*height, 9)
                    k = 2*np.pi*layer_n/c.wavelength
                    assert_allclose(s.harmonics(z)[:, c.M],
                                    down*np.exp(-1j*k*z)+up*np.exp(1j*k*z), atol=3e-14)

    def test_zero_lateral_contrast_never_generates_higher_orders(self):
        for layer_n in (1., 1.4):
            c = replace(self.config, n_ridge=layer_n, n_groove=layer_n, duty_cycle=.29)
            s = solve_scalar_grating(c)
            assert_allclose(s.harmonics([.2, -.1, -.6])[:, s.orders != 0], 0, atol=1e-14)

    def test_complex_coefficients_converge_in_magnitude_and_phase(self):
        # Compare the same orders, including the first evanescent orders.
        orders = np.arange(-4, 5)
        reference = solve_scalar_grating(replace(self.config, M=60))
        target = np.r_[reference.reflection[orders+60], reference.transmission[orders+60]]
        self.assertGreater(np.min(abs(target)), 1e-3)  # phase is meaningful here
        errors = []
        for M in (5, 10, 20):
            s = solve_scalar_grating(replace(self.config, M=M))
            coefficients = np.r_[s.reflection[orders+M], s.transmission[orders+M]]
            errors.append((np.max(abs(coefficients-target)),
                           np.max(abs(abs(coefficients)-abs(target))),
                           np.max(abs(np.angle(coefficients*np.conj(target))))))
        errors = np.asarray(errors)
        self.assertTrue(np.all(errors[1:] < errors[:-1]/2))
        self.assertLess(errors[-1, 0], 1e-4)
        self.assertLess(errors[-1, 1], 1e-4)
        self.assertLess(errors[-1, 2], .003)

    def test_lossless_flux_balance_and_evanescent_exclusion(self):
        for c in (self.config, replace(self.config, duty_cycle=.37, n_ridge=2.2,
                                      n_incident=1.1, n_output=1.4, M=18)):
            s = solve_scalar_grating(c)
            self.assertAlmostEqual(s.power_balance, 1., places=12)
            assert_allclose(s.reflection_efficiency[s.incident_kinds != 'propagating'], 0.)
            assert_allclose(s.transmission_efficiency[s.output_kinds != 'propagating'], 0.)
            self.assertGreater(np.max(abs(s.transmission[s.output_kinds == 'evanescent'])), .001)
            # Net downward flux is constant even inside the modulated layer.
            z = [.2, -.03, -.13, -.27, -.5]
            a, derivative = s.harmonics(z), s.harmonics(z, derivative=1)
            flux = -np.imag(np.sum(a.conj()*derivative, axis=1))/s.beta_incident[c.M].real
            assert_allclose(flux, np.sum(s.transmission_efficiency), atol=2e-13)

    def test_reference_planes_preserve_propagating_magnitudes_and_rotate_phases(self):
        s, c = self.solution, self.config
        distance = .17
        near = s.at_reference_plane(-c.thickness)
        far = s.at_reference_plane(-c.thickness-distance)
        assert_allclose(far, near*np.exp(1j*s.beta_output*distance), atol=1e-15)
        propagating = s.output_kinds == 'propagating'
        assert_allclose(abs(far[propagating]), abs(near[propagating]), atol=1e-15)
        assert_allclose(np.angle(far[propagating]/near[propagating]),
                        np.angle(np.exp(1j*s.beta_output[propagating]*distance)), atol=1e-14)
        evanescent = s.output_kinds == 'evanescent'
        self.assertTrue(np.all(abs(far[evanescent]) < abs(near[evanescent])))
        # Isolate reflection above z=0 by removing the known incoming field.
        reflected_far = s.at_reference_plane(distance)
        reflected_far[c.M] -= np.exp(-1j*s.beta_incident[c.M]*distance)
        assert_allclose(reflected_far, s.reflection*np.exp(1j*s.beta_incident*distance), atol=2e-16)

    def test_field_fourier_extraction_recovers_reference_coefficients(self):
        s, c = self.solution, self.config
        x = np.arange(256)*c.period/256
        z = [.2, 0., -.12, -.3, -.37]
        field = s.reconstruct(x, z)
        extracted = np.fft.fft(field, axis=1)[:, s.orders % x.size]/x.size
        assert_allclose(extracted, s.harmonics(z), atol=8e-16)
        for row, plane in zip(extracted, z):
            assert_allclose(row, s.at_reference_plane(plane), atol=8e-16)

    def test_horizontal_interfaces_match_field_and_normal_derivative(self):
        s, c = self.solution, self.config
        dz = 1e-7
        for boundary, step in ((0., dz), (-c.thickness, -dz)):
            inside = boundary-step
            a = s.harmonics(inside)[0]
            da = s.harmonics(inside, derivative=1)[0]
            # Extrapolate from the layer; exterior evaluation alone cannot test matching.
            assert_allclose(a+step*da, s.harmonics(boundary)[0], atol=2e-12)
            assert_allclose(da-step*(s.coupling_matrix @ a),
                            s.harmonics(boundary, derivative=1)[0], atol=3e-10)

    def test_layer_field_obeys_coupled_helmholtz_equation(self):
        s = self.solution
        z, dz = np.array([-.05, -.15, -.25]), 1e-6
        second = (s.harmonics(z+dz, derivative=1)-s.harmonics(z-dz, derivative=1))/(2*dz)
        expected = -s.harmonics(z) @ s.coupling_matrix.T
        assert_allclose(second, expected, rtol=2e-7, atol=3e-8)

    def test_grazing_classification_and_zero_normal_flux(self):
        c = replace(self.config, wavelength=1., period=1., n_output=1.5, M=4)
        s = solve_scalar_grating(c)
        self.assertEqual(np.count_nonzero(s.incident_kinds == 'grazing'), 2)
        self.assertEqual(np.count_nonzero(s.output_kinds == 'grazing'), 0)
        assert_allclose(s.beta_incident[abs(s.orders) == 1], 0.)
        assert_allclose(s.reflection_efficiency[abs(s.orders) == 1], 0.)
        self.assertAlmostEqual(s.power_balance, 1., places=13)
        # A resolved physical change across threshold changes the classification.
        for factor, kind in ((1-1e-8, 'evanescent'), (1+1e-8, 'propagating')):
            shifted = solve_scalar_grating(replace(c, period=factor))
            self.assertTrue(np.all(shifted.incident_kinds[abs(s.orders) == 1] == kind))

    def test_internal_cutoff_retains_constant_and_linear_solutions(self):
        c = replace(self.config, wavelength=1., period=1/1.5, duty_cycle=1., M=3)
        s = solve_scalar_grating(c)
        r, t = slab_coefficients(c.wavelength, c.thickness, 1., 1.5, 1.)
        assert_allclose([s.reflection[c.M], s.transmission[c.M]], [r, t], atol=2e-14)
        self.assertTrue(np.all(np.isfinite(s.harmonics([-.05, -.15, -.25]))))
        # A patterned M=1 layer has an excited even mode at zero eigenvalue
        # when alpha_1^2=a-2*b^2/a (a=epsilon_0, b=epsilon_1).
        a, b = 1.625, 1.25/np.pi
        c = replace(self.config, M=1, period=self.config.wavelength/np.sqrt(a-2*b*b/a))
        cutoff = solve_scalar_grating(c)
        for factor in (1-1e-7, 1+1e-7):
            nearby = solve_scalar_grating(replace(c, period=c.period*factor))
            assert_allclose(nearby.transmission, cutoff.transmission, atol=2e-6)
        self.assertAlmostEqual(cutoff.power_balance, 1., places=13)

    def test_thin_layer_limit_is_a_single_scalar_interface(self):
        c = replace(self.config, thickness=1e-10, n_output=1.4)
        s = solve_scalar_grating(c)
        assert_allclose(s.reflection[c.M], (1-1.4)/(1+1.4), atol=2e-9)
        assert_allclose(s.transmission[c.M], 2/(1+1.4), atol=2e-9)
        assert_allclose(s.transmission[s.orders != 0], 0., atol=2e-9)

    def test_deep_layer_does_not_overflow_evanescent_modes(self):
        c = replace(self.config, thickness=30., M=30)
        s = solve_scalar_grating(c)
        # A naive growing transfer factor would exceed float64's exponent range.
        eigenvalues = np.linalg.eigvalsh(s.coupling_matrix)
        self.assertGreater(np.sqrt(-np.min(eigenvalues))*c.thickness, 709)
        self.assertTrue(np.all(np.isfinite(s.reconstruct([-.2, 0., .2], [0., -1., -15., -30., -31.]))))
        self.assertAlmostEqual(s.power_balance, 1., places=12)

    def test_unit_rescaling_and_order_symmetry(self):
        c, s = self.config, self.solution
        metres = replace(c, wavelength=c.wavelength*1e-6, period=c.period*1e-6,
                         thickness=c.thickness*1e-6)
        scaled = solve_scalar_grating(metres)
        assert_allclose(scaled.reflection, s.reflection, atol=2e-14)
        assert_allclose(scaled.transmission, s.transmission, atol=2e-14)
        assert_allclose(scaled.reconstruct([0., 2e-7], [-1e-7, -4e-7]),
                        s.reconstruct([0., .2], [-.1, -.4]), atol=3e-14)
        assert_allclose(s.reflection, s.reflection[::-1], atol=2e-14)
        assert_allclose(s.transmission, s.transmission[::-1], atol=2e-14)

    def test_unresolved_exact_grazing_problem_raises_useful_error(self):
        # A uniform grazing harmonic can be an arbitrary constant on both sides.
        c = replace(self.config, wavelength=1., period=1., n_ridge=1., M=1)
        with self.assertRaisesRegex(np.linalg.LinAlgError, 'grazing'):
            solve_scalar_grating(c)

    def test_near_singular_slab_warns_about_conditioning(self):
        # At exact grazing a half-wave slab supports an unforced Neumann mode.
        c = replace(self.config, wavelength=1., period=1., duty_cycle=1., M=2,
                    thickness=(1+1e-13)/(2*np.sqrt(1.5**2-1)))
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            s = solve_scalar_grating(c)
        self.assertGreater(s.condition_number, 1e12)
        self.assertTrue(any('Ill-conditioned' in str(w.message) for w in recorded))

    def test_configuration_and_sampling_validation(self):
        for name, value in (('wavelength', 0), ('period', -1), ('thickness', 0),
                            ('n_ridge', 1+.1j), ('n_output', np.inf), ('n_groove', -1),
                            ('duty_cycle', 1.1), ('duty_cycle', np.nan),
                            ('M', -1), ('M', 2.5), ('M', True)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                replace(self.config, **{name: value})
        with self.assertRaises(TypeError):
            solve_scalar_grating({})
        for invalid in ([], [[1., 2.]], [np.nan], [1+1j]):
            with self.assertRaises(ValueError):
                self.solution.harmonics(invalid)
            with self.assertRaises(ValueError):
                self.solution.reconstruct(invalid, -.5)
        with self.assertRaises(ValueError):
            self.solution.at_reference_plane([-.5, -.6])
        with self.assertRaises(ValueError):
            self.solution.harmonics(-.1, derivative=2)
        with self.assertRaises(ValueError):
            solve_scalar_grating(replace(self.config, n_ridge=1e200))
        with self.assertRaises(ValueError):
            self.solution.reconstruct([1e308], -.5)

    def test_single_order_shapes_and_immutable_results(self):
        s = solve_scalar_grating(replace(self.config, M=0))
        self.assertEqual(s.harmonics(-.1).shape, (1, 1))
        self.assertEqual(s.reconstruct(0., -.1).shape, (1, 1))
        self.assertEqual(s.at_reference_plane(-.1).shape, (1,))
        self.assertEqual(s.transmission.dtype, np.complex128)
        with self.assertRaises(ValueError):
            s.transmission[0] = 2
        self.assertLess(s.boundary_residual, 1e-14)


if __name__ == '__main__':
    unittest.main()
