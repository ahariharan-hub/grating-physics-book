"""Physical and numerical checks for the introductory scalar-wave models."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.scalar_theory import (
    boundary_family, helmholtz_trial, scalar_solution, slit_far_field, slit_propagation,
)


class ScalarTheoryTests(unittest.TestCase):
    def test_medium_wavenumber_selects_two_travelling_wave_directions(self):
        x = np.linspace(-2, 2, 57)
        wavelength, amplitude = 1.7, .8
        k = 2*np.pi/wavelength
        for ratio in (-1, 0, .6, 1, 1.4):
            with self.subTest(spatial_ratio=ratio):
                data = helmholtz_trial(x, wavelength, ratio, amplitude, 37)
                self.assertEqual(data['k'], k)
                self.assertEqual(data['q'], ratio*k)
                assert_allclose(abs(data['field']), amplitude)
                assert_allclose(data['residual'], (1-ratio**2)*k**2*data['field'], atol=1e-13)
                if abs(ratio) == 1:
                    assert_allclose(data['residual'], 0, atol=1e-13)
                else:
                    self.assertGreater(np.min(abs(data['residual'])), 1)
        # Finite differences independently check the analytic second derivative.
        h = 1e-4*wavelength
        centre = helmholtz_trial(x, wavelength, .6, amplitude, 37)
        right = helmholtz_trial(x+h, wavelength, .6, amplitude, 37)['field']
        left = helmholtz_trial(x-h, wavelength, .6, amplitude, 37)['field']
        assert_allclose((right-2*centre['field']+left)/h**2,
                        centre['second'], rtol=1e-6, atol=1e-7)

    def test_standing_wave_is_two_opposite_plane_waves_with_common_phase(self):
        points = np.array([[[0, 0, 0], [1, -2, .5], [3, 4, 1]],
                           [[-1, 2, 0], [5, 3, .5], [0, 0, 1]]], dtype=float)
        amplitude, phase = 1.4, 61
        forward = scalar_solution(points, amplitude=amplitude/2, phase_deg=phase)
        reversed_points = points.copy()
        reversed_points[..., 2] *= -1
        backward = scalar_solution(reversed_points, amplitude=amplitude/2, phase_deg=phase)
        standing = scalar_solution(points, kind='standing', amplitude=amplitude, phase_deg=phase)
        self.assertEqual(standing.shape, (2, 3))
        assert_allclose(standing, forward+backward, atol=1e-14)
        # Transverse translations leave these plane waves unchanged.
        assert_allclose(forward[0], forward[1], atol=1e-14)
        # Half a wavelength separates opposite antinodes; a quarter gives a node.
        assert_allclose(standing[0, 1], 0, atol=1e-14)
        assert_allclose(standing[0, 2], -standing[0, 0], atol=1e-14)
        self.assertEqual(np.ndim(scalar_solution([0, 0, 1])), 0)

    def test_spherical_wave_is_radial_with_inverse_radius_amplitude_and_excluded_source(self):
        wavelength, amplitude, phase = 1.7, .8, 31
        points = wavelength*np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1],
                                      [2, 0, 0], [0, 0, 3], [0, 0, 0]])
        field = scalar_solution(points, wavelength, 'spherical', amplitude, phase)
        assert_allclose(field[:3], amplitude*np.exp(1j*np.deg2rad(phase)), atol=1e-14)
        assert_allclose(abs(field[:5]), [amplitude]*3+[amplitude/2, amplitude/3])
        self.assertTrue(np.isnan(field[-1].real))
        self.assertTrue(np.isnan(field[-1].imag))
        # Zero amplitude does not turn the excluded source coordinate into data.
        self.assertTrue(np.isnan(scalar_solution([0, 0, 0], kind='spherical', amplitude=0)))
        assert_allclose(scalar_solution([1, 0, 0], kind='spherical', amplitude=0), 0)

    def test_spherical_solution_satisfies_radial_helmholtz_with_mesh_convergence(self):
        wavelength = 1.7
        k = 2*np.pi/wavelength
        radii = wavelength*np.array([.9, 1.3, 2.5])

        def radial_field(radius):
            points = np.column_stack((radius, np.zeros_like(radius), np.zeros_like(radius)))
            return scalar_solution(points, wavelength, 'spherical', .8, 31)

        field = radial_field(radii)
        errors = []
        for h in wavelength/np.array([40, 80, 160]):
            left, right = radial_field(radii-h), radial_field(radii+h)
            first = (right-left)/(2*h)
            second = (right-2*field+left)/h**2
            # For a radial 3D field, the Laplacian includes the geometric term.
            residual = second+2*first/radii+k*k*field
            errors.append(np.max(abs(residual)/(k*k*abs(field))))
        self.assertLess(errors[-1], .001)
        self.assertTrue(np.all(np.asarray(errors[:-1])/errors[1:] > 3.8))
        self.assertTrue(np.all(np.asarray(errors[:-1])/errors[1:] < 4.2))
        self.assertGreater(np.max(abs(second+k*k*field)/(k*k*abs(field))), .1)

    def test_boundary_value_alone_does_not_determine_the_normal_derivative(self):
        phase = 47
        boundary_value = np.exp(1j*np.deg2rad(phase))
        derivatives = []
        for fraction in (0, .25, .5, .8, 1):
            data = boundary_family(0, backward_fraction=fraction, phase_deg=phase)
            assert_allclose(data['field'], boundary_value, atol=1e-14)
            assert_allclose(data['derivative'], 1j*np.pi*(1-2*fraction)*boundary_value, atol=1e-14)
            self.assertEqual(data['forward_amplitude']+data['backward_amplitude'], 1)
            derivatives.append(data['derivative'])
        self.assertGreater(abs(derivatives[0]-derivatives[-1]), 6)
        z, h = np.array([-.9, 0, .7]), 1e-5
        centre = boundary_family(z, backward_fraction=.3, phase_deg=phase)
        left = boundary_family(z-h, backward_fraction=.3, phase_deg=phase)['field']
        right = boundary_family(z+h, backward_fraction=.3, phase_deg=phase)['field']
        assert_allclose((right-left)/(2*h), centre['derivative'], rtol=1e-8, atol=1e-9)

    def test_boundary_family_real_fields_travel_in_the_declared_directions(self):
        z = np.linspace(-3, 3, 101)
        wavelength, tau = 1.7, .3
        for fraction, direction in ((0, 1), (1, -1)):
            initial = boundary_family(z, wavelength, fraction, 31, tau=0)
            following = boundary_family(z+direction*wavelength*tau, wavelength, fraction, 31, tau=tau)
            assert_allclose(following['real_field'], initial['real_field'], atol=1e-14)
        standing = boundary_family(z, wavelength, .5, 31, tau)
        assert_allclose(standing['real_field'],
                        np.cos(2*np.pi*z/wavelength)*np.cos(np.deg2rad(31)-2*np.pi*tau), atol=1e-14)

    def test_slit_formula_matches_numerical_aperture_fourier_integral(self):
        width, wavelength, distance = 31, .55, 750
        first_zero = wavelength*distance/width
        x_mm = first_zero*np.array([-1.45, -1, -.35, 0, .35, 1, 1.45])
        aperture_um = np.linspace(-width/2, width/2, 20001)
        # xi and lambda are micrometres; x/L is dimensionless using millimetres.
        phase = -2*np.pi*aperture_um[None, :]*x_mm[:, None]/(wavelength*distance)
        quadrature = np.trapz(np.exp(1j*phase), aperture_um, axis=1)
        data = slit_far_field(x_mm, width, wavelength, distance)
        assert_allclose(data['amplitude_scaled'], quadrature, atol=3e-7)
        assert_allclose(data['relative_intensity'], abs(quadrature/width)**2, atol=1e-8)
        assert_allclose(data['first_zero_mm'], first_zero)
        assert_allclose(data['relative_intensity'][[1, 5]], 0, atol=1e-28)
        self.assertEqual(data['relative_intensity'][3], 1)

    def test_wider_slit_narrows_relative_pattern_but_increases_scaled_central_amplitude(self):
        x = np.linspace(-60, 60, 31)
        narrow = slit_far_field(2*x, width_um=20)
        wide = slit_far_field(x, width_um=40)
        assert_allclose(wide['relative_intensity'], narrow['relative_intensity'])
        assert_allclose(wide['amplitude_scaled'], 2*narrow['amplitude_scaled'])
        self.assertEqual(wide['first_zero_mm'], narrow['first_zero_mm']/2)
        self.assertEqual(wide['fresnel_number'], 4*narrow['fresnel_number'])
        assert_allclose(narrow['fresnel_number'], 20**2/(.6*1_000_000))
        self.assertEqual(slit_far_field(0, width_um=20)['amplitude_scaled'], 20)
        self.assertEqual(slit_far_field(0, width_um=40)['amplitude_scaled'], 40)

    def test_invalid_coordinates_and_parameters_are_rejected(self):
        wavelength_calls = (
            lambda value: helmholtz_trial(0, wavelength=value),
            lambda value: scalar_solution([0, 0, 1], wavelength=value),
            lambda value: boundary_family(0, wavelength=value),
            lambda value: slit_far_field(0, wavelength_um=value),
        )
        for call in wavelength_calls:
            for value in (0, -1, np.nan, np.inf, [1, 2], 1+1j):
                with self.subTest(call=call, wavelength=value), self.assertRaises(ValueError):
                    call(value)
        invalid_calls = (
            lambda: helmholtz_trial([0, np.nan]),
            lambda: helmholtz_trial(np.array([1+1j])),
            lambda: helmholtz_trial(0, spatial_ratio=np.inf),
            lambda: helmholtz_trial(0, amplitude=-1),
            lambda: helmholtz_trial(0, phase_deg=np.nan),
            lambda: scalar_solution([0, 0]),
            lambda: scalar_solution(1),
            lambda: scalar_solution([[0, 1, np.inf]]),
            lambda: scalar_solution([0, 0, 1], amplitude=-1),
            lambda: scalar_solution([0, 0, 1], kind='unknown'),
            lambda: scalar_solution([0, 0, 1], phase_deg=np.inf),
            lambda: boundary_family(np.inf),
            lambda: boundary_family(0, backward_fraction=-.1),
            lambda: boundary_family(0, backward_fraction=1.1),
            lambda: boundary_family(0, backward_fraction=np.nan),
            lambda: boundary_family(0, tau=np.inf),
            lambda: slit_far_field(np.nan),
            lambda: slit_far_field(0, width_um=0),
            lambda: slit_far_field(0, width_um=np.inf),
            lambda: slit_far_field(0, distance_mm=-1),
            lambda: slit_far_field(0, distance_mm=np.nan),
        )
        for index, call in enumerate(invalid_calls):
            with self.subTest(invalid_case=index), self.assertRaises(ValueError):
                call()

    def test_slit_propagation_recovers_boundary_and_preserves_transverse_symmetry(self):
        data = slit_propagation(20, .6, distances_um=[0, 20, 1000])
        x, field = data['x_um'], data['field']
        assert_allclose(field[0, abs(x) < 10], 1)
        assert_allclose(field[0, abs(x) > 10], 0)
        assert_allclose(field[0, abs(x) == 10], .5)
        assert_allclose(field, field[:, ::-1], atol=1e-12)
        uniform = slit_propagation(20, .6, distances_um=[0, 500, 1000])
        assert_allclose(uniform['field'][-1], field[-1], atol=1e-11)
        assert_allclose(data['intensity'], abs(field)**2)
        self.assertGreater(data['intensity'][1, np.argmin(abs(x-11))], .01)

    def test_slit_map_is_stable_under_padding_and_sampling_refinement(self):
        for width, wavelength in ((10, .4), (10, .8), (40, .4), (40, .8)):
            with self.subTest(width=width, wavelength=wavelength):
                options = dict(width_um=width, wavelength_um=wavelength,
                               distances_um=[0, 2, 20, 200, 1000])
                base = slit_propagation(**options)
                padded = slit_propagation(**options, window_um=8192)
                finer = slit_propagation(**options, dx_um=.03125)
                # Absolute differences use incident intensity = 1, not local
                # relative errors at dark fringes. The map is an illustration.
                assert_allclose(base['intensity'], padded['intensity'], atol=.006, rtol=0)
                assert_allclose(base['intensity'], finer['intensity'], atol=.004, rtol=0)

    def test_slit_angular_spectrum_approaches_the_paraxial_far_field(self):
        data = slit_propagation(10, .8, distances_um=[1000])
        intensity = data['intensity'][0]
        center = np.argmin(abs(data['x_um']))
        paraxial = np.sinc(10*data['x_um']/(.8*1000))**2
        assert_allclose(intensity/intensity[center], paraxial, atol=.01, rtol=0)



if __name__ == '__main__':
    unittest.main()
