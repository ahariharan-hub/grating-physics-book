"""Independent checks of exact Helmholtz families and a paraxial beam."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.helmholtz_families import gaussian_beam, helmholtz_family


class HelmholtzFamilyTests(unittest.TestCase):
    def test_three_dimensional_laplacian_converges_for_every_family(self):
        wavelength = 1.7
        k = 2*np.pi/wavelength
        points = wavelength*np.array([[.3, .25, .35], [.7, -.2, .65], [-.45, .1, .9]])
        for kind in ('cylindrical', 'bessel', 'multipole', 'evanescent'):
            with self.subTest(kind=kind):
                parameters = dict(wavelength=wavelength, kind=kind, amplitude=.8,
                                  phase_deg=37, transverse_ratio=.63,
                                  evanescent_ratio=1.35, degree=2)
                field = helmholtz_family(points, **parameters)
                errors = []
                for step in wavelength/np.array([30, 60, 120]):
                    laplacian = np.zeros_like(field)
                    for axis in np.eye(3):
                        above = helmholtz_family(points+step*axis, **parameters)
                        below = helmholtz_family(points-step*axis, **parameters)
                        laplacian += (above-2*field+below)/step**2
                    errors.append(np.max(abs(laplacian+k*k*field))/(k*k*np.max(abs(field))))
                self.assertLess(errors[-1], .001)
                # A vanishing residual at the expected central-difference rate
                # checks the PDE independently of the closed-form code.
                rates = np.asarray(errors[:-1])/errors[1:]
                self.assertTrue(np.all(rates > 3.8), (kind, errors))
                self.assertTrue(np.all(rates < 4.2), (kind, errors))

    def test_cylindrical_wave_is_y_invariant_outgoing_and_axis_singular(self):
        wavelength, amplitude, phase = 1.7, .8, 31
        points = np.array([[wavelength, 0, 0], [wavelength, 21, 0],
                           [0, 7, wavelength], [0, 1, 0]])
        field = helmholtz_family(points, wavelength, 'cylindrical', amplitude, phase)
        assert_allclose(field[:3], np.full(3, field[0]), atol=1e-14)
        assert_allclose(abs(field[:3]), amplitude)
        self.assertTrue(np.isnan(field[-1].real))
        self.assertTrue(np.isnan(field[-1].imag))
        self.assertTrue(np.isnan(helmholtz_family([0, -5, 0], kind='cylindrical', amplitude=0)))
        radii = wavelength*np.array([20, 80])
        distant = helmholtz_family(np.column_stack((radii, np.zeros((2, 2)))),
                                   wavelength, 'cylindrical', amplitude, phase)
        # Far from the axis, amplitude scales as 1/sqrt(rho), not 1/rho.
        assert_allclose(abs(distant[0])/abs(distant[1]), 2, rtol=1e-4)
        nearby = helmholtz_family([[30*wavelength, 0, 0], [30.125*wavelength, 0, 0]],
                                  wavelength, 'cylindrical')
        assert_allclose(np.angle(nearby[1]/nearby[0]), np.pi/4, atol=2e-5)

    def test_bessel_wave_equals_an_azimuthal_cone_of_plane_waves(self):
        wavelength, ratio, amplitude, phase = 1.7, .63, .8, 37
        k = 2*np.pi/wavelength
        azimuth = np.linspace(0, 2*np.pi, 2048, endpoint=False)
        vectors = k*np.column_stack((ratio*np.cos(azimuth), ratio*np.sin(azimuth),
                                     np.full_like(azimuth, np.sqrt(1-ratio**2))))
        assert_allclose(np.linalg.norm(vectors, axis=1), k, atol=1e-14)
        points = np.array([[0, 0, 0], [.7, -.3, 1.2], [-1.1, .4, -.8], [0, 0, 2]])
        numerical_cone = amplitude*np.mean(
            np.exp(1j*(points@vectors.T+np.deg2rad(phase))), axis=-1)
        field = helmholtz_family(points, wavelength, 'bessel', amplitude, phase,
                                 transverse_ratio=ratio)
        assert_allclose(field, numerical_cone, atol=1e-14)
        plane = helmholtz_family(points, wavelength, 'bessel', amplitude, phase,
                                 transverse_ratio=0)
        assert_allclose(plane, amplitude*np.exp(1j*(k*points[:, 2]+np.deg2rad(phase))), atol=1e-14)
        transverse = helmholtz_family(points, wavelength, 'bessel', transverse_ratio=1)
        shifted = points.copy()
        shifted[:, 2] += 7
        assert_allclose(helmholtz_family(shifted, wavelength, 'bessel', transverse_ratio=1), transverse)
        field_shifted = helmholtz_family(shifted, wavelength, 'bessel', amplitude, phase,
                                         transverse_ratio=ratio)
        assert_allclose(abs(field_shifted), abs(field), atol=1e-14)

    def test_regular_multipole_angular_shapes_and_origin_for_degrees_zero_one_two(self):
        wavelength, amplitude, phase = 1.7, 1.2, 37
        radius = .37*wavelength
        argument = 2*np.pi*radius/wavelength
        cosines = np.array([-1, -.5, 0, .5, 1])
        points = radius*np.column_stack((np.sqrt(1-cosines**2), np.zeros(5), cosines))
        radial = [np.sin(argument)/argument,
                  np.sin(argument)/argument**2-np.cos(argument)/argument,
                  (3/argument**3-1/argument)*np.sin(argument)-3*np.cos(argument)/argument**2]
        angular = [np.ones_like(cosines), cosines, (3*cosines**2-1)/2]
        for degree in range(3):
            with self.subTest(degree=degree):
                field = helmholtz_family(points, wavelength, 'multipole', amplitude, phase, degree=degree)
                assert_allclose(field, amplitude*radial[degree]*angular[degree]
                                *np.exp(1j*np.deg2rad(phase)), atol=1e-14)
                origin = helmholtz_family([0, 0, 0], wavelength, 'multipole', amplitude, phase, degree=degree)
                expected = amplitude*np.exp(1j*np.deg2rad(phase)) if degree == 0 else 0
                assert_allclose(origin, expected)
                self.assertTrue(np.isfinite(origin))
                # m=0 has no azimuthal dependence.
                rotated = points[:, [1, 0, 2]]
                assert_allclose(helmholtz_family(rotated, wavelength, 'multipole', amplitude,
                                                phase, degree=degree), field, atol=1e-14)

    def test_evanescent_decay_and_complex_wavevector_satisfy_the_same_equation(self):
        wavelength, ratio, amplitude = 1.7, 1.25, .7
        k = 2*np.pi/wavelength
        q, alpha = ratio*k, k*np.sqrt(ratio**2-1)
        wavevector = np.array([q, 0, 1j*alpha])
        # The dispersion relation uses the bilinear dot, without conjugation.
        assert_allclose(wavevector@wavevector, k*k)
        self.assertGreater(np.vdot(wavevector, wavevector).real, k*k)
        z = np.array([0, .2, .6, 1.1])
        points = np.column_stack((np.full_like(z, .3), np.zeros_like(z), z))
        field = helmholtz_family(points, wavelength, 'evanescent', amplitude, 31,
                                 evanescent_ratio=ratio)
        assert_allclose(abs(field)**2, amplitude**2*np.exp(-2*alpha*z), atol=1e-14)
        assert_allclose(field/field[0], np.exp(-alpha*z), atol=1e-14)
        shifted = points+np.array([wavelength/ratio, 4, 0])
        assert_allclose(helmholtz_family(shifted, wavelength, 'evanescent', amplitude, 31,
                                        evanescent_ratio=ratio), field, atol=1e-14)

    def test_validation_checks_each_family_domain_without_restricting_unused_parameters(self):
        invalid = (
            dict(kind='unknown'),
            dict(wavelength=0), dict(wavelength=np.inf), dict(amplitude=-1), dict(phase_deg=np.nan),
            dict(kind='bessel', transverse_ratio=-.1), dict(kind='bessel', transverse_ratio=1.1),
            dict(kind='bessel', transverse_ratio=np.nan),
            dict(kind='evanescent', evanescent_ratio=1), dict(kind='evanescent', evanescent_ratio=.9),
            dict(kind='evanescent', evanescent_ratio=np.inf),
            dict(kind='multipole', degree=-1), dict(kind='multipole', degree=1.5),
            dict(kind='multipole', degree=9), dict(kind='multipole', degree=np.nan),
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                helmholtz_family([1, 0, 1], **parameters)
        for points in ([0, 1], [[0, np.inf, 1]], np.array([0, 1j, 1])):
            with self.subTest(points=points), self.assertRaises(ValueError):
                helmholtz_family(points)
        with self.assertRaises(ValueError):
            helmholtz_family([1, 0, -.1], kind='evanescent')
        self.assertTrue(np.isfinite(helmholtz_family([1, 0, -.1], kind='bessel')))
        self.assertTrue(np.isfinite(helmholtz_family([1, 0, 1], kind='cylindrical',
                                                    degree=np.nan, transverse_ratio=-1,
                                                    evanescent_ratio=0)))


class GaussianBeamTests(unittest.TestCase):
    def test_waist_profile_rayleigh_width_and_gouy_phase(self):
        wavelength, amplitude, phase = 1.7, .8, 31
        waist = 2.5*wavelength
        k = 2*np.pi/wavelength
        rayleigh = k*waist**2/2
        parameters = dict(wavelength=wavelength, amplitude=amplitude,
                          phase_deg=phase, waist_ratio=2.5)
        radii = waist*np.array([0, .5, 1, 2])
        at_waist = np.column_stack((radii, np.zeros((len(radii), 2))))
        field = gaussian_beam(at_waist, **parameters)
        assert_allclose(field, amplitude*np.exp(-(radii/waist)**2)
                        *np.exp(1j*np.deg2rad(phase)), atol=1e-14)
        for sign in (-1, 1):
            depth = sign*rayleigh
            points = [[0, 0, depth], [np.sqrt(2)*waist, 0, depth]]
            field = gaussian_beam(points, **parameters)
            assert_allclose(abs(field), amplitude/np.sqrt(2)*np.array([1, np.exp(-1)]))
            # Remove the carrier and initial phase to isolate the on-axis Gouy phase.
            envelope = field[0]*np.exp(-1j*(k*depth+np.deg2rad(phase)))
            assert_allclose(envelope, amplitude/np.sqrt(2)*np.exp(-sign*1j*np.pi/4), atol=1e-14)
            # At the 1/e radius and z=+/-zR, the curvature contributes +/-1 radian.
            assert_allclose(field[1]/field[0], np.exp(-1+sign*1j), atol=1e-14)

    def test_integrated_transverse_intensity_is_conserved_during_spreading(self):
        wavelength, ratio, amplitude = 1.3, 2, .7
        waist = ratio*wavelength
        rayleigh = np.pi*waist**2/wavelength
        expected = np.pi*amplitude**2*waist**2/2
        for depth in (-2*rayleigh, 0, rayleigh, 2*rayleigh):
            width = waist*np.sqrt(1+(depth/rayleigh)**2)
            radius = np.linspace(0, 5*width, 8001)
            points = np.column_stack((radius, np.zeros_like(radius), np.full_like(radius, depth)))
            field = gaussian_beam(points, wavelength, amplitude, waist_ratio=ratio)
            integral = 2*np.pi*np.trapz(radius*abs(field)**2, radius)
            assert_allclose(integral, expected, rtol=2e-7)

    def test_envelope_satisfies_paraxial_equation_but_not_full_helmholtz_equation(self):
        wavelength, ratio = 1.7, 2
        waist = ratio*wavelength
        k = 2*np.pi/wavelength
        rayleigh = k*waist**2/2
        points = np.array([[.2*waist, .3*waist, .2*rayleigh],
                           [-.7*waist, .4*waist, .9*rayleigh]])

        def field(coordinates):
            return gaussian_beam(coordinates, wavelength=wavelength, waist_ratio=ratio)

        def envelope(coordinates):
            return field(coordinates)*np.exp(-1j*k*coordinates[..., 2])

        central = envelope(points)
        errors = []
        for fractional_step in (.08, .04, .02):
            transverse_step = fractional_step*waist
            axial_step = fractional_step*rayleigh
            transverse_laplacian = np.zeros_like(central)
            for direction in np.eye(3)[:2]:
                transverse_laplacian += (envelope(points+transverse_step*direction)-2*central
                                          +envelope(points-transverse_step*direction))/transverse_step**2
            axial = np.array([0, 0, axial_step])
            derivative = (envelope(points+axial)-envelope(points-axial))/(2*axial_step)
            errors.append(waist**2*np.max(abs(transverse_laplacian+2j*k*derivative)))
        self.assertLess(errors[-1], .003)
        rates = np.asarray(errors[:-1])/errors[1:]
        self.assertTrue(np.all((rates > 3.8) & (rates < 4.2)), errors)

        # The full Helmholtz residual retains the envelope's axial second
        # derivative. On axis at the waist, psi=1/(1+i*z/zR), so psi''=-2/zR^2.
        origin = np.zeros(3)
        step = wavelength/2048
        laplacian = sum((field(origin+step*axis)-2*field(origin)
                         +field(origin-step*axis))/step**2 for axis in np.eye(3))
        relative_residual = (laplacian+k*k*field(origin))/(k*k)
        expected_residual = -2/(k*rayleigh)**2
        assert_allclose(relative_residual, expected_residual, atol=1e-6)
        self.assertGreater(abs(relative_residual), 1e-4)

    def test_validation_rejects_invalid_beam_parameters_and_coordinates(self):
        for parameters in (dict(wavelength=0), dict(amplitude=-1), dict(phase_deg=np.nan),
                           dict(waist_ratio=1.99), dict(waist_ratio=np.nan),
                           dict(waist_ratio=np.inf)):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                gaussian_beam([0, 0, 0], **parameters)
        for points in ([0, 1], [[0, np.inf, 1]], [0, 1j, 1]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                gaussian_beam(points)


if __name__ == '__main__':
    unittest.main()
