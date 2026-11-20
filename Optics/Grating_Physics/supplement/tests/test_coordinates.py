"""Physical equivalences and interactive edge cases for optical coordinates."""

import gc
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics import wavevector_coordinates


class CoordinateTests(unittest.TestCase):
    def test_representations_reconstruct_the_same_wave(self):
        for theta in (0, 37, 90, 143, 180):
            d = wavevector_coordinates(0.55, 1.5, theta, -123, 1.2)
            k = d['cartesian']
            assert_allclose(np.linalg.norm(k), 2*np.pi*1.5/0.55)
            assert_allclose(d['direction_cosines'] * d['k'], k)
            assert_allclose(2*np.pi*d['spatial_frequency'], k)
            assert_allclose(d['pupil']*d['k0']*1.2, k[:2])
            transverse, phi, kz = d['cylindrical']
            if phi is not None:
                angle = np.deg2rad(phi)
                assert_allclose([transverse*np.cos(angle), transverse*np.sin(angle), kz], k)

    def test_pupil_boundary_and_backward_projection(self):
        # sin(30 degrees)=1/2; NA/n=1/2 fixes the pupil edge.
        forward = wavevector_coordinates(0.6, 1.5, 30, 20, 0.75)
        backward = wavevector_coordinates(0.6, 1.5, 150, 20, 0.75)
        self.assertAlmostEqual(forward['pupil_radius'], 1)
        assert_allclose(forward['pupil'], backward['pupil'])
        self.assertTrue(forward['collected'])
        self.assertFalse(backward['collected'])
        outside = wavevector_coordinates(0.6, 1.5, 31, 20, 0.75)
        self.assertFalse(outside['collected'])
        self.assertEqual(outside['hemisphere'], '+z')

    def test_wavelength_scaling_and_pole_singularity(self):
        a = wavevector_coordinates(0.5, 1.4, 47, 65, 0.9)
        b = wavevector_coordinates(1.0, 1.4, 47, 65, 0.9)
        assert_allclose(a['cartesian'], 2*b['cartesian'])
        assert_allclose(a['pupil'], b['pupil'])
        for theta in (0, 180):
            d = wavevector_coordinates(0.5, theta_deg=theta)
            self.assertIsNone(d['spherical'][2])
            self.assertIsNone(d['cylindrical'][1])
            assert_allclose(d['pupil'], [0, 0])

    def test_invalid_optical_inputs(self):
        invalid = [dict(vacuum_wavelength=0), dict(refractive_index=0),
                   dict(numerical_aperture=1.1), dict(numerical_aperture=0),
                   dict(theta_deg=181), dict(phi_deg=float('nan'))]
        for values in invalid:
            args = dict(vacuum_wavelength=0.55)
            args.update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                wavevector_coordinates(**args)


class CoordinateWidgetTests(unittest.TestCase):
    def test_projection_switches_camera_and_na_clamping(self):
        from wave_optics.widgets import WavevectorCoordinateExplorer

        view = WavevectorCoordinateExplorer()
        try:
            view.figure.layout.scene.camera.eye = dict(x=2, y=-1, z=0.7)
            camera = view.figure.layout.scene.camera.to_plotly_json()
            for projection, expected in [
                ('cartesian', view.data['cartesian'][:2]),
                ('frequency', view.data['spatial_frequency'][:2]),
                ('pupil', view.data['pupil']),
                ('direction', view.data['direction_cosines'][:2]),
            ]:
                view.projection.value = projection
                trace = view.figure.data[-1]
                assert_allclose([trace.x[-1], trace.y[-1]], expected)
                self.assertEqual(view.figure.layout.scene.camera.to_plotly_json(), camera)
            view.index.value = 1.5
            view.na.value = 1.4
            view.index.value = 1.0
            self.assertLessEqual(view.na.value, view.index.value)
            self.assertEqual(view.data['numerical_aperture'], 1.0)
            view.theta.value = 140
            self.assertFalse(view.data['collected'])
            self.assertIn('Backward wave', view.readout.value)
            view.theta.value = 0
            self.assertIn('undefined at pole', view.readout.value)
        finally:
            view.close()
            del view
            gc.collect()


if __name__ == '__main__':
    unittest.main()
