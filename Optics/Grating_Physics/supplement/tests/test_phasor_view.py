"""The selected physical point and its detached complex-plane view agree."""
import gc
import unittest
import numpy as np
from numpy.testing import assert_allclose
from ipywidgets.widgets.widget import _instances

from wave_optics.phasor_view import LocalPhasorExplorer
from wave_optics.widgets import PlaneWaveExplorer, ScalarPlaneWave3D


class DetachedPhasorTests(unittest.TestCase):
    def test_point_selects_phasor_and_time_only_rotates_optional_arrow(self):
        view = LocalPhasorExplorer(1)
        try:
            self.assertFalse(view.show_rotation.value)
            assert_allclose(view.phasor, 1j, atol=1e-14)
            before = view.spatial_field.copy()
            view.time.value = .25
            assert_allclose(view.phasor, 1j, atol=1e-14)
            assert_allclose(view.real_value, 1)
            self.assertGreater(np.max(abs(view.spatial_field-before)), .5)
            view.show_rotation.value = True
            assert_allclose(view.rotating, 1, atol=1e-14)
            assert_allclose(view.spatial_axis.lines[3].get_ydata(), [view.real_value])
            view.coordinates['x'].value = 0
            assert_allclose(view.phasor, 1)
            assert_allclose(view.real_value, 0, atol=1e-14)
            self.assertIn('Position x', view.spatial_axis.get_xlabel())
            self.assertEqual(view.complex_axis.get_xlabel(), 'Real amplitude / A')
            self.assertEqual(view.complex_axis.get_ylabel(), 'Imaginary amplitude / A')
            view.reset.click()
            self.assertFalse(view.show_rotation.value)
            assert_allclose(view.phasor, 1j, atol=1e-14)
        finally:
            view.close()

    def test_two_and_three_coordinates_produce_one_local_phase(self):
        for dimensions in (2, 3):
            view = LocalPhasorExplorer(dimensions)
            try:
                coordinate = 'z' if dimensions == 2 else 'y'
                view.coordinates[coordinate].value = .5
                assert_allclose(view.phasor, 1j, atol=1e-14)
                view.coordinates['x'].value = .7
                assert_allclose(view.phasor, 1j, atol=1e-14)
                view.time.value = .25
                assert_allclose(view.real_value, 1, atol=1e-14)
                expected = np.cos(view.spatial_points @ view.k_vector - np.pi/2)
                assert_allclose(view.spatial_field, expected, atol=1e-14)
                if dimensions == 3:
                    assert_allclose(view.spatial_points[..., 1], .5)
                    assert_allclose(view.spatial_field, 1, atol=1e-14)
            finally:
                view.close()

    def test_optional_grid_is_fixed_by_default_and_can_restore_time(self):
        for dimensions in (1, 2, 3):
            view = LocalPhasorExplorer(dimensions, grid=True)
            try:
                self.assertFalse(view.show_rotation.value)
                initial = bytes(view.picture.value)
                fixed = [values.copy() for values in view.grid_plot.values]
                view.time.value = .25
                self.assertEqual(bytes(view.picture.value), initial)
                view.show_rotation.value = True
                for before, after in zip(fixed, view.grid_plot.values):
                    assert_allclose(after, -1j*before, atol=1e-14)
            finally:
                view.close()

    def test_physical_explorers_and_detached_views_release_only_owned_widgets(self):
        baseline = set(_instances)
        for dimensions in (1, 2, 3):
            wave = (ScalarPlaneWave3D(include_phasors=False) if dimensions == 3 else
                    PlaneWaveExplorer(dimensions, include_phasors=False))
            view = LocalPhasorExplorer(dimensions)
            try:
                self.assertIsNone(wave.phasor)
                self.assertIsNone(wave.views)
                original = bytes(wave.picture.value)
                view.time.value = .25
                view.wavelength.value = 3
                self.assertEqual(bytes(wave.picture.value), original)
                value = view.phasor
                wave.wavelength.value = 1.5
                self.assertEqual(view.phasor, value)
                view.close()
                view.close()
                self.assertIsNone(view.ui.comm)
                self.assertIsNone(view.figure)
                self.assertTrue(bytes(wave.picture.value).startswith(b'\x89PNG'))
                view.time.value = .5  # Detached callbacks remain inert.
                view.reset.click()
                self.assertEqual(view.time.value, .5)
            finally:
                view.close()
                wave.close()
                wave.close()
                del view, wave
                gc.collect()
            self.assertEqual(set(_instances), baseline)

if __name__ == '__main__':
    unittest.main()
