"""Surface-phase physics and resource ownership of the introductory explorer."""

import gc
import unittest
import weakref

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from wave_optics.grating_intro import (
    amplitude_grating, GratingSurfaceExplorer, OrderSurfaceExplorer,
    order_surface_intersections,
)


class AmplitudeGratingTests(unittest.TestCase):
    def test_openings_repeat_without_changing_their_width(self):
        x = np.array([-.75, -.25, 0, .25, .75])
        expected = [0, 1, 1, 1, 0]
        for shift in (-6, -2, 0, 2, 8):
            assert_array_equal(amplitude_grating(x + shift, 2, 1), expected)
        assert_array_equal(amplitude_grating(x, 2, .25), [0, 0, 1, 0, 0])

    def test_closed_open_and_edge_conventions(self):
        x = np.array([-2, -1, -.5, 0, .5, 1, 2])
        assert_array_equal(amplitude_grating(x, 2, 0), np.zeros_like(x))
        assert_array_equal(amplitude_grating(x, 2, 2), np.ones_like(x))
        assert_array_equal(amplitude_grating([-.5, .5], 2, 1), [0, 0])

    def test_invalid_dimensions(self):
        for period, width in [(0, 0), (-2, 1), (2, -1), (2, 3),
                              (np.nan, 1), (2, np.inf)]:
            with self.subTest(period=period, width=width), self.assertRaises(ValueError):
                amplitude_grating([0], period, width)
        with self.assertRaises(ValueError):
            amplitude_grating([np.nan], 2, 1)

    def test_periodic_mask_preserves_incident_phase_offset(self):
        x = np.array([-.2, .1, .8])
        period, kx = 2, np.pi / 2
        field = amplitude_grating(x, period, 1) * np.exp(1j * kx * x)
        shifted = (amplitude_grating(x + period, period, 1)
                   * np.exp(1j * kx * (x + period)))
        assert_allclose(shifted, field * np.exp(1j * kx * period), atol=1e-14)
        self.assertFalse(np.allclose(shifted, field))
        assert_allclose(abs(shifted), abs(field))


class GratingSurfaceTests(unittest.TestCase):
    def test_normal_incidence_and_period_changes(self):
        explorer = GratingSurfaceExplorer()
        try:
            assert_allclose(explorer.data['surface_phase'], 0)
            assert_allclose(explorer.data['incident'][-1], 1)
            assert_allclose(explorer.data['k_vector'], [0, 0, -np.pi])
            self.assertTrue(np.all(explorer.data['z'] >= 0))
            before = explorer.data['incident'].copy()
            explorer.period.value = 4
            assert_allclose(explorer.data['incident'], before)
            self.assertEqual(explorer.data['phase_step'], 0)
            self.assertIn('Normal incidence', explorer.readout.value)
        finally:
            explorer.close()

    def test_signed_tilt_wavelength_and_normal_button(self):
        explorer = GratingSurfaceExplorer()
        try:
            for angle in (30, -30):
                explorer.angle.value = angle
                expected_kx = np.sign(angle) * np.pi / 2
                self.assertAlmostEqual(explorer.data['k_vector'][0], expected_kx)
                self.assertAlmostEqual(explorer.data['k_vector'][2], -np.sqrt(3) * np.pi / 2)
                self.assertAlmostEqual(explorer.data['phase_step'], np.sign(angle) * np.pi)
                assert_allclose(explorer.data['surface_phase'], expected_kx * explorer.data['x'])
                assert_allclose(explorer.data['incident'][-1],
                                np.cos(explorer.data['surface_phase']), atol=1e-14)
                centres = np.isclose(explorer.data['x'] % 2, 0)
                assert_allclose(explorer.data['incident'][-1, centres], [1, -1, 1, -1, 1])
            explorer.wavelength.value = 4
            self.assertAlmostEqual(explorer.data['phase_step'], -np.pi / 2)
            explorer.normal.click()
            assert_allclose(explorer.data['surface_phase'], 0)
            self.assertEqual(explorer.angle.value, 0)
        finally:
            explorer.close()

    def test_replacements_release_resources_and_leave_another_explorer_live(self):
        from ipywidgets.widgets.widget import _instances
        from matplotlib._pylab_helpers import Gcf
        from wave_optics.widgets import PlaneWaveExplorer

        baseline = set(_instances)
        managers = set(Gcf.figs)
        other = PlaneWaveExplorer(dimensions=1)
        explorer = None
        try:
            other_id = other.ui.model_id
            other_count = len(_instances)
            for _ in range(3):
                explorer = GratingSurfaceExplorer()
                population = set(_instances)
                figure_ref = weakref.ref(explorer.figure)
                callbacks = {name: set(handlers) for name, handlers
                             in explorer.figure.canvas.callbacks.callbacks.items()}
                for angle in (0, 30, -30, 60):
                    explorer.angle.value = angle
                    self.assertEqual(len(explorer.figure.axes), 3)
                    self.assertEqual(set(_instances), population)
                    self.assertEqual(callbacks, {name: set(handlers) for name, handlers
                                     in explorer.figure.canvas.callbacks.callbacks.items()})
                    for control in explorer._controls:
                        self.assertEqual(control._trait_notifiers['value']['change'].count(
                            explorer._update), 1)
                old_control = explorer.angle
                explorer.close()
                explorer.close()
                self.assertNotIn(explorer._update, old_control._trait_notifiers['value']['change'])
                self.assertNotIn(explorer._normal_incidence, explorer.normal._click_handlers.callbacks)
                old_control.value = -45
                self.assertIsNone(explorer.figure)
                self.assertIsNone(explorer.ui.comm)
                self.assertIsNone(explorer.picture.comm)
                self.assertEqual(bytes(explorer.picture.value), b'')
                self.assertEqual(len(_instances), other_count)
                self.assertEqual(other.ui.model_id, other_id)
                previous = bytes(other.picture.value)
                other.time.value += 1
                self.assertNotEqual(bytes(other.picture.value), previous)
                gc.collect()
                self.assertIsNone(figure_ref())
                self.assertEqual(set(Gcf.figs), managers)
        finally:
            if explorer is not None:
                explorer.close()
            other.close()
        gc.collect()
        self.assertEqual(set(_instances), baseline)


class OrderSurfaceTests(unittest.TestCase):
    def test_intersections_lie_on_the_same_rays_and_requested_surfaces(self):
        vectors = np.array([[0, 0, -4], [3, 0, -4], [2, 3, -6]])
        data = order_surface_intersections(vectors, radius=2, distance=3)
        assert_allclose(data['plane'], [[0, 0, -3], [2.25, 0, -3], [1, 1.5, -3]])
        assert_allclose(np.linalg.norm(data['sphere'], axis=1), 2)
        assert_allclose(np.cross(data['sphere'], data['plane']), 0, atol=1e-14)
        scaled_vectors = order_surface_intersections(5 * vectors, 2, 3)
        assert_allclose(scaled_vectors['sphere'], data['sphere'])
        assert_allclose(scaled_vectors['plane'], data['plane'])
        farther = order_surface_intersections(vectors, 2, 6)
        assert_allclose(farther['plane'], 2 * data['plane'])
        assert_allclose(farther['sphere'], data['sphere'])

    def test_grazing_direction_has_no_plane_intersection(self):
        data = order_surface_intersections([[1, 0, 0], [0, 0, -1]], 2, 3)
        assert_array_equal(data['reaches_plane'], [False, True])
        assert_allclose(data['sphere'][0], [2, 0, 0])
        self.assertTrue(np.isnan(data['plane'][0]).all())
        assert_allclose(data['plane'][1], [0, 0, -3])

    def test_invalid_rays_and_surface_dimensions(self):
        for vectors in ([], [0, 0, -1], [[0, 0, 0]], [[0, 0, 1]],
                        [[0, np.nan, -1]], [[0, 0, -1j]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                order_surface_intersections(vectors, 2, 3)
        for radius, distance in ((0, 3), (2, -3), (np.inf, 3), (2, np.nan)):
            with self.subTest(radius=radius, distance=distance), self.assertRaises(ValueError):
                order_surface_intersections([[0, 0, -1]], radius, distance)

    def test_slider_moves_plane_without_changing_sphere_directions_or_scale(self):
        explorer = OrderSurfaceExplorer()
        try:
            initial = {key: value.copy() for key, value in explorer.data.items()}
            limits = [(axis.get_xlim(), axis.get_ylim()) for axis in explorer.figure.axes]
            explorer.distance.value = 6
            assert_allclose(explorer.data['plane'], 2 * initial['plane'])
            assert_allclose(explorer.data['sphere'], initial['sphere'])
            assert_allclose(explorer.data['directions'], initial['directions'])
            self.assertEqual(limits, [(axis.get_xlim(), axis.get_ylim())
                                     for axis in explorer.figure.axes])
            self.assertIn('2.619', explorer.readout.value)
            self.assertTrue(bytes(explorer.picture.value).startswith(b'\x89PNG'))
            self.assertEqual(len(explorer.figure.axes), 2)
            self.assertTrue(all(not hasattr(axis, 'zaxis') for axis in explorer.figure.axes))
        finally:
            explorer.close()

    def test_physical_controls_recompute_orders_and_signed_incidence(self):
        explorer = OrderSurfaceExplorer()
        try:
            for angle, expected in ((30, [-3, -2, -1, 0, 1]),
                                    (-30, [-1, 0, 1, 2, 3])):
                explorer.angle.value = angle
                assert_array_equal(explorer.orders, expected)
                assert_allclose(explorer.directions[:, 0],
                    np.sin(np.deg2rad(angle)) + explorer.orders / 2.5, atol=1e-12)
                assert_allclose(explorer.directions[:, 1], 0)
                self.assertTrue(np.all(explorer.directions[:, 2] < 0))
            explorer.reset.click()
            explorer.wavelength.value = 4
            assert_array_equal(explorer.orders, [0])
            assert_allclose(explorer.directions, [[0, 0, -1]])
            explorer.period.value = 4
            assert_array_equal(explorer.orders, [-1, 0, 1])
            assert_array_equal(explorer.data['reaches_plane'], [False, True, False])
        finally:
            explorer.close()

    def test_radius_changes_only_sphere_and_reset_restores_example(self):
        explorer = OrderSurfaceExplorer()
        try:
            sphere = explorer.data['sphere'].copy()
            plane = explorer.data['plane'].copy()
            directions = explorer.directions.copy()
            explorer.radius.value = 4
            assert_allclose(explorer.data['sphere'], 2 * sphere)
            assert_allclose(explorer.data['plane'], plane)
            assert_allclose(explorer.directions, directions)
            explorer.distance.value = 1  # plane now lies inside the sphere
            assert_allclose(explorer.data['plane'][:, 2], -1)
            explorer.wavelength.value = 2
            explorer.period.value = 3
            explorer.angle.value = -45
            explorer.reset.click()
            self.assertEqual(tuple(c.value for c in explorer._controls), (1, 2.5, 0, 2, 3))
            assert_allclose(explorer.data['sphere'], sphere)
            assert_allclose(explorer.data['plane'], plane)
        finally:
            explorer.close()

    def test_grazing_and_offscreen_orders_remain_distinct(self):
        explorer = OrderSurfaceExplorer()
        try:
            explorer.period.value = 2
            explorer.angle.value = 30  # exact grazing, including roundoff in sin(30)
            grazing = np.isin(explorer.orders, [-3, 1])
            assert_array_equal(explorer.data['reaches_plane'], ~grazing)
            self.assertTrue(np.isnan(explorer.data['plane'][grazing]).all())
            assert_allclose(explorer.directions[grazing, 2], 0)
            self.assertIn('Grazing orders', explorer.readout.value)
            explorer.reset.click()
            explorer.angle.value = 35
            explorer.distance.value = 6
            first = np.flatnonzero(explorer.orders == 1)[0]
            self.assertTrue(explorer.data['reaches_plane'][first])
            self.assertFalse(explorer.data['visible_plane'][first])
            self.assertGreater(explorer.data['plane'][first, 0], 9)
            self.assertIn('outside the shown x range', explorer.readout.value)
            shown = explorer.figure.axes[0].collections[-1].get_offsets()
            assert_allclose(shown, explorer.data['plane'][explorer.data['visible_plane']][:, [0, 2]])
        finally:
            explorer.close()

    def test_ruler_preserves_order_spacing_and_matches_the_spatial_x_scale(self):
        explorer = OrderSurfaceExplorer()
        try:
            explorer.period.value = 4
            explorer.radius.value = 4
            assert_array_equal(explorer.orders, np.arange(-4, 5))
            # The m=+4 grazing marker still has the same horizontal step.
            assert_allclose(explorer.data['sphere'][:, 0], np.arange(-4, 5), atol=1e-12)
            for wavelength, angle in ((1, 0), (1.2, 0), (1.2, -30)):
                explorer.wavelength.value = wavelength
                explorer.angle.value = angle
                spatial, ruler = explorer.figure.axes
                shown = np.asarray(ruler.collections[0].get_offsets())
                assert_allclose(shown[:, 0], explorer.data['sphere'][:, 0])
                assert_allclose(shown[:, 1], 0)
                assert_allclose(np.diff(shown[:, 0]), wavelength, atol=1e-12)
                alpha = explorer.directions[:, 0]
                assert_allclose(np.diff(alpha), wavelength / 4, atol=1e-12)
                assert_allclose(np.diff((2 * np.pi / wavelength) * alpha),
                                np.pi / 2, atol=1e-12)
                # A shared numeric x limit alone would not catch an aspect-
                # induced mismatch in the actual horizontal pixel positions.
                explorer.figure.canvas.draw()
                assert_allclose(spatial.transData.transform(shown)[:, 0],
                                ruler.transData.transform(shown)[:, 0])
            explorer.wavelength.value = 4
            explorer.period.value = .5
            self.assertIn('no neighbouring pair', explorer.readout.value)
            self.assertEqual(len(explorer.figure.axes[1].collections[0].get_offsets()), 1)
        finally:
            explorer.close()

    def test_component_triangles_follow_the_shared_controls_and_signed_direction(self):
        explorer = OrderSurfaceExplorer()
        try:
            self.assertEqual(explorer.component_order.value, 1)
            self.assertEqual(len(explorer.component_figure.axes), 3)
            limits = [(axis.get_xlim(), axis.get_ylim())
                      for axis in explorer.component_figure.axes]
            initial = {key: value.copy() for key, value in explorer.component_data.items()
                       if isinstance(value, np.ndarray)}
            explorer.radius.value = 4
            assert_allclose(explorer.component_data['sphere'], 2 * initial['sphere'])
            assert_allclose(explorer.component_data['k_vector'], initial['k_vector'])
            assert_allclose(explorer.component_data['spatial_frequency'],
                            initial['spatial_frequency'])
            image = bytes(explorer.component_picture.value)
            explorer.distance.value = 6
            self.assertEqual(bytes(explorer.component_picture.value), image)
            explorer.wavelength.value = 1.2
            # At normal incidence and fixed d, kx stays fixed as k changes.
            self.assertAlmostEqual(explorer.component_data['k_vector'][0], 2*np.pi/2.5)
            self.assertAlmostEqual(explorer.component_data['spatial_frequency'][0], 1/2.5)
            explorer.period.value = 4
            explorer.angle.value = -30
            self.assertEqual(explorer.component_order.value, 1)
            self.assertAlmostEqual(explorer.component_data['alpha'], -.2)
            self.assertIn('left', explorer.component_readout.value)
            for order in explorer.orders:
                explorer.component_order.value = int(order)
                index = int(np.flatnonzero(explorer.orders == order)[0])
                data = explorer.component_data
                assert_allclose(data['direction'], explorer.directions[index])
                assert_allclose(data['sphere'], explorer.data['sphere'][index])
                assert_allclose(data['k_vector'], (2*np.pi/1.2)*explorer.directions[index])
                assert_allclose(2*np.pi*data['spatial_frequency'], data['k_vector'])
                self.assertAlmostEqual(np.linalg.norm(data['spatial_frequency']), 1/1.2)
                self.assertAlmostEqual(data['alpha'], np.sin(np.deg2rad(data['theta_deg'])))
                for axis, endpoint in zip(explorer.component_figure.axes,
                                           (data['sphere'], data['k_vector'],
                                            data['spatial_frequency'])):
                    assert_allclose(axis.collections[0].get_offsets(), [endpoint[[0, 2]]])
                self.assertEqual(limits, [(axis.get_xlim(), axis.get_ylim())
                                         for axis in explorer.component_figure.axes])
        finally:
            explorer.close()

    def test_component_selection_handles_grazing_cutoff_zero_and_reset(self):
        explorer = OrderSurfaceExplorer()
        try:
            explorer.period.value = 4
            for order in (-4, 4):
                explorer.component_order.value = order
                self.assertTrue(explorer.component_data['grazing'])
                self.assertEqual(explorer.component_data['k_vector'][2], 0)
                self.assertEqual(explorer.component_data['theta_deg'], np.sign(order)*90)
                self.assertIn('Grazing', explorer.component_readout.value)
                assert_allclose(explorer.component_data['sphere'],
                                [np.sign(order)*explorer.radius.value, 0, 0])
                assert_allclose(explorer.component_data['spatial_frequency'],
                                [np.sign(order)/explorer.wavelength.value, 0, 0])
                self.assertTrue(bytes(explorer.component_picture.value).startswith(b'\x89PNG'))
            explorer.wavelength.value = 1.1
            self.assertEqual(explorer.component_order.value, 0)
            self.assertIn('m = +4 no longer', explorer.component_readout.value)
            self.assertIn('showing m = 0', explorer.component_readout.value)
            self.assertIn('horizontal component is zero', explorer.component_readout.value)
            self.assertFalse(explorer.component_data['grazing'])
            assert_allclose(explorer.component_data['direction'], [0, 0, -1])
            assert_allclose(explorer.component_data['spatial_frequency'],
                            [0, 0, -1/1.1])
            self.assertNotIn(4, [value for label, value in explorer.component_order.options])
            explorer.reset.click()
            self.assertEqual(explorer.component_order.value, 1)
            self.assertEqual(explorer.component_data['order'], 1)
            self.assertAlmostEqual(explorer.component_data['alpha'], .4)
            self.assertNotIn('no longer', explorer.component_readout.value)
        finally:
            explorer.close()

    def test_reruns_and_close_release_only_owned_resources(self):
        from ipywidgets.widgets.widget import _instances
        from matplotlib._pylab_helpers import Gcf
        baseline = set(_instances)
        managers = set(Gcf.figs)
        other = GratingSurfaceExplorer()
        population_with_other = set(_instances)
        try:
            for _ in range(2):
                explorer = OrderSurfaceExplorer()
                try:
                    population = set(_instances)
                    figure_ref = weakref.ref(explorer.figure)
                    component_ref = weakref.ref(explorer.component_figure)
                    for control, value in zip(explorer._controls, (2, 3, -30, 4, 6)):
                        control.value = value
                        self.assertEqual(set(_instances), population)
                        self.assertEqual(len(explorer.figure.axes), 2)
                        for observed in explorer._controls:
                            self.assertEqual(observed._trait_notifiers['value']['change'].count(
                                explorer._update), 1)
                    old_controls = explorer._controls
                    old_selector = explorer.component_order
                    self.assertEqual(old_selector._trait_notifiers['value']['change'].count(
                        explorer._update), 1)
                finally:
                    explorer.close()
                explorer.close()
                for control in old_controls:
                    self.assertNotIn(explorer._update, control._trait_notifiers['value']['change'])
                    control.value = control.min
                self.assertNotIn(explorer._reset_example, explorer.reset._click_handlers.callbacks)
                self.assertNotIn(explorer._update, old_selector._trait_notifiers['value']['change'])
                old_selector.value = 0
                self.assertIsNone(explorer.figure)
                self.assertIsNone(explorer.component_figure)
                self.assertIsNone(explorer.ui.comm)
                self.assertIsNone(explorer.components_ui.comm)
                self.assertIsNone(explorer.component_picture.comm)
                self.assertEqual(bytes(explorer.component_picture.value), b'')
                self.assertEqual(set(_instances), population_with_other)
                gc.collect()
                self.assertIsNone(figure_ref())
                self.assertIsNone(component_ref())
                self.assertEqual(set(Gcf.figs), managers)
            previous = bytes(other.picture.value)
            other.angle.value = 30
            self.assertNotEqual(bytes(other.picture.value), previous)
        finally:
            other.close()
        gc.collect()
        self.assertEqual(set(_instances), baseline)


if __name__ == '__main__':
    unittest.main()
