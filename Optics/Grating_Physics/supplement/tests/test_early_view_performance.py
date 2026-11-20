"""Dependency routing and bounded cache invalidation in notebooks two to four."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from wave_optics import grating_intro, order_array, order_formation, order_lesson, order_waves
from wave_optics.waves import scalar_plane_wave


class EarlyViewPerformanceTests(unittest.TestCase):
    def test_uniform_images_preserve_sample_centres_direction_and_masks(self):
        from matplotlib.figure import Figure

        for z in (np.linspace(-4, 0, 7), np.linspace(8, 0, 7)):
            x = np.linspace(-2, 2, 9)
            values = np.cos(x[None, :] + z[:, None])
            values[2, 3] = np.nan
            figure = Figure()
            try:
                axis = figure.subplots()
                axis.set_aspect('auto')
                image = grating_intro._uniform_field_image(
                    axis, x, z, values, cmap='RdBu_r', vmin=-1, vmax=1)
                left, right, bottom, top = image.get_extent()
                assert_allclose(left + (np.arange(len(x))+.5)*(right-left)/len(x), x)
                assert_allclose(bottom + (np.arange(len(z))+.5)*(top-bottom)/len(z), z)
                assert_allclose(image.get_array().filled(np.nan), values, equal_nan=True)
                self.assertTrue(image.get_array().mask[2, 3])
                self.assertEqual(image.get_clim(), (-1, 1))
                self.assertEqual(image.get_interpolation(), 'nearest')
                self.assertEqual(axis.get_aspect(), 'auto')
            finally:
                figure.clear()

    def test_surface_and_component_views_render_only_for_their_inputs(self):
        with patch.object(grating_intro, 'figure_png', return_value=b'png') as render:
            explorer = grating_intro.OrderSurfaceExplorer()
            try:
                component_axes = tuple(explorer.component_figure.axes)
                render.reset_mock()
                explorer.distance.value = 5
                self.assertEqual(render.call_count, 1)
                self.assertIs(render.call_args.args[0], explorer.figure)
                self.assertEqual(tuple(explorer.component_figure.axes), component_axes)
                assert_allclose(explorer.data['plane'][explorer.data['reaches_plane'], 2], -5)

                spatial_axes = tuple(explorer.figure.axes)
                render.reset_mock()
                explorer.component_order.value = -1
                self.assertEqual(render.call_count, 1)
                self.assertIs(render.call_args.args[0], explorer.component_figure)
                self.assertEqual(tuple(explorer.figure.axes), spatial_axes)
                self.assertEqual(explorer.component_data['order'], -1)
                self.assertIn('Selected order m = -1', explorer.readout.value)

                render.reset_mock()
                explorer.wavelength.value = 1.2
                self.assertEqual(render.call_count, 2)
            finally:
                explorer.close()

    def test_incident_grating_field_survives_period_change_but_not_tilt(self):
        with patch.object(grating_intro, 'figure_png', return_value=b'png'):
            explorer = grating_intro.GratingSurfaceExplorer()
            try:
                field, x = explorer.data['incident'], explorer.data['x']
                explorer.period.value = 3
                self.assertIs(explorer.data['incident'], field)
                self.assertIs(explorer.data['x'], x)
                explorer.angle.value = -30
                self.assertIsNot(explorer.data['incident'], field)
                xx, zz = np.meshgrid(explorer.data['x'], explorer.data['z'])
                points = np.stack((xx, np.zeros_like(xx), zz), axis=-1)
                assert_allclose(explorer.data['incident'],
                                scalar_plane_wave(points, explorer.data['k_vector']), atol=1e-14)
            finally:
                explorer.close()
            self.assertIsNone(explorer._incident_cache)

    def test_selected_order_reuses_incident_field_and_invalidates_outgoing(self):
        cache = {}
        first = order_waves.order_wave_data(_field_cache=cache)
        second = order_waves.order_wave_data(selected_order=0, _field_cache=cache)
        self.assertIs(first['incident_field'], second['incident_field'])
        self.assertIs(first['x'], second['x'])
        self.assertIsNot(first['outgoing_field'], second['outgoing_field'])
        changed_period = order_waves.order_wave_data(period=3, selected_order=0,
                                                      _field_cache=cache)
        self.assertIs(second['incident_field'], changed_period['incident_field'])
        self.assertIs(second['outgoing_field'], changed_period['outgoing_field'])
        tilted = order_waves.order_wave_data(angle_deg=-30, _field_cache=cache)
        self.assertIsNot(tilted['incident_field'], second['incident_field'])
        xx, zz = np.meshgrid(tilted['x'], tilted['z_below'])
        points = np.stack((xx, np.zeros_like(xx), zz), axis=-1)
        assert_allclose(tilted['outgoing_field'],
                        scalar_plane_wave(points, tilted['vectors'][tilted['index']]), atol=1e-14)
        self.assertEqual(set(cache), {'grid', 'incident', 'outgoing'})

    def test_linked_paths_and_comparison_skip_repeated_unchanged_requests(self):
        with patch.object(order_waves, 'figure_png', return_value=b'png'):
            explorer = order_waves.OrderWaveExplorer()
            try:
                explorer.paths_ui
                explorer.compare.value = True
                with patch.object(order_waves, 'cell_comparison_data',
                                  wraps=order_waves.cell_comparison_data) as calculate:
                    with patch.object(order_waves, 'draw_path_connection',
                                      wraps=order_waves.draw_path_connection) as draw:
                        explorer._update_comparison()
                        explorer.use_order.click()
                        calculate.assert_not_called()
                        draw.assert_not_called()
                        explorer.comparison.value = -.5
                        self.assertEqual(calculate.call_count, 1)
                        self.assertEqual(draw.call_count, 1)
                        self.assertIs(explorer.path_data['wave'], explorer.comparison_data)
                        self.assertFalse(explorer.comparison_data['matched'])
            finally:
                explorer.close()
            self.assertEqual(explorer._field_cache, {})

    def test_cursor_reuses_scan_and_opening_count_reuses_wavefronts(self):
        with patch('wave_optics.plotting.figure_png', return_value=b'png'):
            explorer = order_formation.OrderFormationExplorer(view='wavefronts')
            try:
                with patch.object(order_formation, 'array_intensity',
                                  wraps=order_formation.array_intensity) as scan:
                    explorer.angle.value = 13.37
                    scan.assert_not_called()
                    angle_index = np.flatnonzero(explorer.data['scan'] == 13.37)
                    self.assertEqual(len(angle_index), 1)
                    self.assertAlmostEqual(explorer.data['intensity'][angle_index[0]],
                                           explorer.data['current'])
                    field = explorer.data['continuous']
                    with patch.object(order_formation, 'wavefront_fields',
                                      wraps=order_formation.wavefront_fields) as waves:
                        explorer.periods.value = 8
                        self.assertEqual(scan.call_count, 1)
                        waves.assert_not_called()
                        self.assertIs(explorer.data['continuous'], field)
                        explorer.wavelength.value = 1.2
                        self.assertEqual(scan.call_count, 2)
                        self.assertEqual(waves.call_count, 1)
                    expected = order_formation.array_intensity(
                        1.2, explorer.period.value, explorer.incidence.value,
                        explorer.data['scan'], 8)
                    assert_allclose(explorer.data['intensity'], expected, atol=1e-14)
            finally:
                explorer.close()
            self.assertIsNone(explorer._scan_cache)
            self.assertIsNone(explorer._wavefront_cache)

    def test_opening_count_reuses_geometry_without_retaining_source_fields(self):
        cache = {}
        first = order_array.opening_array_data(4, 1, 0, 1, _geometry_cache=cache)
        second = order_array.opening_array_data(4, 1, 0, 2, _geometry_cache=cache)
        self.assertIs(first['x'], second['x'])
        self.assertIs(first['angles'], second['angles'])
        self.assertIsNot(first['field'], second['field'])
        reference = order_array.opening_array_data(4, 1, 0, 2)
        assert_allclose(second['field'], reference['field'], equal_nan=True)
        assert_allclose(second['intensity'], reference['intensity'])
        tilted = order_array.opening_array_data(4, 1, 30, 2, _geometry_cache=cache)
        self.assertIs(tilted['x'], second['x'])
        self.assertIsNot(tilted['angles'], second['angles'])
        self.assertEqual(set(cache), {'grid', 'orders'})

    def test_phase_lessons_retain_owned_figure_and_canvas(self):
        with patch.object(order_lesson, 'figure_png', return_value=b'png') as render:
            for cls, control, value in (
                    (order_lesson.SurfacePhaseExplorer, 'angle', -30),
                    (order_lesson.PhaseMatchExplorer, 'outgoing', 0)):
                explorer = cls()
                try:
                    figure, canvas = explorer.figure, explorer.figure.canvas
                    render.reset_mock()
                    getattr(explorer, control).value = value
                    self.assertIs(explorer.figure, figure)
                    self.assertIs(explorer.figure.canvas, canvas)
                    self.assertEqual(len(explorer.figure.axes), 2)
                    self.assertEqual(render.call_count, 1)
                finally:
                    explorer.close()


if __name__ == '__main__':
    unittest.main()
