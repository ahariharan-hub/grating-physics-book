"""Physical equivalence and invalidation of the early-notebook fast updates."""
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from wave_optics import plotting
from wave_optics.coordinates import wavevector_coordinates
from wave_optics.diffraction import diffraction_orders


class RenderUpdateReuseTests(unittest.TestCase):
    def test_plane_wave_cached_phase_tracks_direction_and_wavelength(self):
        with patch.object(plotting, 'figure_png', return_value=b''):
            for dimensions in (1, 2):
                plot = plotting.PlaneWavePlot(dimensions)
                for wavelength, direction, fraction in ((2., 1 if dimensions == 1 else 30., 0.),
                        (2., 1 if dimensions == 1 else 30., .31),
                        (1.5, -1 if dimensions == 1 else -70., .31)):
                    plot.render(wavelength, direction, fraction)
                    if dimensions == 1:
                        expected = np.cos(plot.kx*plot.x-2*np.pi*fraction)
                    else:
                        expected = np.cos(plot.kx*plot.X+plot.kz*plot.Z-2*np.pi*fraction)
                    assert_allclose(plot.field, expected, atol=2e-14)
                plot.figure.clear()

    def test_3d_time_reuses_static_artists_and_camera_reuses_fronts(self):
        with patch.object(plotting, 'figure_png', return_value=b''):
            plot = plotting.ScalarPlaneWave3DPlot()
            plot.render(2., 45., 30., 0.)
            artists = (plot.sphere, plot.k_arrow, plot.direction_arrow)
            plot.render(2., 45., 30., .25)
            self.assertEqual(artists, (plot.sphere, plot.k_arrow, plot.direction_arrow))
            fronts = plot.fronts
            plot.render(2., 45., 30., .25, elevation=55)
            self.assertIs(plot.fronts, fronts)
            for value, _, vertices in fronts:
                assert_allclose(plot.field_at(vertices), value, atol=1e-9)
            plot.render(1.5, 70., -45., .25)
            self.assertIsNot(plot.sphere, artists[0])
            for value, _, vertices in plot.fronts:
                assert_allclose(plot.field_at(vertices), value, atol=1e-9)
            plot.figure.clear()

    def test_coordinate_updates_match_fresh_view_and_keep_camera(self):
        initial = wavevector_coordinates(.55, 1., 40., 30., .8)
        figure = plotting.coordinate_figure(initial)
        figure.layout.scene.camera.eye = dict(x=2., y=-1., z=.5)
        camera = figure.layout.scene.camera.to_plotly_json()
        previous = initial
        for projection, args in (
            ('pupil', (.55, 1., 40., 30., .8)),
            ('frequency', (.6, 1.5, 120., -30., .9)),
            ('cartesian', (.6, 1.5, 0., 120., 1.5)),
            ('direction', (.4, 1., 90., 0., .5))):
            data = wavevector_coordinates(*args)
            plotting.update_coordinate_figure(figure, data, projection, previous)
            fresh = plotting.coordinate_figure(data, projection)
            for current, expected in zip(figure.data, fresh.data):
                for key in ('x', 'y', 'z', 'u', 'v', 'w'):
                    if hasattr(expected, key):
                        assert_allclose(np.asarray(getattr(current, key), dtype=float),
                                        np.asarray(getattr(expected, key), dtype=float),
                                        atol=1e-14, equal_nan=True)
                if hasattr(expected, 'marker'):
                    self.assertEqual(current.marker.to_plotly_json(), expected.marker.to_plotly_json())
                self.assertEqual(current.name, expected.name)
            for axis in ('xaxis', 'yaxis'):
                self.assertEqual(figure.layout[axis].to_plotly_json(), fresh.layout[axis].to_plotly_json())
            self.assertEqual(figure.layout.annotations, fresh.layout.annotations)
            self.assertEqual(figure.layout.scene.camera.to_plotly_json(), camera)
            previous = data

    def test_requested_geometry_has_same_traces_without_other_builder(self):
        data = diffraction_orders(2., 4., 25., 30.)
        both = plotting.geometry_figures(data)
        for name, expected in zip(('direction', 'ewald'), both):
            with patch.object(plotting, 'geometry_base', wraps=plotting.geometry_base) as build:
                single, = plotting.geometry_figures(data, views=(name,))
            self.assertEqual(build.call_count, 1)
            self.assertEqual(single.to_json(), expected.to_json())

    def test_direction_change_preserves_coordinate_zoom_until_scale_changes(self):
        from wave_optics.widgets import WavevectorCoordinateExplorer
        explorer = WavevectorCoordinateExplorer()
        try:
            explorer.figure.layout.xaxis.range = [-.4, .4]
            explorer.figure.layout.yaxis.range = [-.3, .3]
            explorer.theta.value = 50.
            self.assertEqual(explorer.figure.layout.xaxis.range, (-.4, .4))
            self.assertEqual(explorer.figure.layout.yaxis.range, (-.3, .3))
            explorer.projection.value = 'frequency'
            self.assertNotEqual(explorer.figure.layout.xaxis.range, (-.4, .4))
        finally:
            explorer.close()

    def test_evanescent_playback_reuses_axes_and_matches_new_snapshot(self):
        from wave_optics import evanescent_view as view
        with patch.object(view, 'figure_png', return_value=b''):
            explorer = view.EvanescentOrderExplorer()
            try:
                for period in (3., 2., 1.5):
                    explorer.angle.value = 0.
                    explorer.period.value = period
                    axes = tuple(explorer.figure.axes)
                    with patch.object(view, 'order_phasor', wraps=view.order_phasor) as field:
                        explorer.time.value = .375 if explorer.time.value != .375 else .625
                    self.assertEqual(field.call_count, 0)
                    self.assertEqual(tuple(explorer.figure.axes), axes)
                    x = np.linspace(-explorer.depth.value, explorer.depth.value, 481)
                    z = np.linspace(-explorer.depth.value, 0, 301)
                    expected = (view.order_phasor(x[None, :], z[:, None], explorer.data)
                                *np.exp(-2j*np.pi*explorer.time.value)).real
                    assert_allclose(explorer._artists['picture'].get_array(), expected)
                explorer.depth.value = .25
                self.assertIsNot(explorer.figure.axes[0], axes[0])
            finally:
                explorer.close()
            self.assertIsNone(explorer._artists)


if __name__ == '__main__':
    unittest.main()
