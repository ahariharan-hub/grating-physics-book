"""Retained notebook-seven displays preserve fields and invalidate only dependencies."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

from wave_optics import interference_setup as setup
from wave_optics import interference_wavefronts as waves
from wave_optics import phase_exposure, phase_geometry


def roles(figure):
    return {trace.meta['role']: trace for trace in figure.data
            if isinstance(trace.meta, dict) and 'role' in trace.meta}


class InterferenceUpdateTests(unittest.TestCase):
    def test_cached_eo_phase_matches_independent_field_calculation(self):
        for tilt, z in ((0, -6.9), (2, -9.15), (-2, -3.5), (0, -10)):
            geometry = setup.two_beam_geometry(tilt, z)
            for power in (0., 1.):
                base = waves.wavefront_data(geometry, power_mw=power, optics_phase_deg=37)
                original = base['fields']['field_2'].copy()
                for beta in (0., 90., 180.):
                    for varies in (False, True):
                        actual = waves.phase_shifted_data(base, beta, phase_varies=varies)
                        expected = waves.wavefront_data(geometry, power_mw=power,
                            optics_phase_deg=37, eo_phase_deg=beta, phase_varies=varies)
                        for sample in ('fields', 'profile', 'point'):
                            for key in ('field_1', 'field_2', 'intensity', 'recorded_intensity'):
                                assert_allclose(actual[sample][key], expected[sample][key], atol=5e-11)
                            self.assertIs(actual[sample]['intensity_1'], base[sample]['intensity_1'])
                            self.assertIs(actual[sample]['intensity_2'], base[sample]['intensity_2'])
                        self.assertEqual(actual['coherence_magnitude'], expected['coherence_magnitude'])
                        self.assertIs(actual['delta_k_per_um'], base['delta_k_per_um'])
                        assert_array_equal(base['fields']['field_2'], original)

    def test_retained_plotly_traces_match_fresh_figures(self):
        data = waves.wavefront_data(setup.two_beam_geometry())
        figure, _ = waves.wavefront_figure(data=data)
        identities = tuple(map(id, figure.data))
        camera = dict(eye=dict(x=1.2, y=-1.4, z=2.1))
        figure.layout.scene.camera = camera
        figure.layout.xaxis.range = (-2, 2)
        for tilt, z, change in ((2, -9.15, 'geometry'), (2, -3.5, 'position'),
                                (-2, -10, 'geometry')):
            base = waves.wavefront_data(setup.two_beam_geometry(tilt, z))
            for beta in (0, 90, 180):
                data = waves.phase_shifted_data(base, beta)
                waves.update_wavefront_figure(figure, data, change=change if beta == 0 else 'phase')
                for mode in ('field_1', 'field_2', 'intensity'):
                    waves.update_wavefront_figure(figure, data, mode=mode, change='mode')
                    expected, _ = waves.wavefront_figure(data=data.copy(), mode=mode)
                    for name, trace in roles(expected).items():
                        actual = roles(figure)[name]
                        for key in ('x', 'y', 'z', 'surfacecolor', 'i', 'j', 'k', 'u', 'v', 'w'):
                            value = getattr(trace, key, None)
                            if value is not None:
                                assert_allclose(np.asarray(getattr(actual, key), dtype=float),
                                                np.asarray(value, dtype=float), atol=5e-11)
                        self.assertEqual(actual.visible, trace.visible)
                    self.assertEqual([a.to_plotly_json() for a in figure.layout.annotations],
                                     [a.to_plotly_json() for a in expected.layout.annotations])
                    self.assertEqual(roles(figure)['detector-map'].zmin,
                                     roles(expected)['detector-map'].zmin)
                    self.assertEqual(roles(figure)['detector-map'].colorbar.title.text,
                                     roles(expected)['detector-map'].colorbar.title.text)
                    self.assertEqual(tuple(map(id, figure.data)), identities)
                    self.assertEqual(figure.layout.scene.camera.to_plotly_json(), camera)
                    self.assertEqual(tuple(figure.layout.xaxis.range), (-2, 2))

    def test_retained_sketch_matches_fresh_drawing_and_clears_old_labels(self):
        for modulated in (False, True):
            sketch = setup._SetupSketch(phase_step_deg=0 if modulated else None)
            identities = tuple(map(id, sketch.axis.get_children()))
            try:
                for tilt, z in ((0, -6.9), (2, -9.15), (-2, -3.5), (0, -10)):
                    sketch.set_geometry(setup.two_beam_geometry(tilt, z), tilt)
                    for beta, varies in ((180, True), (0, False)):
                        if modulated:
                            sketch.set_phase(beta, varies)
                        sketch.png()
                        fresh, _ = setup.draw_two_plane_wave_setup(mirror_tilt_deg=tilt,
                            detector_z=z, phase_step_deg=beta if modulated else None,
                            phase_varies=varies)
                        try:
                            fresh.set_dpi(110)
                            canvas = FigureCanvasAgg(fresh)
                            canvas.draw()
                            assert_array_equal(np.asarray(sketch.canvas.buffer_rgba()),
                                               np.asarray(canvas.buffer_rgba()))
                            self.assertEqual(tuple(map(id, sketch.axis.get_children())), identities)
                        finally:
                            plt.close(fresh)
            finally:
                sketch.close()

    def test_explorer_reuses_fields_and_defers_closed_views(self):
        explorer = setup.MirrorTiltExplorer()
        try:
            explorer.detector_ui
            explorer.modulated_ui
            explorer.detector_modulated_ui
            explorer.phase_geometry_ui
            self.assertIsNone(explorer.phase_geometry_data)
            self.assertEqual(explorer.phase_geometry_picture.value, b'')
            with patch.object(waves, 'wavefront_data', wraps=waves.wavefront_data) as calculate, \
                 patch.object(waves, 'wavefront_figure', side_effect=AssertionError('figure rebuilt')), \
                 patch.object(setup, 'draw_two_plane_wave_setup', side_effect=AssertionError('sketch rebuilt')), \
                 patch.object(phase_geometry, 'draw_phase_geometry', wraps=phase_geometry.draw_phase_geometry) as projection:
                source = explorer._field_cache
                for mode in ('field_1', 'field_2', 'intensity'):
                    explorer.wavefront_mode.value = mode
                    explorer.modulated_mode.value = mode
                explorer.eo_phase.value = 90
                self.assertEqual(calculate.call_count, 0)
                self.assertIs(explorer._field_cache, source)
                explorer.tilt.value = 2
                self.assertEqual(calculate.call_count, 1)
                self.assertEqual(projection.call_count, 0)
                self.assertIs(explorer.wavefront_data['fields'], explorer._field_cache['fields'])
                explorer.phase_geometry_ui.selected_index = 0
                self.assertEqual(projection.call_count, 1)
                explorer.phase_geometry_ui.selected_index = None
                explorer.detector_z.value = -9.15
                self.assertEqual(calculate.call_count, 2)
                self.assertEqual(projection.call_count, 1)
                explorer.phase_geometry_ui.selected_index = 0
                self.assertEqual(projection.call_count, 2)
                assert_array_equal(explorer.phase_geometry_data['point'], [0, 0, -9.15])
            explorer.exposure_options.selected_index = 0
            view = explorer._switching_view
            frames = tuple(map(id, view._frames))
            with patch.object(waves, 'wavefront_data', wraps=waves.wavefront_data) as calculate, \
                 patch.object(phase_exposure, 'phase_switching_figures', side_effect=AssertionError('exposure rebuilt')):
                explorer.eo_phase.value = 180
                self.assertEqual(calculate.call_count, 0)
                self.assertEqual(tuple(map(id, view._frames)), frames)
                assert_allclose(view.data['average'], explorer._field_cache['fields']['intensity_1']
                                + explorer._field_cache['fields']['intensity_2'], atol=5e-11)
                view.play.value = 1
                assert_allclose(roles(view.figure)['detector-surface'].surfacecolor,
                                view.data['second'], atol=5e-11)
                view.play.playing = True
                explorer.exposure_options.selected_index = None
                self.assertFalse(view.play.playing)
                old_average = view.data['average']
                explorer.eo_phase.value = 45
                self.assertIs(view.data['average'], old_average)
                explorer.exposure_options.selected_index = 0
                self.assertIsNot(view.data['average'], old_average)
                self.assertEqual(view.state.value, 0)
        finally:
            explorer.close()
            explorer.close()
        self.assertIsNone(explorer._field_cache)
        self.assertIsNone(explorer._eo_cache)
        self.assertIsNone(explorer._switching_view._frames)
        self.assertIsNone(explorer._switching_view.play.comm)


if __name__ == '__main__':
    unittest.main()
