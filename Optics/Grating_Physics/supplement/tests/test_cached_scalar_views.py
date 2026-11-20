"""Dependency invalidation, retained artists and physics of cached view updates.

PNG serialization is mocked here: these checks inspect real Matplotlib artists
and computed arrays. Existing explorer tests still exercise real PNG rendering.
"""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from wave_optics._view_cache import ViewCache
from wave_optics import amplitude_view, scalar_theory_view, scalar_propagation_view, superposition_view
from wave_optics import helmholtz_families, scalar_propagation, superposition


class CachedScalarViewTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.renderers = {}
        for module in (amplitude_view, scalar_theory_view, scalar_propagation_view, superposition_view):
            self.renderers[module] = self.stack.enter_context(patch.object(module, 'figure_png', return_value=b''))

    def explorer(self, constructor, *args):
        explorer = constructor(*args)
        self.addCleanup(explorer.close)
        return explorer

    def test_cache_replaces_old_slider_states_and_releases_arrays(self):
        cache = ViewCache()
        for index in range(30):
            value = cache.get('field', (index,), lambda: np.ones(10)*index)
            self.assertIs(cache.get('field', (index,), lambda: self.fail('Unchanged dependency rebuilt')), value)
        self.assertEqual(len(cache), 1)
        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_special_function_field_is_reused_by_time_phase_and_amplitude(self):
        view = self.explorer(scalar_theory_view.ScalarTheoryExplorer, 'solutions')
        original = helmholtz_families.helmholtz_family
        with patch.object(helmholtz_families, 'helmholtz_family', wraps=original) as evaluate:
            view.solution.value = 'cylindrical'
            self.assertEqual(evaluate.call_count, 2)  # map and profile
            view.phase.value = 45
            view.amplitude.value = .5
            self.assertEqual(evaluate.call_count, 2)
            field, profile = view.data['field'], view.data['profile']
            axes = tuple(view.figure.axes)
            view.time.value = .25
            self.assertEqual(evaluate.call_count, 2)
            self.assertIs(view.data['field'], field)
            self.assertIs(view.data['profile'], profile)
            self.assertEqual(tuple(view.figure.axes), axes)
            assert_allclose(view.data['signed_field'], np.real(field*np.exp(-.5j*np.pi)), equal_nan=True)
            expected = original(view.data['points'], 2., 'cylindrical', .5, 45.)
            assert_allclose(field, expected, equal_nan=True)
            view.wavelength.value = 2.5
            self.assertEqual(evaluate.call_count, 4)
        view.close()
        self.assertEqual(len(view._cache), 0)

    def test_boundary_time_moves_only_cursor_and_current_field(self):
        view = self.explorer(scalar_theory_view.ScalarTheoryExplorer, 'boundary')
        field, history = view.data['field'], view.data['history']
        axes = tuple(view.figure.axes)
        with patch.object(scalar_theory_view, 'boundary_family', wraps=scalar_theory_view.boundary_family) as evaluate:
            view.time.value = .25
            self.assertEqual(evaluate.call_count, 0)
            self.assertIs(view.data['history'], history)
            self.assertEqual(tuple(view.figure.axes), axes)
            assert_allclose(view._time_cursor.get_ydata(), [.25, .25])
            assert_allclose(view.data['real_field'], np.real(field*np.exp(-.5j*np.pi)))
            view.fraction.value = .5
            self.assertEqual(evaluate.call_count, 1)

    def test_gaussian_plane_reuses_longitudinal_field_and_unit_amplitudes(self):
        view = self.explorer(scalar_theory_view.ScalarTheoryExplorer, 'solutions')
        view.solution.value = 'gaussian'
        view.view.value = 'beam'
        longitudinal = view.data['longitudinal_intensity']
        original = helmholtz_families.gaussian_beam
        with patch.object(helmholtz_families, 'gaussian_beam', wraps=original) as evaluate:
            view.axial.value = 1.
            self.assertEqual(evaluate.call_count, 2)  # only new transverse plane and profile
            self.assertIs(view.data['longitudinal_intensity'], longitudinal)
            before = view.data['transverse_intensity'].copy()
            view.amplitude.value = .5
            view.phase.value = 45.
            self.assertEqual(evaluate.call_count, 2)
            assert_allclose(view.data['transverse_intensity'], before*.25)
            expected = original(view.data['transverse_points'], 2., .5, 45., 2.)
            assert_allclose(view.data['transverse_field'], expected)

    def test_bessel_plane_keeps_intensity_but_advances_complex_phase(self):
        view = self.explorer(scalar_theory_view.ScalarTheoryExplorer, 'solutions')
        view.solution.value = 'bessel'
        view.view.value = 'beam'
        intensity = view.data['transverse_intensity']
        original = helmholtz_families.helmholtz_family
        with patch.object(helmholtz_families, 'helmholtz_family', wraps=original) as evaluate:
            view.beam_axial.value = 2.
            self.assertEqual(evaluate.call_count, 0)
            self.assertIs(view.data['transverse_intensity'], intensity)
            expected = original(view.data['transverse_points'], 2., 'bessel', transverse_ratio=.6)
            assert_allclose(view.data['transverse_field'], expected, atol=1e-14)
            view.transverse.value = .7
            self.assertEqual(evaluate.call_count, 3)

    def test_normalized_propagation_wavelength_updates_skip_render_and_rescale_units(self):
        for stage in ('geometry', 'transfer'):
            with self.subTest(stage=stage):
                view = self.explorer(scalar_propagation_view.ScalarPropagationExplorer, stage)
                data_key = 'field' if stage == 'geometry' else 'asm'
                unchanged = view.data[data_key]
                axes = tuple(view.figure.axes)
                renderer = self.renderers[scalar_propagation_view]
                renderer.reset_mock()
                view.wavelength.value = .8
                renderer.assert_not_called()
                self.assertIs(view.data[data_key], unchanged)
                self.assertEqual(tuple(view.figure.axes), axes)
                if stage == 'geometry':
                    expected = scalar_propagation.spectrum_geometry(.3/.8, .4/.8, .8)
                else:
                    expected = scalar_propagation.spectrum_geometry(view.data['s']/.8, 0., .8)
                    self.assertAlmostEqual(view.data['distance'], 1.6)
                    assert_allclose(view.data['asm'], scalar_propagation.transfer_function(view.data['s']/.8, 0., .8, 1.6))
                for key in ('fx', 'fy', 'k', 'kx', 'ky', 'kz', 'q', 'gamma', 'theta_deg'):
                    assert_allclose(view.data['geometry'][key], expected[key], equal_nan=True, atol=1e-13)

    def test_amplitude_order_reuses_plane_and_time_artists(self):
        view = self.explorer(amplitude_view.AmplitudeIntensityExplorer, 'order')
        with patch.object(amplitude_view, 'plane_wave_phasor', wraps=amplitude_view.plane_wave_phasor) as evaluate:
            view.amplitude.value = .5
            view.phase.value = 90.
            # cycle_samples calls the scalar helper once per amplitude/phase;
            # the large spatial plane remains cached across both changes.
            self.assertEqual(evaluate.call_count, 2)
            axes = tuple(view.figure.axes)
            evaluate.reset_mock()
            view.time.value = .25
            self.assertEqual(evaluate.call_count, 0)
            self.assertEqual(tuple(view.figure.axes), axes)
            assert_allclose(np.asarray(view._order_mesh.get_array()),
                            np.real(view._order_phasor*np.exp(-.5j*np.pi)))

    def test_two_wave_addition_time_keeps_cycles_and_artists(self):
        view = self.explorer(superposition_view.TwoWaveExplorer, 'addition')
        cycles, axes = view.data['fields'], tuple(view.figure.axes)
        with patch.object(superposition_view, 'two_wave_cycle', wraps=superposition_view.two_wave_cycle) as evaluate:
            view.time.value = .25
            evaluate.assert_not_called()
            self.assertIs(view.data['fields'], cycles)
            self.assertEqual(tuple(view.figure.axes), axes)
            for cursor in view._time_panel_artists[0]:
                assert_allclose(cursor.get_xdata(), [.25, .25])
            view.phase2.value = 180.
            self.assertEqual(evaluate.call_count, 1)
            self.assertAlmostEqual(view.data['relative_intensity'], 0.)

    def test_local_phase_time_retains_scene_and_advances_crest_planes(self):
        view = self.explorer(superposition_view.TwoWaveExplorer, 'local_phase')
        reference = self.explorer(superposition_view.TwoWaveExplorer, 'local_phase')
        vectors = superposition.symmetric_wavevectors(2, 30)

        def configure(explorer, amplitudes):
            explorer._resetting = True
            try:
                explorer.amplitude1.value, explorer.amplitude2.value = amplitudes
                explorer.phase1.value, explorer.phase2.value = 75., 210.
                explorer.detector_x.value = .75
                explorer.time.value = 0.
            finally:
                explorer._resetting = False
            explorer._update()

        def crest_positions(fronts, vector, offset, tau):
            result = {}
            for front in fronts:
                phase = front@vector+offset-2*np.pi*tau
                order = round(phase[0]/(2*np.pi))
                assert_allclose(phase, 2*np.pi*order, atol=1e-13)
                result[order] = np.mean(front@vector)/np.linalg.norm(vector)
            return result

        for amplitudes in ((1., .6), (0., .8), (.7, 0.), (0., 0.)):
            with self.subTest(amplitudes=amplitudes):
                configure(view, amplitudes)
                configure(reference, amplitudes)
                axes = tuple(view.figure.axes)
                children = tuple(tuple(axis.get_children()) for axis in axes)
                handles = view._local_time_artists
                cycles = view.local_cycle_data['fields']
                initial_fronts = view.wavefronts
                offsets = np.deg2rad([75., 210.])
                for tau in (.025, .25, .625, 1.):
                    with patch.object(superposition_view, 'two_wave_field',
                                      wraps=superposition_view.two_wave_field) as field:
                        with patch.object(superposition_view, 'two_wave_cycle',
                                          wraps=superposition_view.two_wave_cycle) as cycle:
                            view.time.value = tau
                            field.assert_not_called()
                            cycle.assert_not_called()
                    self.assertIs(view._local_time_artists, handles)
                    self.assertEqual(tuple(view.figure.axes), axes)
                    self.assertEqual(tuple(tuple(axis.get_children()) for axis in axes), children)
                    self.assertIs(view.local_cycle_data['fields'], cycles)

                    # A forced full redraw at the same settings is the visual
                    # state oracle; direct field and crest equations also
                    # verify the shared calculation independently.
                    reference._resetting = True
                    reference.time.value = tau
                    reference._resetting = False
                    reference._update()
                    self.assertEqual(view.readout.value, reference.readout.value)
                    point = [.75, 0., -1.]
                    expected = superposition.two_wave_field(point, vectors, *amplitudes,
                        135., tau=tau, phase1_deg=75.)
                    assert_allclose(view.local_readings, expected['fields'], atol=1e-13)
                    assert_allclose(view.local_data['relative_intensity'], expected['relative_intensity'])
                    values = [*expected['fields'], expected['field_sum']]
                    for bar, label, rebuilt_label, value in zip(handles['bars'], handles['bar_labels'],
                            reference._local_time_artists['bar_labels'], values):
                        self.assertAlmostEqual(bar.get_height(), value)
                        self.assertEqual(label.get_text(), rebuilt_label.get_text())
                        self.assertAlmostEqual(label.get_position()[1], value+(.12 if value >= 0 else -.12))
                        self.assertEqual(label.get_va(), 'bottom' if value >= 0 else 'top')
                    assert_allclose(handles['cursor'].get_xdata(), [tau, tau])
                    assert_allclose(handles['cursor_tip'].get_xdata(), [tau])
                    for marker, value in zip(handles['markers'], expected['fields']):
                        assert_allclose(marker.get_data(), [[tau], [value]])
                    self.assertEqual(handles['cycle_label'].get_text(),
                                     reference._local_time_artists['cycle_label'].get_text())
                    for index, (amp, vector, offset) in enumerate(zip(amplitudes, vectors, offsets)):
                        fronts = view.wavefronts[index]
                        drawn = [line for line in handles['front_lines'][index] if line.get_visible()]
                        self.assertEqual(len(drawn), len(fronts))
                        self.assertEqual(len(fronts), len(reference.wavefronts[index]))
                        for front, line, rebuilt in zip(fronts, drawn, reference.wavefronts[index]):
                            assert_allclose(front, rebuilt)
                            assert_allclose(line.get_xdata(), front[:, 0])
                            assert_allclose(line.get_ydata(), front[:, 2])
                        if amp == 0:
                            self.assertEqual(fronts, [])
                            continue
                        self.assertTrue(fronts)
                        initial = crest_positions(initial_fronts[index], vector, offset, 0.)
                        current = crest_positions(fronts, vector, offset, tau)
                        shared = initial.keys() & current.keys()
                        self.assertTrue(shared)
                        for order in shared:
                            self.assertAlmostEqual(current[order]-initial[order], 2*tau)
        view.close()
        view.close()
        self.assertIsNone(view._local_time_artists)
        self.assertIsNone(view.local_data)
        self.assertIsNone(view.local_cycle_data)
        self.assertIsNone(view.local_readings)
        self.assertIsNone(view.wavefronts)
        self.assertEqual(len(view._cache), 0)

    def test_two_order_time_preserves_intensity_and_detector_shift_preserves_spatial_field(self):
        view = self.explorer(superposition_view.TwoWaveExplorer, 'orders')
        for mode in ('experiment', 'fields'):
            view.order_view.value = mode
            intensity = view.spatial_data['relative_intensity']
            axes = tuple(view.figure.axes)
            with patch.object(superposition_view, 'two_wave_field', wraps=superposition_view.two_wave_field) as evaluate:
                with patch.object(superposition_view, 'ideal_camera_profile', wraps=superposition_view.ideal_camera_profile) as camera:
                    view.time.value = .25 if view.time.value != .25 else .5
                    evaluate.assert_not_called()
                    camera.assert_not_called()
                    self.assertIs(view.spatial_data['relative_intensity'], intensity)
                    self.assertEqual(tuple(view.figure.axes), axes)
                    _, _, points, vectors = view._order_grid()
                    expected = superposition.two_wave_field(points, vectors, tau=view.time.value)
                    assert_allclose(view.spatial_data['field_sum'], expected['field_sum'])
                    for artist, index in view._order_time_artists:
                        field = expected['field_sum'] if index is None else expected['fields'][index]
                        assert_allclose(np.asarray(artist.get_array()), field)
                    view.detector_z.value = -1.5 if view.detector_z.value != -1.5 else -2.
                    evaluate.assert_not_called()
                    self.assertEqual(camera.call_count, 1)
        old = view.spatial_data['relative_intensity'].copy()
        view.phase2.value = 90.
        self.assertFalse(np.allclose(old, view.spatial_data['relative_intensity']))
        view.close()
        self.assertEqual(len(view._cache), 0)


if __name__ == '__main__':
    unittest.main()
