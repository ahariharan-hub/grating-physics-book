"""Check two-wave physics, phase conventions, and notebook-seven controls."""

import gc
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.superposition import (
    grating_pair_wavevectors, ideal_camera_profile, symmetric_wavevectors,
    two_wave_cycle, two_wave_field,
)
from wave_optics.superposition_view import BLUE, ORANGE, STAGES, TwoWaveExplorer, _wavefront_patches


class SuperpositionTests(unittest.TestCase):
    def test_real_cycles_agree_with_complex_sum_and_interference_formula(self):
        a1, a2, phi1, delta = 1.3, .7, -37, 123
        data = two_wave_cycle(a1, a2, delta, phase1_deg=phi1, sample_count=397)
        expected = (a1*np.cos(np.deg2rad(phi1)-2*np.pi*data['tau'])
                    + a2*np.cos(np.deg2rad(phi1+delta)-2*np.pi*data['tau']))
        intensity = a1*a1+a2*a2+2*a1*a2*np.cos(np.deg2rad(delta))
        assert_allclose(data['field_sum'], expected, atol=1e-14)
        assert_allclose(data['relative_intensity'], intensity)
        assert_allclose(data['mean_square'], intensity/2)
        assert_allclose(data['mean_field'], 0, atol=1e-14)
        assert_allclose(data['separate_intensities']+data['interference_term'], intensity)

    def test_cancelled_amplitude_differs_from_a_zero_crossing(self):
        cancelled = two_wave_cycle(1, 1, 180)
        in_phase = two_wave_cycle(1, 1, 0)
        unequal = two_wave_cycle(1, .5, 180)
        quarter_phase = two_wave_cycle(1, 1, 90)
        assert_allclose(cancelled['field_sum'], 0, atol=1e-14)
        self.assertIsNone(cancelled['phase_deg'])
        assert_allclose(in_phase['field_sum'][100], 0, atol=1e-14)
        self.assertEqual(in_phase['relative_intensity'], 4)
        assert_allclose(unequal['amplitude'], .5)
        assert_allclose(unequal['relative_intensity'], .25)
        assert_allclose(quarter_phase['total_phasor'], 1+1j)
        assert_allclose(quarter_phase['phase_deg'], 45)
        assert_allclose(quarter_phase['relative_intensity'], 2)

    def test_common_phase_preserves_intensity_and_absent_wave_has_no_phase_effect(self):
        base = two_wave_cycle(1, .7, 83)
        rotated = two_wave_cycle(1, .7, 83, phase1_deg=61)
        assert_allclose(rotated['total_phasor'], base['total_phasor']*np.exp(1j*np.deg2rad(61)))
        assert_allclose(rotated['relative_intensity'], base['relative_intensity'])
        for phase in (0, 90, 180, 270):
            absent = two_wave_cycle(1, 0, phase)
            assert_allclose(absent['total_phasor'], 1)

    def test_grating_pair_matches_two_equal_length_negative_z_vectors(self):
        vectors = symmetric_wavevectors(2, 30)
        assert_allclose(vectors, grating_pair_wavevectors(), atol=1e-14)
        assert_allclose(np.linalg.norm(vectors, axis=1), np.pi)
        assert_allclose(vectors[1]-vectors[0], [np.pi, 0, 0], atol=1e-14)
        self.assertTrue(np.all(vectors[:, 2] < 0))

    def test_fringe_spacing_and_origin_phase_shift(self):
        vectors = grating_pair_wavevectors()
        x = np.array([0, .5, 1, 2])
        points = np.column_stack((x, np.zeros_like(x), -np.ones_like(x)))
        field = two_wave_field(points, vectors)
        assert_allclose(field['relative_intensity'], [4, 2, 0, 4], atol=1e-14)
        assert_allclose(field['phase_difference'], np.pi*x)
        shifted_phase = two_wave_field(points, vectors, phase_difference_deg=90)
        shifted_points = points.copy()
        shifted_points[:, 0] += .5
        shifted_position = two_wave_field(shifted_points, vectors)
        assert_allclose(shifted_phase['relative_intensity'], shifted_position['relative_intensity'], atol=1e-14)
        shifted_points[:, 0] = points[:, 0]+2
        shifted_points[:, 2] -= 1.37
        repeated = two_wave_field(shifted_points, vectors)
        assert_allclose(repeated['relative_intensity'], field['relative_intensity'], atol=1e-14)

    def test_time_changes_fields_but_not_intensity_and_spatial_mean_cancels_cross_term(self):
        vectors = grating_pair_wavevectors()
        x = np.linspace(0, 2, 401, endpoint=False)
        points = np.column_stack((x, np.zeros_like(x), -np.ones_like(x)))
        initial = two_wave_field(points, vectors, amplitude2=.5, phase_difference_deg=37)
        later = two_wave_field(points, vectors, amplitude2=.5, phase_difference_deg=37, tau=.25)
        self.assertGreater(np.max(abs(initial['field_sum']-later['field_sum'])), .1)
        assert_allclose(later['relative_intensity'], initial['relative_intensity'])
        assert_allclose(np.mean(initial['relative_intensity']), 1.25)
        same_direction = symmetric_wavevectors(2, 0)
        uniform = two_wave_field(points, same_direction, phase_difference_deg=90)
        assert_allclose(uniform['relative_intensity'], 2)

    def test_all_sections_update_preserve_scales_and_close(self):
        for stage in STAGES:
            with self.subTest(stage=stage):
                lesson = TwoWaveExplorer(stage)
                limits = [(axis.get_xlim(), axis.get_ylim()) for axis in lesson.figure.axes]
                before = bytes(lesson.picture.value)
                try:
                    if lesson.time is not None:
                        lesson.time.value = .25
                    lesson.phase1.value = 75
                    lesson.phase2.value = 255
                    if lesson.angle is not None:
                        lesson.angle.value = 15
                    self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
                    self.assertNotEqual(before, bytes(lesson.picture.value))
                    if stage == 'phasor':
                        self.assertIn('undefined', lesson.readout.value)
                    lesson.amplitude1.value = .5
                    lesson.amplitude2.value = .8
                    lesson.phase1.value = 210
                    lesson.phase2.value = 30
                    expected = .5*np.exp(1j*np.deg2rad(210)) + .8*np.exp(1j*np.deg2rad(30))
                    assert_allclose(lesson.data['total_phasor'], expected, atol=1e-14)
                    self.assertEqual(lesson.phase_difference_deg, -180)
                    before_common = lesson.data['total_phasor']
                    before_intensity = lesson.data['relative_intensity']
                    if stage in ('directions', 'orders'):
                        before_map = lesson.spatial_data['relative_intensity'].copy()
                    lesson.phase1.value = 270
                    lesson.phase2.value = 90
                    assert_allclose(lesson.data['total_phasor'], before_common*np.exp(1j*np.pi/3), atol=1e-14)
                    assert_allclose(lesson.data['relative_intensity'], before_intensity, atol=1e-14)
                    if stage in ('directions', 'orders'):
                        assert_allclose(lesson.spatial_data['relative_intensity'], before_map, atol=1e-14)
                    lesson.amplitude2.value = 0
                    assert_allclose(lesson.data['relative_intensity'], .25)
                    self.assertIn('Wave 2 is absent', lesson.parameter_readout.value)
                    if stage == 'orders':
                        assert_allclose(lesson.camera_data['signal_dn'], 35)
                    lesson.amplitude1.value = 0
                    assert_allclose(lesson.data['relative_intensity'], 0)
                    self.assertIn('Both waves are absent', lesson.parameter_readout.value)
                    if stage == 'orders':
                        assert_allclose(lesson.camera_data['signal_dn'], 10)
                    self.assertEqual(len(lesson.figure.axes), len(limits))
                    for axis, (xlim, ylim) in zip(lesson.figure.axes, limits):
                        assert_allclose(axis.get_xlim(), xlim)
                        assert_allclose(axis.get_ylim(), ylim)
                    lesson.reset_button.click()
                    self.assertEqual(lesson.amplitude1.value, 1)
                    self.assertEqual(lesson.amplitude2.value, 1)
                    self.assertEqual(lesson.phase1.value, 0)
                    self.assertEqual(lesson.phase2.value, 90 if stage == 'phasor' else 0)
                finally:
                    lesson.close()
                    lesson.close()
                self.assertIsNone(lesson.figure)
                self.assertEqual(bytes(lesson.picture.value), b'')
                self.assertIsNone(lesson.ui.comm)
                self.assertFalse(lesson.reset_button._click_handlers.callbacks)
                del lesson
                gc.collect()

    def test_camera_calibration_matches_temporally_averaged_real_fields(self):
        x = np.array([0, .5, 1, 1.5])
        z, a1, a2, phi1, delta = -1.3, .7, .5, .91, .43
        tau = np.linspace(0, 1, 397, endpoint=False)
        kx, kz = np.pi/2, -np.sqrt(3)*np.pi/2
        real_sum = (a1*np.cos(-kx*x[:, None]+kz*z+phi1-2*np.pi*tau)
                    + a2*np.cos(kx*x[:, None]+kz*z+phi1+delta-2*np.pi*tau))
        data = ideal_camera_profile(x, z, a2, np.rad2deg(delta),
                                    amplitude1=a1, phase1_deg=np.rad2deg(phi1))
        # Reference-amplitude-one calibration: twice the cycle mean square.
        assert_allclose(data['signal_dn'], 10+100*2*np.mean(real_sum**2, axis=1))
        unit_pair = ideal_camera_profile([0, 1])
        assert_allclose(unit_pair['signal_dn'], [410, 10], atol=1e-12)
        one_wave = ideal_camera_profile(x, amplitude2=0)
        assert_allclose(one_wave['signal_dn'], 110)
        assert_allclose(np.sqrt((data['signal_dn']-10)/100), abs(data['total_phasor']))

    def test_drawn_wavefronts_are_crest_planes_advancing_along_the_wavevector(self):
        for angle in (5, 30, 45):
            for vector in symmetric_wavevectors(2, angle):
                k = np.linalg.norm(vector)
                initial_positions = {}
                later_positions = {}
                for tau, positions in ((0, initial_positions), (.25, later_positions)):
                    patches = _wavefront_patches(vector, 75, tau)
                    self.assertGreaterEqual(len(patches), 2)
                    for patch in patches:
                        # Every vertex lies on one crest plane and on the
                        # incoming side of the observation plane.
                        phase = patch@vector+np.deg2rad(75)-2*np.pi*tau
                        assert_allclose(phase, phase[0], atol=1e-13)
                        order = round(phase[0]/(2*np.pi))
                        assert_allclose(phase, 2*np.pi*order, atol=1e-13)
                        self.assertTrue(np.all(patch[:, 2] >= -1-1e-12))
                        self.assertTrue(np.all(patch[:, 2] <= 5+1e-12))
                        positions[order] = np.mean(patch@vector)/k
                shared = initial_positions.keys() & later_positions.keys()
                self.assertTrue(shared)
                for order in shared:
                    # lambda=2 um: a quarter cycle advances a crest by +0.5 um.
                    assert_allclose(later_positions[order]-initial_positions[order], .5)
                orders = sorted(initial_positions)
                for first, second in zip(orders, orders[1:]):
                    assert_allclose(initial_positions[second]-initial_positions[first],
                                    2*(second-first))

    def test_directions_depiction_and_detector_share_controls_and_intensity(self):
        lesson = TwoWaveExplorer('directions')
        try:
            self.assertEqual(lesson.time.description, 'Model time t/T')
            image = lesson.screen_intensity.copy()
            for row in image:
                assert_allclose(row, lesson.spatial_data['relative_intensity'])
            # The actual plotted detector and profile also use these values.
            assert_allclose(lesson.figure.axes[1].images[0].get_array(), image)
            assert_allclose(lesson.figure.axes[3].lines[0].get_ydata(), image[0])
            # The orthogonal side view preserves perpendicularity on screen.
            side = lesson.figure.axes[0]
            self.assertEqual(side.get_aspect(), 1)
            for line in side.lines:
                if line.get_color() in (BLUE, ORANGE):
                    dx = np.diff(line.get_xdata())[0]
                    dz = np.diff(line.get_ydata())[0]
                    wave = 0 if line.get_color() == BLUE else 1
                    assert_allclose(np.array([dx, 0, dz])@symmetric_wavevectors(2, 30)[wave],
                                    0, atol=1e-13)
            detector = lesson.figure.axes[1]
            ruler_x = np.array([line.get_xdata()[0] for line in detector.lines[1:3]])
            np.testing.assert_allclose(np.diff(ruler_x), [2])
            ruler_points = np.column_stack((ruler_x, np.zeros(2), -np.ones(2)))
            assert_allclose(two_wave_field(ruler_points, symmetric_wavevectors(2, 30))['relative_intensity'], 4)
            initial_front = lesson.wavefronts[0][1].copy()
            lesson.time.value = .25
            assert_allclose(lesson.screen_intensity, image)
            self.assertFalse(np.allclose(lesson.wavefronts[0][1], initial_front))
            lesson.phase2.value = 180
            self.assertGreater(np.max(abs(lesson.screen_intensity-image)), 3)
            lesson.amplitude2.value = 0
            self.assertEqual(lesson.wavefronts[1], [])
            assert_allclose(lesson.screen_intensity, 1)
            lesson.amplitude1.value = 0
            self.assertEqual(lesson.wavefronts[0], [])
            assert_allclose(lesson.screen_intensity, 0)
            lesson.reset_button.click()
            self.assertEqual(lesson.time.value, 0)
            assert_allclose(lesson.screen_intensity, image)
        finally:
            lesson.close()
            self.assertIsNone(lesson.screen_intensity)
            self.assertIsNone(lesson.wavefronts)
            del lesson
            gc.collect()

    def test_camera_fringe_position_recovers_relative_phase_with_known_origin(self):
        x = np.array([-.5, 0, .5, 1])
        initial = ideal_camera_profile(x, phase_difference_deg=90)
        assert_allclose(initial['signal_dn'], [410, 210, 10, 210], atol=1e-12)
        bright_x = x[np.argmax(initial['signal_dn'])]
        assert_allclose(np.rad2deg(-np.pi*bright_x) % 360, 90)
        later_and_farther = ideal_camera_profile(x, -2.5, phase_difference_deg=90, tau=.37)
        assert_allclose(later_and_farther['signal_dn'], initial['signal_dn'], atol=1e-12)

    def test_detector_controls_connect_theory_and_camera_views(self):
        lesson = TwoWaveExplorer('orders')
        try:
            signal = lesson.camera_data['signal_dn'].copy()
            field = lesson.spatial_data['field_sum'].copy()
            lesson.time.value = .25
            lesson.detector_z.value = -2.5
            assert_allclose(lesson.camera_data['signal_dn'], signal, atol=1e-12)
            self.assertGreater(np.max(abs(lesson.spatial_data['field_sum']-field)), .1)
            lesson.phase2.value = 90
            self.assertGreater(np.max(abs(lesson.camera_data['signal_dn']-signal)), 100)
            lesson.order_view.value = 'fields'
            lesson.amplitude2.value = 0
            assert_allclose(lesson.camera_data['signal_dn'], 110)
            assert_allclose(lesson.spatial_data['relative_intensity'], 1)
            lesson.order_view.value = 'experiment'
            self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
        finally:
            lesson.close()
            del lesson
            gc.collect()


if __name__ == '__main__':
    unittest.main()
