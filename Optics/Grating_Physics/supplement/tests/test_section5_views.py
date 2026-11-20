"""Check the focused wavefront, local-phase, and fringe-spacing views."""

import gc
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.superposition_view import BLUE, ORANGE, TwoWaveExplorer


class SectionFiveViewTests(unittest.TestCase):
    def test_crest_rulers_join_adjacent_positive_crests_along_propagation(self):
        lesson = TwoWaveExplorer('wavefronts')
        vectors = np.pi*np.array([[-.5, 0., -np.sqrt(3)/2],
                                  [.5, 0., -np.sqrt(3)/2]])
        try:
            for phi1, phi2, tau in ((0, 0, 0), (60, 135, .25), (300, 45, .625)):
                with self.subTest(phi1=phi1, phi2=phi2, tau=tau):
                    lesson.phase1.value = phi1
                    lesson.phase2.value = phi2
                    lesson.time.value = tau
                    for index, (vector, phase) in enumerate(zip(vectors, (phi1, phi2))):
                        ruler = lesson.crest_rulers[index]
                        self.assertIsNotNone(ruler)
                        self.assertEqual(ruler.shape, (2, 3))
                        crest_orders = (ruler@vector+np.deg2rad(phase)-2*np.pi*tau)/(2*np.pi)
                        assert_allclose(crest_orders, np.round(crest_orders), atol=1e-13)
                        assert_allclose(np.diff(crest_orders), [1], atol=1e-13)
                        assert_allclose(ruler[1]-ruler[0], 2*vector/np.pi, atol=1e-13)
                        axis = lesson.figure.axes[index]
                        self.assertEqual(axis.get_aspect(), 1)
                        arrows = [item for item in axis.texts if getattr(item, 'arrowprops', None)]
                        self.assertEqual(sum(item.arrowprops['arrowstyle'] == '->' for item in arrows), 1)
                        drawn_ruler = next(item for item in arrows if item.arrowprops['arrowstyle'] == '<->')
                        assert_allclose(drawn_ruler.get_position(), ruler[0, [0, 2]])
                        assert_allclose(drawn_ruler.xy, ruler[1, [0, 2]])
            lesson.amplitude2.value = 0
            self.assertIsNone(lesson.crest_rulers[1])
            self.assertEqual(lesson.wavefronts[1], [])
            self.assertIsNotNone(lesson.crest_rulers[0])
            lesson.amplitude1.value = 0
            self.assertEqual(lesson.crest_rulers, [None, None])
            self.assertEqual(lesson.wavefronts, [[], []])
        finally:
            lesson.close()
            lesson.close()
            self.assertIsNone(lesson.crest_rulers)
            self.assertIsNone(lesson.wavefronts)
            self.assertIsNone(lesson.figure)
            self.assertEqual(bytes(lesson.picture.value), b'')
            del lesson
            gc.collect()

    def test_selected_point_connects_incident_crests_phases_and_wave_cycles(self):
        lesson = TwoWaveExplorer('local_phase')
        vectors = np.pi*np.array([[-.5, 0., -np.sqrt(3)/2],
                                  [.5, 0., -np.sqrt(3)/2]])
        limits = [(axis.get_xlim(), axis.get_ylim()) for axis in lesson.figure.axes]

        def arrows(axis):
            return [item for item in axis.texts
                    if getattr(item, 'arrowprops', None)
                    and item.arrowprops.get('arrowstyle') == '->']

        def check_selected_point():
            x = lesson.detector_x.value
            tau = lesson.time.value
            point = np.array([x, 0., -1.])
            amplitudes = np.array([lesson.amplitude1.value, lesson.amplitude2.value])
            phases = np.deg2rad([lesson.phase1.value, lesson.phase2.value])
            expected = amplitudes*np.exp(1j*(vectors@point+phases))
            rotating = expected*np.exp(-2j*np.pi*tau)
            assert_allclose(lesson.local_data['phasors'], expected, atol=1e-13)
            assert_allclose(lesson.local_data['total_phasor'], expected.sum(), atol=1e-13)
            assert_allclose(lesson.local_data['relative_intensity'], abs(expected.sum())**2, atol=1e-13)
            assert_allclose(lesson.local_data['phase_difference'], np.pi*x+phases[1]-phases[0])
            assert_allclose(lesson.local_readings, rotating.real, atol=1e-13)
            scene, readings, cycle = lesson.figure.axes
            selected = next(line for line in scene.lines if line.get_label() == 'Selected point P')
            assert_allclose(selected.get_xdata(), [x])
            assert_allclose(selected.get_ydata(), [-1])

            # One ray direction per wave terminates at the selected point.
            incidence = arrows(scene)
            self.assertEqual(len(incidence), 2)
            for arrow, vector in zip(incidence, vectors):
                assert_allclose(arrow.xy, point[[0, 2]], atol=1e-13)
                displacement = np.asarray(arrow.xy)-arrow.get_position()
                assert_allclose(displacement/np.linalg.norm(displacement),
                                vector[[0, 2]]/np.pi, atol=1e-13)

            # Simultaneous real readings add with their signs; no complex-plane
            # arrows are needed to compare the two cycles at the detector.
            self.assertEqual(arrows(readings), [])
            assert_allclose([bar.get_height() for bar in readings.patches],
                            [*rotating.real, rotating.real.sum()], atol=1e-13)
            for index, (amplitude, color) in enumerate(zip(amplitudes, (BLUE, ORANGE))):
                if amplitude == 0:
                    self.assertEqual(lesson.wavefronts[index], [])
                    continue
                self.assertTrue(lesson.wavefronts[index])
                for front in lesson.wavefronts[index]:
                    orders = (np.asarray(front)@vectors[index]+phases[index]-2*np.pi*tau)/(2*np.pi)
                    assert_allclose(orders, np.round(orders), atol=1e-13)
                    assert_allclose(orders, orders[0], atol=1e-13)

            # The time axis shows whole cycles at P; the selected time moves
            # the cursor and readings without translating those full curves.
            cycle_tau = np.linspace(0, 1, 401)
            expected_fields = amplitudes[:, None]*np.cos(
                (vectors@point+phases)[:, None]-2*np.pi*cycle_tau)
            assert_allclose(lesson.local_cycle_data['tau'], cycle_tau, atol=1e-13)
            assert_allclose(lesson.local_cycle_data['fields'], expected_fields, atol=1e-13)
            cycle_lines = {line.get_label(): line for line in cycle.lines}
            self.assertEqual(sum(line.get_label() == 'Selected time' for line in cycle.lines), 1)
            assert_allclose(cycle_lines['Selected time'].get_xdata(), [tau, tau])
            for index, color in enumerate((BLUE, ORANGE)):
                trace = cycle_lines[f'Wave {index+1} at P']
                marker = cycle_lines[f'Wave {index+1} at selected time']
                assert_allclose(trace.get_xdata(), cycle_tau, atol=1e-13)
                assert_allclose(trace.get_ydata(), expected_fields[index], atol=1e-13)
                self.assertEqual(trace.get_color(), color)
                self.assertEqual(trace.get_linestyle(), '-' if index == 0 else '--')
                assert_allclose(marker.get_xdata(), [tau])
                assert_allclose(marker.get_ydata(), [rotating[index].real], atol=1e-13)
                assert_allclose(marker.get_ydata(), [lesson.local_data['fields'][index]], atol=1e-13)
            assert_allclose(cycle.get_xlim(), [0, 1])
            assert_allclose(cycle.get_ylim(), [-1.2, 1.2])
            self.assertEqual(cycle.get_aspect(), 'auto')

            for axis, (xlim, ylim) in zip(lesson.figure.axes, limits):
                assert_allclose(axis.get_xlim(), xlim)
                assert_allclose(axis.get_ylim(), ylim)
            self.assertEqual(scene.get_aspect(), 1)

        try:
            self.assertEqual(len(lesson.figure.axes), 3)
            self.assertEqual((lesson.detector_x.min, lesson.detector_x.max), (-2, 2))
            check_selected_point()
            assert_allclose(lesson.local_cycle_data['fields'][0], lesson.local_cycle_data['fields'][1])
            initial_cycles = lesson.local_cycle_data['fields'].copy()
            lesson.detector_x.value = 1
            check_selected_point()
            assert_allclose(lesson.local_cycle_data['fields'][0], -lesson.local_cycle_data['fields'][1], atol=1e-13)
            self.assertGreater(np.max(abs(lesson.local_cycle_data['fields']-initial_cycles)), 1)
            lesson.detector_x.value = 0
            # The shared k_z z phase is essential even at x=0, where the
            # controlled origin offsets are zero but the local phasors are not 1.
            self.assertGreater(abs(lesson.local_data['phasors'][0]-1), 1)
            lesson.detector_x.value = .5
            lesson.amplitude2.value = .6
            before = lesson.local_data['phasors'].copy()
            before_cycles = lesson.local_cycle_data['fields'].copy()
            lesson.phase1.value = 60
            assert_allclose(lesson.local_data['phasors'][1], before[1])
            assert_allclose(lesson.local_data['phasors'][0], before[0]*np.exp(1j*np.pi/3))
            assert_allclose(lesson.local_cycle_data['fields'][1], before_cycles[1], atol=1e-13)
            self.assertGreater(np.max(abs(lesson.local_cycle_data['fields'][0]-before_cycles[0])), .9)
            check_selected_point()
            before = lesson.local_data['phasors'].copy()
            before_cycles = lesson.local_cycle_data['fields'].copy()
            lesson.phase2.value = 135
            assert_allclose(lesson.local_data['phasors'][0], before[0])
            assert_allclose(lesson.local_data['phasors'][1], before[1]*np.exp(1j*3*np.pi/4))
            assert_allclose(lesson.local_cycle_data['fields'][0], before_cycles[0], atol=1e-13)
            self.assertGreater(np.max(abs(lesson.local_cycle_data['fields'][1]-before_cycles[1])), 1)
            check_selected_point()
            for x in (-1.75, 0, 1.25, 2):
                lesson.detector_x.value = x
                check_selected_point()

            before = lesson.local_data['phasors'].copy()
            before_intensity = float(lesson.local_data['relative_intensity'])
            before_difference = float(lesson.local_data['phase_difference'])
            before_cycles = lesson.local_cycle_data['fields'].copy()
            before_readings = lesson.local_data['fields'].copy()
            for tau in (.25, .625, 1):
                lesson.time.value = tau
                check_selected_point()
                assert_allclose(lesson.local_data['phasors'], before, atol=1e-13)
                assert_allclose(lesson.local_data['relative_intensity'], before_intensity, atol=1e-13)
                assert_allclose(lesson.local_data['phase_difference'], before_difference, atol=1e-13)
                assert_allclose(lesson.local_cycle_data['fields'], before_cycles, atol=1e-13)
                if tau != 1:
                    self.assertGreater(np.max(abs(lesson.local_data['fields']-before_readings)), .1)

            before = lesson.local_data['phasors'].copy()
            lesson.phase1.value = 105
            lesson.phase2.value = 180
            assert_allclose(lesson.local_data['phasors'], before*np.exp(1j*np.pi/4), atol=1e-13)
            assert_allclose(lesson.local_data['relative_intensity'], before_intensity, atol=1e-13)
            check_selected_point()

            lesson.amplitude2.value = 0
            check_selected_point()
            self.assertIn('undefined', lesson.readout.value.lower())
            lesson.amplitude1.value = 0
            check_selected_point()
            assert_allclose(lesson.local_data['relative_intensity'], 0)
            lesson.reset_button.click()
            self.assertEqual(lesson.time.value, 0)
            self.assertEqual(lesson.detector_x.value, 0)
            self.assertEqual((lesson.amplitude1.value, lesson.amplitude2.value), (1, 1))
            self.assertEqual((lesson.phase1.value, lesson.phase2.value), (0, 0))
            check_selected_point()
        finally:
            lesson.close()
            lesson.close()
            self.assertIsNone(lesson.local_data)
            self.assertIsNone(lesson.local_cycle_data)
            self.assertIsNone(lesson.local_readings)
            self.assertIsNone(lesson.wavefronts)
            self.assertIsNone(lesson.figure)
            self.assertEqual(bytes(lesson.picture.value), b'')
            del lesson
            gc.collect()

    @staticmethod
    def fringe_ruler(lesson):
        rulers = [line for line in lesson.figure.axes[1].lines if line.get_label() == 'Fringe ruler']
        return np.asarray(rulers[0].get_xdata())[[0, 3]] if rulers else None

    def test_spacing_ruler_marks_profile_maxima_and_vector_difference(self):
        lesson = TwoWaveExplorer('spacing')
        try:
            lesson.amplitude2.value = .6
            for theta in (15, 30, 45):
                with self.subTest(theta=theta):
                    lesson.angle.value = theta
                    radians = np.deg2rad(theta)
                    delta_kx = 2*np.pi*np.sin(radians)
                    ruler = self.fringe_ruler(lesson)
                    self.assertIsNotNone(ruler)
                    assert_allclose(np.diff(ruler), [2*np.pi/delta_kx])
                    assert_allclose(1.36+1.2*np.cos(delta_kx*ruler), 2.56, atol=1e-13)
                    directions, profile = lesson.figure.axes
                    row_x = profile.lines[0].get_xdata()
                    assert_allclose(profile.lines[0].get_ydata(),
                                    1.36+1.2*np.cos(delta_kx*row_x), atol=1e-13)
                    difference = next(item for item in directions.texts
                                      if getattr(item, 'arrowprops', None)
                                      and item.arrowprops.get('color') == '#79539e')
                    assert_allclose(difference.get_position(), [-np.sin(radians), -np.cos(radians)])
                    assert_allclose(difference.xy, [np.sin(radians), -np.cos(radians)])
            lesson.angle.value = 30
            lesson.phase2.value = 30
            before = self.fringe_ruler(lesson)
            lesson.phase2.value = 60
            after = self.fringe_ruler(lesson)
            assert_allclose(after-before, -np.deg2rad(30)/np.pi, atol=1e-13)
            assert_allclose(np.diff(after), np.diff(before), atol=1e-13)
            assert_allclose(1.36+1.2*np.cos(np.pi*after+np.pi/3), 2.56, atol=1e-13)
        finally:
            lesson.close()
            self.assertIsNone(lesson.spatial_data)
            del lesson
            gc.collect()

    def test_spacing_ruler_is_hidden_without_two_visible_bright_fringes(self):
        lesson = TwoWaveExplorer('spacing')
        try:
            lesson.angle.value = 5
            self.assertIsNone(self.fringe_ruler(lesson))
            self.assertIn('Two bright fringes do not fit', lesson.readout.value)
            # At this same angle a half-cycle shift brings two maxima into
            # the window, so ruler visibility must depend on phase as well.
            lesson.phase2.value = 180
            ruler = self.fringe_ruler(lesson)
            self.assertIsNotNone(ruler)
            assert_allclose(np.diff(ruler), [1/np.sin(np.deg2rad(5))])
            lesson.angle.value = 30
            lesson.amplitude2.value = 0
            self.assertIsNone(self.fringe_ruler(lesson))
            assert_allclose(lesson.figure.axes[1].lines[0].get_ydata(), 1)
            lesson.amplitude1.value = 0
            self.assertIsNone(self.fringe_ruler(lesson))
            assert_allclose(lesson.figure.axes[1].lines[0].get_ydata(), 0)
        finally:
            lesson.close()
            del lesson
            gc.collect()


if __name__ == '__main__':
    unittest.main()
