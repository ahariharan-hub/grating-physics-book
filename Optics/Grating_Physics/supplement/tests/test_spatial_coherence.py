"""Spatial coherence agrees with independent source-point field averaging."""

import gc
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from wave_optics.spatial_coherence import (
    SpatialCoherenceExplorer, uniform_line_coherence, young_spatial_data,
)


class SpatialCoherenceTests(unittest.TestCase):
    def test_detector_matches_independent_source_point_field_sum(self):
        nodes, weights = np.polynomial.legendre.leggauss(160)
        wavelength, source_distance, detector_distance = .0006328, 500., 1000.
        k = 2*np.pi/wavelength
        for separation, width in ((.5, 0.), (.5, .3), (.5, .6328),
                                  (.5, .9), (1., 2.), (.1, 2.)):
            with self.subTest(separation=separation, width=width):
                data = young_spatial_data(separation, width)
                source_x = width/2*nodes[:, None]
                detector_x = data['x_mm'][None, :]
                fields = []
                illumination = []
                for aperture_x in (-separation/2, separation/2):
                    source_path = (aperture_x-source_x)**2/(2*source_distance)
                    detector_path = (detector_x-aperture_x)**2/(2*detector_distance)
                    illumination.append(np.exp(1j*k*source_path[:, 0]))
                    fields.append(np.exp(1j*k*(source_path+detector_path)))
                # Independent emitters add intensities; each opening alone has intensity one.
                averaged = (weights @ np.abs(fields[0]+fields[1])**2)/4
                correlation = np.sum(weights*illumination[0].conj()*illumination[1])/2
                assert_allclose(data['profile'], averaged, atol=3e-12)
                assert_allclose(data['coherence'], correlation, atol=3e-13)
                assert_allclose(data['image'], np.broadcast_to(averaged, data['image'].shape),
                                atol=3e-12)

    def test_limits_zero_and_revival_preserve_signed_correlation(self):
        assert_array_equal(uniform_line_coherence([0., .1, 1.], 0.), np.ones(3))
        self.assertEqual(float(uniform_line_coherence(0., 2.)), 1.)
        for separation in (.1, .5, 1.):
            zero_width = .0006328*500/separation
            zero = young_spatial_data(separation, zero_width)
            assert_allclose(zero['profile'], 1., atol=1e-14)
            self.assertIsNone(zero['coherence_phase_deg'])
            self.assertLess(zero['visibility'], 1e-14)
        revival = young_spatial_data(.5, .9)
        self.assertLess(revival['coherence'], 0)
        self.assertGreater(revival['visibility'], .2)
        self.assertEqual(revival['coherence_phase_deg'], 180)
        self.assertLess(revival['profile'][320], 1.)  # Darker centre after the sign change.

    def test_fringe_period_changes_with_separation_not_source_width(self):
        for separation in (.5, 1.):
            periods = []
            for width in (0., .2, .9):
                data = young_spatial_data(separation, width)
                profile = data['profile']
                maxima = np.where((profile[1:-1] > profile[:-2])
                                  & (profile[1:-1] >= profile[2:]))[0]+1
                measured = np.mean(np.diff(data['x_mm'][maxima]))
                self.assertAlmostEqual(measured, data['fringe_spacing_mm'], delta=.013)
                periods.append(data['fringe_spacing_mm'])
            self.assertEqual(len(set(periods)), 1)
        self.assertAlmostEqual(young_spatial_data(1.)['fringe_spacing_mm'],
                               young_spatial_data(.5)['fringe_spacing_mm']/2)

    def test_controls_reuse_artists_and_close_detaches_observers(self):
        explorer = SpatialCoherenceExplorer()
        figure = explorer.figure
        identities = [tuple(map(id, ax.get_children())) for ax in figure.axes]
        try:
            self.assertTrue(bytes(explorer.picture.value).startswith(b'\x89PNG'))
            with patch('wave_optics.plotting.figure_png', return_value=b'preview') as render:
                explorer.source_width.value = .9
                self.assertLess(explorer.data['coherence'], 0)
                self.assertIn('half a period', explorer.readout.value)
                explorer.separation.value = .1
                self.assertGreater(explorer.data['coherence'], 0)
                explorer.source_width.value = 0.
                self.assertEqual(explorer.data['visibility'], 1.)
                self.assertFalse(explorer.first_zero_guide.get_visible())
                self.assertEqual(sum(line.get_visible() for line in explorer.source_curves), 1)
                self.assertEqual(render.call_count, 3)
            self.assertIs(explorer.figure, figure)
            self.assertEqual(identities, [tuple(map(id, ax.get_children())) for ax in figure.axes])
        finally:
            explorer.close()
            explorer.close()
        self.assertEqual(explorer.picture.value, b'')
        self.assertIsNone(explorer.ui.comm)
        self.assertIsNone(explorer.data)
        with patch('wave_optics.plotting.figure_png', side_effect=AssertionError('closed render')):
            explorer.separation.value = .5
            explorer.source_width.value = 1.
        del explorer, figure
        gc.collect()

    def test_invalid_physical_parameters(self):
        for arguments in ((-1., 1.), (1., -1.), (np.nan, 1.), (1., np.inf)):
            with self.assertRaises(ValueError):
                uniform_line_coherence(*arguments)
        for parameters in ({'separation_mm': 0.}, {'detector_distance_mm': 0.},
                           {'source_distance_mm': -1.}, {'wavelength_nm': 0.}):
            with self.assertRaises(ValueError):
                young_spatial_data(**parameters)


if __name__ == '__main__':
    unittest.main()
