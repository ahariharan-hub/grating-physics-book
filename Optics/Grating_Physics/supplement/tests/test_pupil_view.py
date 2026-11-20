"""Physical mapping and independent resource ownership of the 5.4 view."""
import gc
import unittest
import weakref

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from wave_optics.pupil_view import SigmaPupilExplorer


class SigmaPupilTests(unittest.TestCase):
    def test_baseline_and_independent_polar_sine_mapping(self):
        view = SigmaPupilExplorer()
        try:
            d = view.data
            expected = d['source_sigma'] + d['labels']*np.array([
                view.wavelength0/(d['dx']*view.NA_mask), view.wavelength0/(d['dy']*view.NA_mask)])
            assert_allclose(d['q'], expected, atol=1e-14)
            transverse = d['k'][:, :2].real/(view.n_mask*d['k0'])
            phi = np.arctan2(transverse[:, 1], transverse[:, 0])
            r = d['R_mm']*view.n_mask/view.NA_mask*np.linalg.norm(transverse, axis=1)
            independent = r[:, None]*np.column_stack([np.cos(phi), np.sin(phi)])
            assert_allclose(d['pupil_xy_mm'], independent, atol=1e-13)
            row = [np.flatnonzero(np.all(d['labels']==[m, 0], axis=1))[0] for m in (-1, 0, 1)]
            assert_array_equal(d['collected'][row], [True, True, False])
            self.assertEqual(np.count_nonzero(d['collected']), 6)
        finally:
            view.close()

    def test_controls_separate_source_pitch_and_physical_scale(self):
        view = SigmaPupilExplorer()
        try:
            from ipywidgets.widgets.widget import _instances
            population = set(_instances)
            original = view.data
            view.pupil_radius.value = 4
            assert_allclose(view.data['q'], original['q'])
            assert_allclose(view.data['k'], original['k'])
            assert_allclose(view.data['pupil_xy_mm'], original['pupil_xy_mm']*4/3)
            self.assertIn('R = 4 mm', view.figure.axes[2].get_title())
            view.source_x.value = .5
            view.source_y.value = .2
            assert_allclose(view.data['q']-original['q'], np.tile([.2, .2], (9, 1)), atol=1e-14)
            before = view.data
            view.pitch_x.value = 1000
            assert_allclose(view.data['q'][:, 1], before['q'][:, 1])
            assert_allclose(view.data['q'][:, 0],
                            .5+view.data['labels'][:, 0]*view.wavelength0/(1.*view.NA_mask))
            view.pitch_y.value = 1200
            assert_allclose(view.data['q'][:, 1],
                            .2+view.data['labels'][:, 1]*view.wavelength0/(1.2*view.NA_mask))
            self.assertEqual(len(view.figure.axes), 3)
            self.assertEqual(set(_instances), population)
            view.on_axis.click()
            assert_allclose(view.data['source_sigma'], [0, 0])
            view.reset.click()
            assert_allclose(view.data['source_sigma'], [.3, 0])
            self.assertEqual(view.pitch_x.value, 800)
            self.assertEqual(view.pitch_y.value, 1000)
            self.assertEqual(view.pupil_radius.value, 3)
        finally:
            view.close()

    def test_extreme_controls_keep_all_displayed_candidates_propagating(self):
        view = SigmaPupilExplorer()
        try:
            view.source_x.value = .8
            view.source_y.value = -.8
            view.pitch_x.value = 500
            view.pitch_y.value = 500
            assert_array_equal(view.data['status'], np.full(9, 'propagating'))
            points = view.data['q']
            low, high = view.figure.axes[1].get_xlim()
            self.assertTrue(np.all((points > low) & (points < high)))
        finally:
            view.close()

    def test_close_releases_its_resources_and_leaves_another_view_live(self):
        from ipywidgets.widgets.widget import _instances
        from matplotlib._pylab_helpers import Gcf
        baseline, managers = set(_instances), set(Gcf.figs)
        other = SigmaPupilExplorer()
        live_population = set(_instances)
        view = SigmaPupilExplorer()
        reference = weakref.ref(view.figure)
        try:
            controls = view._controls
            view.close()
            view.close()
            for control in controls:
                self.assertNotIn(view._update, control._trait_notifiers['value']['change'])
            self.assertEqual(set(_instances), live_population)
            view.source_x.value = -.5
            self.assertIsNone(view.data)
            self.assertIsNone(view.figure)
            before = bytes(other.picture.value)
            other.source_y.value = .3
            self.assertNotEqual(bytes(other.picture.value), before)
            gc.collect()
            self.assertIsNone(reference())
            self.assertEqual(set(Gcf.figs), managers)
        finally:
            view.close()
            other.close()
        self.assertEqual(set(_instances), baseline)


if __name__ == '__main__':
    unittest.main()
