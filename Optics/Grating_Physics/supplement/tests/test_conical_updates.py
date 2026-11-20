"""Regression checks for selective Notebook 5 updates and retained view state."""

import contextlib
import gc
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from wave_optics import conical_plotting, conical_view


def roles(figure):
    return {trace.meta['role']: trace for trace in figure.data
            if isinstance(trace.meta, dict) and 'role' in trace.meta}


class SelectiveConicalUpdateTests(unittest.TestCase):
    def setUp(self):
        self.explorer = conical_view.ConicalDiffractionExplorer()
        self.explorer.set_example(.4)

    def tearDown(self):
        self.explorer.close()
        self.explorer = None
        gc.collect()

    def open_views(self):
        for key in ('incidence', 'orders', 'harvey', 'ewald'):
            getattr(self.explorer, key+'_ui')

    def spy_builders(self, stack):
        return {name: stack.enter_context(patch.object(conical_view, name,
                    wraps=getattr(conical_view, name))) for name in
                ('conical_orders', 'incidence_figure', 'spatial_figure', 'representation_figures')}

    def test_first_open_builds_only_the_requested_representation(self):
        with patch.object(conical_plotting, 'geometry_figures',
                          wraps=conical_plotting.geometry_figures) as build:
            first = self.explorer.harvey_ui
            self.assertIs(first, self.explorer.harvey_ui)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(build.call_args.kwargs['views'], ('direction',))
            build.reset_mock()
            self.explorer.ewald_ui
            self.assertEqual(build.call_count, 1)
            self.assertEqual(build.call_args.kwargs['views'], ('ewald',))
        self.assertEqual(set(self.explorer._figures), {'harvey', 'ewald'})

    def test_display_controls_do_not_recalculate_physics_or_unrelated_views(self):
        self.open_views()
        explorer = self.explorer
        data = explorer.data
        unchanged = {key: figure.to_json() for key, figure in explorer._figures.items()
                     if key != 'orders'}
        with contextlib.ExitStack() as stack:
            builders = self.spy_builders(stack)
            explorer.detector_distance.value = 4.
            explorer.cone.value = True
            explorer.detector.value = False
            self.assertTrue(all(mock.call_count == 0 for mock in builders.values()))
        self.assertIs(explorer.data, data)
        for key, previous in unchanged.items():
            self.assertEqual(explorer._figures[key].to_json(), previous)
        spatial = roles(explorer._figures['orders'])
        assert_allclose(spatial['detector-plane'].z, -4.)
        self.assertFalse(spatial['detector-plane'].visible)
        self.assertEqual(len(spatial['detector-hits'].x), 0)
        self.assertTrue(spatial['cone-surface'].visible)

        unchanged = {key: figure.to_json() for key, figure in explorer._figures.items()
                     if key != 'harvey'}
        with contextlib.ExitStack() as stack:
            builders = self.spy_builders(stack)
            explorer.coordinate_scale.value = 'frequency'
            self.assertEqual(builders['representation_figures'].call_count, 1)
            self.assertEqual(builders['representation_figures'].call_args.kwargs['views'], ('harvey',))
            for name in ('conical_orders', 'incidence_figure', 'spatial_figure'):
                self.assertEqual(builders[name].call_count, 0)
        for key, previous in unchanged.items():
            self.assertEqual(explorer._figures[key].to_json(), previous)

    def test_selection_reuses_figures_through_zero_and_evanescent_orders(self):
        self.open_views()
        explorer = self.explorer
        explorer.coordinate_scale.value = 'frequency'
        incidence = explorer._figures['incidence'].to_json()
        originals = {key: tuple(id(trace) for trace in figure.data)
                     for key, figure in explorer._figures.items()}
        with contextlib.ExitStack() as stack:
            builders = self.spy_builders(stack)
            for order in (0, 2, -2, 1):
                explorer.order.value = order
                data = explorer.data
                index = data['selected_index']
                selected = data['components'][index]
                real = bool(data['valid'][index])
                h = roles(explorer._figures['harvey'])
                e = roles(explorer._figures['ewald'])
                s = roles(explorer._figures['orders'])
                assert_allclose([h['selected-disk'].x[0], h['selected-disk'].y[0]],
                                selected[:2].real/data['wavelength'])
                self.assertEqual(h['selected-direction'].visible, real)
                self.assertEqual(e['outgoing-vector'].visible, real)
                assert_allclose(e['selected-rod'].x, order*data['spacing'])
                total = s['selected-wavevector']
                assert_allclose([total.x[-1], total.y[-1], total.z[-1]], data['k']*selected.real)
                self.assertEqual(s['order-table'].cells.fill.color[0][index], '#fff0d8')
                if real:
                    q = selected.real-data['incident']
                    assert_allclose([e['selected-ewald'].x[0], e['selected-ewald'].y[0],
                                     e['selected-ewald'].z[0]], q)
                if order == 0:
                    self.assertEqual(len(e['change-head'].x), 0)
            for name in ('incidence_figure', 'spatial_figure', 'representation_figures'):
                self.assertEqual(builders[name].call_count, 0)
        self.assertEqual(explorer._figures['incidence'].to_json(), incidence)
        for key, identities in originals.items():
            self.assertEqual(tuple(id(trace) for trace in explorer._figures[key].data), identities)

    def test_retained_far_candidate_can_leave_without_rebuilding_incidence(self):
        self.open_views()
        explorer = self.explorer
        explorer.wavelength.value = .5
        explorer.period.value = 6.
        explorer.order.value = -10
        explorer.wavelength.value = 4.
        explorer.period.value = 1.
        self.assertIn(-10, explorer.data['labels'])
        incidence = explorer._figures['incidence'].to_json()
        with contextlib.ExitStack() as stack:
            builders = self.spy_builders(stack)
            explorer.order.value = 0
            self.assertEqual(builders['incidence_figure'].call_count, 0)
            self.assertEqual(builders['spatial_figure'].call_count, 1)
            self.assertEqual(builders['representation_figures'].call_count, 1)
        self.assertNotIn(-10, explorer.data['labels'])
        self.assertEqual(explorer._figures['incidence'].to_json(), incidence)

    def test_physics_updates_preserve_cameras_zoom_and_unchanged_nan_mask(self):
        self.open_views()
        explorer = self.explorer
        camera = dict(eye=dict(x=-1.1, y=1.8, z=.7))
        for figure in explorer._figures.values():
            figure.layout.scene.camera = camera
        spatial = explorer._figures['orders']
        spatial.layout.scene2.camera = camera
        spatial.layout.scene.xaxis.range = [-2., 2.]
        incidence = explorer._figures['incidence']
        incidence.layout.xaxis.range = [-1., 1.]
        surface = roles(incidence)['incident-opening-field']
        original_mask = np.asarray(surface.z).copy()
        with patch.object(surface, 'update', wraps=surface.update) as update:
            explorer.alpha.value = .32
            self.assertGreater(update.call_count, 0)
            for call in update.call_args_list:
                self.assertNotIn('z', call.args[0] if call.args else call.kwargs)
        assert_allclose(surface.z, original_mask, equal_nan=True)
        self.assertEqual(tuple(spatial.layout.scene.xaxis.range), (-2., 2.))
        self.assertEqual(tuple(incidence.layout.xaxis.range), (-1., 1.))
        explorer.order.value = 0
        explorer.detector_distance.value = 5.
        explorer.coordinate_scale.value = 'wavevector'
        for figure in explorer._figures.values():
            self.assertEqual(figure.layout.scene.camera.to_plotly_json(), camera)
        self.assertEqual(spatial.layout.scene2.camera.to_plotly_json(), camera)


if __name__ == '__main__':
    unittest.main()
