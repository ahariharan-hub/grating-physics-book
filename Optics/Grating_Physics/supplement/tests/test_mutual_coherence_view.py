"""Callbacks, axes, cached geometry and notebook-owned lifecycle, not browser clicks."""
from contextlib import redirect_stdout
import gc
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import ipywidgets as widgets
from ipywidgets.widgets.widget import _instances
import numpy as np
from numpy.testing import assert_allclose

from wave_optics import mutual_coherence_view as view
from wave_optics.grating_intro import _close_widget_tree


ROOT = Path(__file__).resolve().parents[2]


class MutualCoherenceViewTests(unittest.TestCase):
    def test_each_control_and_reset_render_once_without_changing_artists_or_axes(self):
        with patch.object(view, 'figure_png', return_value=b'png') as render:
            for stage in view.STAGES:
                explorer = view.MutualCoherenceExplorer(stage)
                limits = [(a.get_xlim(), a.get_ylim()) for a in explorer.axes]
                original = explorer.figure
                try:
                    for control in explorer._controls:
                        self.assertEqual(control._trait_notifiers['value']['change'].count(explorer._update), 1)
                        render.reset_mock()
                        control.value = (control.min if control.value == control.max else control.max) if isinstance(control, widgets.FloatSlider) else control.options[-1][1]
                        self.assertEqual(render.call_count, 1)
                    render.reset_mock()
                    explorer.reset_button.click()
                    self.assertEqual(render.call_count, 1)
                    self.assertTrue(all(c.value == d for c, d in explorer._defaults))
                    self.assertIs(explorer.figure, original)
                    self.assertEqual(limits, [(a.get_xlim(), a.get_ylim()) for a in explorer.axes])
                finally:
                    explorer.close()
                    explorer.close()
                for control in explorer._controls:
                    self.assertNotIn(explorer._update, control._trait_notifiers.get('value', {}).get('change', []))
                self.assertNotIn(explorer._reset, explorer.reset_button._click_handlers.callbacks)
                self.assertIsNone(explorer.figure)
                self.assertIsNone(explorer.data)
                self.assertIsNone(explorer.ui.comm)

    def test_order_phase_reuses_basis_and_exposure_state_preserves_average(self):
        with patch.object(view, 'figure_png', return_value=b'png'):
            explorer = view.MutualCoherenceExplorer('orders')
            basis = explorer.basis
            try:
                with patch.object(view.model, 'two_order_basis', side_effect=AssertionError('geometry recalculated')):
                    explorer.phase.value = 95
                    explorer.ratio.value = .5
                    explorer.magnitude.value = .5
                self.assertIs(explorer.basis, basis)
            finally:
                explorer.close()
            exposure = view.MutualCoherenceExplorer('exposure')
            try:
                expected = exposure.data['intensity'].copy()
                exposure.state.value = 1
                assert_allclose(exposure.data['intensity'], expected)
                assert_allclose(exposure.data['gamma'], 0, atol=1e-15)
                self.assertIsNone(exposure.data['coherence_phase_deg'])
                exposure.duty.value = 1
                assert_allclose(abs(exposure.data['gamma']), 1)
            finally:
                exposure.close()

    def test_real_render_and_undefined_arm_readout(self):
        explorer = view.MutualCoherenceExplorer('visibility')
        try:
            self.assertTrue(bytes(explorer.picture.value).startswith(b'\x89PNG'))
            explorer.ratio.value = 0
            self.assertIsNone(explorer.data['gamma'])
            self.assertIn('undefined', explorer.readout.value)
            self.assertEqual(explorer.data['visibility'], 0)
        finally:
            explorer.close()

    def test_notebook_cell_reruns_release_models_and_preserve_unrelated_widgets(self):
        book = json.loads((ROOT/'7_1_Interference_and_Mutual_Coherence_of_Two_Plane_Waves.ipynb').read_text(encoding='utf8'))
        cells = {c['id']: ''.join(c['source']) for c in book['cells'] if c['cell_type'] == 'code'}
        namespace = {}
        unrelated = widgets.VBox([widgets.FloatSlider()])
        before = set(_instances)
        try:
            with patch.object(view, 'figure_png', return_value=b''), patch('IPython.display.display'), redirect_stdout(io.StringIO()):
                counts = []
                for _ in range(3):
                    for key in ('n71-setup', 'n71-apparatus', 'n71-visibility', 'n71-eo-view'):
                        exec(cells[key], namespace)
                    gc.collect()
                    counts.append(len(set(_instances)-before))
                    old = namespace['interference_lessons']['phase']
                    old_models = {old.ui.model_id, old.phase.model_id}
                    exec(cells['n71-eo-view'], namespace)
                    self.assertTrue(old._closed)
                    self.assertFalse(old_models & set(_instances))
                    self.assertTrue(before <= set(_instances))
                self.assertEqual(counts, [counts[0]]*3)
        finally:
            for lesson in namespace.get('interference_lessons', {}).values():
                lesson.close()
            gc.collect()
            self.assertEqual(set(_instances), before)
            _close_widget_tree(unrelated)

    def test_optional_apparatus_is_lazy_uses_existing_model_and_closes(self):
        background = view.ApparatusBackground()
        self.assertIsNone(background.apparatus)
        try:
            background.ui.selected_index = 0
            apparatus = background.apparatus
            self.assertTrue(bytes(apparatus.picture.value).startswith(b'\x89PNG'))
            apparatus.eo_phase.value = 90
            apparatus.tilt.value = 2
            apparatus.detector_z.value = -9.15
            self.assertTrue(bytes(apparatus.modulated_picture.value).startswith(b'\x89PNG'))
            background.ui.selected_index = None
            background.ui.selected_index = 0
            self.assertIs(background.apparatus, apparatus)
        finally:
            background.close()
            background.close()
        self.assertTrue(apparatus._closed)
        self.assertIsNone(background.ui.comm)


if __name__ == '__main__':
    unittest.main()
