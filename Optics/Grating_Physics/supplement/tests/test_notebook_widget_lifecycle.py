"""Exercise notebook-owned panels, including reruns outside the setup cell.

Only PNG serialization is stubbed; actual notebook code, widget trees, observers,
Matplotlib artists and numerical arrays participate in these lifecycle checks.
Fresh-kernel certification separately exercises real PNG rendering.
"""

from collections import Counter
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import gc
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import ipywidgets as widgets
from ipywidgets.widgets.widget import _instances
from numpy.testing import assert_equal

from wave_optics import scalar_propagation_view, scalar_theory_view
from wave_optics.grating_intro import _close_widget_tree


ROOT = Path(__file__).resolve().parents[2]


class NotebookWidgetLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.display = self.stack.enter_context(patch('IPython.display.display'))
        self.renderers = {
            module: self.stack.enter_context(patch.object(module, 'figure_png', return_value=b''))
            for module in (scalar_theory_view, scalar_propagation_view)
        }
        self.stack.enter_context(patch.object(scalar_propagation_view, 'slit_setup_png', return_value=b''))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

        # An unrelated nested view must survive every notebook cleanup path.
        self.external_control = widgets.FloatSlider()
        self.external_callback = Mock()
        self.external_control.observe(self.external_callback, names='value')
        self.external = widgets.VBox([widgets.HBox([self.external_control])])
        self.addCleanup(_close_widget_tree, self.external)
        self.preexisting = set(_instances)
        self.namespace = {}
        self.addCleanup(self.dispose_notebook)

    def dispose_notebook(self):
        for name in ('scalar_theory_lessons', 'propagation_lessons'):
            for view in self.namespace.get(name, {}).values():
                view.close()  # Explorer-owned observers are detached first.
        for name in ('superposition_check_panel', 'slit_check_panel'):
            _close_widget_tree(self.namespace.get(name))
        for panel in self.namespace.get('propagation_panels', {}).values():
            _close_widget_tree(panel)

    def cells(self, number):
        path = next(ROOT.glob(f'0_{number}_*.ipynb'))
        book = json.loads(path.read_text(encoding='utf-8'))
        return {c['id']: ''.join(c['source']) for c in book['cells'] if c['cell_type'] == 'code'}

    def run_cells(self, cells):
        self.display.reset_mock()
        for cell_id, source in cells.items():
            exec(compile(source, cell_id, 'exec'), self.namespace)
        gc.collect()

    def owned_models(self):
        return set(_instances) - self.preexisting

    def check_active_controls(self, view, module):
        for control in view._controls:
            callbacks = control._trait_notifiers.get('value', {}).get('change', [])
            self.assertEqual(callbacks.count(view._update), 1)
        self.assertEqual(view.reset_button._click_handlers.callbacks.count(view._reset), 1)
        before = deepcopy(view.data)
        figure, axes_count = view.figure, len(view.figure.axes)
        renderer = self.renderers[module]
        control = view._controls[-1]
        renderer.reset_mock()
        if isinstance(control, widgets.Checkbox):
            control.value = not control.value
        else:
            control.value = control.min if control.value == control.max else control.value + control.step
        self.assertEqual(renderer.call_count, 1, 'A control change must render exactly once')
        self.assertIs(view.figure, figure)
        self.assertEqual(len(view.figure.axes), axes_count)
        self.assertIsNotNone(view.data)
        renderer.reset_mock()
        view.reset_button.click()
        self.assertEqual(renderer.call_count, 1, 'Reset must render exactly once')
        self.assertTrue(all(c.value == default for c, default in view._defaults))
        assert_equal(view.data, before)

    def check_disposed_view(self, view, module):
        self.assertTrue(view._closed)
        self.assertIsNone(view.figure)
        self.assertIsNone(view.data)
        for control in view._controls:
            self.assertNotIn(view._update, control._trait_notifiers.get('value', {}).get('change', []))
        self.assertNotIn(view._reset, view.reset_button._click_handlers.callbacks)
        renderer = self.renderers[module]
        renderer.reset_mock()
        control = view._controls[-1]
        control.value = not control.value if isinstance(control, widgets.Checkbox) else control.min
        view.reset_button.click()
        renderer.assert_not_called()

    def check_full_reruns(self, number):
        cells = self.cells(number)
        module = scalar_theory_view if number == 3 else scalar_propagation_view
        key = 'scalar_theory_lessons' if number == 3 else 'propagation_lessons'
        previous, old_views, initial_data, initial_counts = set(), [], None, None
        for repeat in range(3):
            self.run_cells(cells)
            current = self.owned_models()
            self.assertFalse(current & previous, 'Old notebook widget models survived a rerun')
            counts = Counter(type(_instances[k]).__name__ for k in current)
            views = self.namespace[key]
            data = {name: view.data for name, view in views.items()}
            if repeat == 0:
                initial_data, initial_counts = deepcopy(data), counts
            else:
                self.assertEqual(counts, initial_counts, 'Widget or Layout counts grew')
                assert_equal(data, initial_data)
            # Five display publications per run: no extra figure publication by callbacks.
            self.assertEqual(self.display.call_count, 5)
            for view in old_views:
                self.check_disposed_view(view, module)
            for view in views.values():
                self.check_active_controls(view, module)
            self.assertEqual(self.display.call_count, 5)
            self.assertTrue(self.preexisting <= set(_instances))
            previous, old_views = current, list(views.values())
        self.dispose_notebook()
        self.dispose_notebook()  # Complete disposal is idempotent.
        gc.collect()
        self.assertEqual(set(_instances), self.preexisting)
        self.external_control.value = 1
        self.external_callback.assert_called_once()

    def test_0_3_three_runs_release_panels_layouts_and_observers(self):
        self.check_full_reruns(3)

    def test_0_4_three_runs_release_panels_layouts_and_observers(self):
        self.check_full_reruns(4)

    def test_0_3_optional_cells_can_be_rerun_without_setup(self):
        cells = self.cells(3)
        self.run_cells(cells)
        view_models = self.owned_models()
        for cell_id, panel_name in (('foundation-3-4', 'superposition_check_panel'),
                                   ('foundation-3-17', 'slit_check_panel')):
            for _ in range(3):
                panel = self.namespace[panel_name]
                child = panel.children[0]
                old = {w.model_id for w in (panel, panel.layout, child, child.layout)}
                self.run_cells({cell_id: cells[cell_id]})
                self.assertFalse(old & set(_instances))
                self.assertEqual(len(self.owned_models()), len(view_models))
                self.assertEqual(self.display.call_count, 1)
                self.assertTrue(self.preexisting <= set(_instances))

    def test_0_4_optional_cell_can_be_rerun_without_setup(self):
        cells = self.cells(4)
        self.run_cells(cells)
        count = len(self.owned_models())
        untouched = {name: view for name, view in self.namespace['propagation_lessons'].items()
                     if name != 'transfer'}
        for _ in range(3):
            panel = self.namespace['propagation_panels']['transfer']
            old_layout = panel.layout.model_id
            old_view = self.namespace['propagation_lessons']['transfer']
            self.run_cells({'optional-methods-view': cells['optional-methods-view']})
            self.assertNotIn(old_layout, set(_instances))
            self.assertEqual(len(self.owned_models()), count)
            self.check_disposed_view(old_view, scalar_propagation_view)
            self.check_active_controls(self.namespace['propagation_lessons']['transfer'], scalar_propagation_view)
            self.assertEqual(self.display.call_count, 1)
            for name, view in untouched.items():
                self.assertIs(self.namespace['propagation_lessons'][name], view)
                self.assertFalse(view._closed)
            self.assertTrue(self.preexisting <= set(_instances))


if __name__ == '__main__':
    unittest.main()
