"""Two-plane geometry, approximation meaning, shared controls and reruns."""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import pytest
from ipywidgets.widgets.widget import _instances
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.testing import assert_allclose
from PIL import Image
from traitlets import TraitError

from wave_optics import propagation_geometry as geometry
from wave_optics.propagation_kernels import kernel_state, kernel_explorer


def test_every_stage_keeps_the_same_two_planes_and_selected_points():
    for s, x, q in [(1., 3., -3.), (4., -1., 2.), (100., 0., 0.)]:
        data = kernel_state(s, x, q)
        expected = None
        for stage in geometry.STAGES:
            fig = geometry.geometry_figure(data, stage)
            try:
                FigureCanvasAgg(fig).draw()
                ax = fig.axes[0]
                planes = [p.get_xy() for p in ax.patches[:2]]
                if expected is None:
                    expected = planes
                for actual, reference in zip(planes, expected):
                    assert_allclose(actual, reference)
                # Opposite y corners project symmetrically: center separation is s.
                a, b = [np.mean(p[:4], axis=0) for p in planes]
                assert_allclose(b-a, [s, 0.], atol=1e-14)
                for horizontal, transverse in [(0., q), (s, x)]:
                    assert ax.get_xlim()[0] < horizontal < ax.get_xlim()[1]
                    assert ax.get_ylim()[0] < transverse < ax.get_ylim()[1]
            finally:
                fig.clear()


def test_geometry_profiles_show_the_actual_path_and_omitted_phase():
    data = kernel_state(3., .7, 1.3)
    fig = geometry.geometry_figure(data, 'fresnel')
    try:
        exact, quadratic = [line.get_ydata() for line in fig.axes[1].lines[:2]]
        assert_allclose(exact, np.sqrt(3.**2+(.7-data['xi'])**2)-3., atol=2e-15)
        # Markers are interleaved with the curves only after both curves.
        assert_allclose(quadratic, (.7-data['xi'])**2/(2*3.))
        assert np.all(quadratic >= exact-1e-14)
        full_magnitude, fresnel_magnitude = [line.get_ydata() for line in fig.axes[2].lines[:2]]
        assert_allclose(full_magnitude, abs(data['profile']['kernels'][0]))
        assert_allclose(fresnel_magnitude, 1/3., atol=2e-16)
        assert np.ptp(full_magnitude) > .1
        assert_allclose(fig.axes[2].get_ylim(), [0., 1.15])
    finally:
        fig.clear()
    fig = geometry.geometry_figure(data, 'fraunhofer')
    try:
        f, ff = [line.get_ydata() for line in fig.axes[1].lines[:2]]
        assert_allclose(f-ff, np.pi*data['xi']**2/3., atol=2e-14)
        assert_allclose(f, data['profile']['phase'][1]-np.pi*.7**2/3., atol=2e-14)
        assert_allclose(ff, data['profile']['phase'][2]-np.pi*.7**2/3., atol=2e-14)
    finally:
        fig.clear()


def test_linked_controls_update_all_panels_and_close_only_their_own_widgets():
    baseline = set(_instances)
    state = geometry.GeometryState(distance=7., observation=.4, boundary=-1.2)
    panels = []
    try:
        panels = [geometry.geometry_explorer(stage, state) for stage in geometry.STAGES]
        panels.append(kernel_explorer(state))
        for panel in panels:
            for name in geometry.GEOMETRY_NAMES:
                assert panel.controls[name].value == getattr(state, name)
        for index, name, value in [(2, 'distance', 12.), (0, 'observation', 2.), (3, 'boundary', -1.)]:
            old = [bytes(p.image.value) for p in panels]
            panels[index].controls[name].value = value
            for panel, previous in zip(panels, old):
                assert panel.controls[name].value == getattr(state, name) == value
                assert bytes(panel.image.value) != previous
                Image.open(BytesIO(panel.image.value)).verify()
        panels[0].close()
        state.observation = 1.
        assert panels[0]._closed and panels[0].data is None
        assert all(not getattr(p, '_closed', False) and p.controls['observation'].value == 1. for p in panels[1:])
    finally:
        for panel in panels:
            panel.close(); panel.close()
    assert set(_instances) == baseline
    assert all(not state._trait_notifiers[name].get('change') for name in geometry.GEOMETRY_NAMES)


def test_notebook_panels_rerun_with_shared_geometry_without_closing_siblings():
    root = Path(__file__).resolve().parents[2]
    book = nbformat.read(root/'8_3_ASM_Fresnel_and_Fraunhofer_Propagation.ipynb', 4)
    # Notebook frontends may regenerate cell IDs; select the actual calls.
    cells = [c for c in book.cells if c.cell_type == 'code' and
             ('propagation_geometry.geometry_explorer(' in c.source or
              'propagation_kernels.kernel_explorer(' in c.source)]
    names = ['_planes_panel', '_full_geometry_panel', '_fresnel_geometry_panel',
             '_fraunhofer_geometry_panel', '_comparison_panel']
    baseline = set(_instances); ns = {}
    try:
        with patch('IPython.display.display'):
            for cell in cells:
                exec(cell.source, ns)
            count = len(_instances)
            ns['_kernel_geometry_state'].boundary = -.7
            for cell, name in zip(cells, names):
                old = ns[name]
                exec(cell.source, ns)
                assert old._closed and len(_instances) == count
                assert ns[name].controls['boundary'].value == -.7
                assert all(not getattr(ns[n], '_closed', False) for n in names)
    finally:
        for name in names:
            if name in ns:
                ns[name].close()
    assert set(_instances) == baseline


def test_failed_geometry_render_releases_widgets_before_state_binding():
    baseline = set(_instances); state = geometry.GeometryState()
    with patch.object(geometry, 'geometry_figure', side_effect=RuntimeError('render failed')):
        with pytest.raises(RuntimeError):
            geometry.geometry_explorer('planes', state)
    assert set(_instances) == baseline
    assert all(not state._trait_notifiers.get(name, {}).get('change') for name in geometry.GEOMETRY_NAMES)


def test_geometry_state_rejects_invalid_ranges():
    for name, value in [('distance', 0.), ('distance', np.inf), ('observation', 4.), ('boundary', np.nan)]:
        with pytest.raises(TraitError):
            geometry.GeometryState(**{name:value})
