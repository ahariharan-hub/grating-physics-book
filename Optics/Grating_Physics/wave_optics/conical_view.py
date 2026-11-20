"""Linked views for Notebook 5 and the standalone Ewald lesson."""

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go

from .conical import conical_orders
from .conical_plotting import (incidence_figure, spatial_figure,
                               representation_figures, selection_readout, detector_readout,
                               direction_scale_readout, update_spatial_controls,
                               update_spatial_selection, update_representation_selection)
from .grating_intro import _close_widget_tree


def _equal_values(left, right):
    """Compare generated properties, treating unchanged masked samples as equal."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(left, right, equal_nan=True)
        except TypeError:
            return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal_values(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal_values(a, b) for a, b in zip(left, right))
    return left == right


def _changed_properties(previous, current):
    """A sparse update; absent properties and interactive cameras stay untouched."""
    changed = {}
    for key, value in current.items():
        if key == 'camera':
            continue
        old = previous.get(key)
        if isinstance(value, dict) and isinstance(old, dict):
            nested = _changed_properties(old, value)
            if nested:
                changed[key] = nested
        elif key not in previous or not _equal_values(old, value):
            changed[key] = value
    return changed


class ConicalDiffractionExplorer:
    """One parameter state shared by incidence, rays, Harvey and Ewald views.

    Figures are created on first display; hidden later lessons are not built.
    Call close() before replacement, then redisplay the new instance's views.
    """
    def __init__(self):
        self._busy = False
        self._closed = False
        self._views = {}
        self._figures = {}
        self._layout_specs = {}
        self._callbacks = []
        style = {'description_width':'105px'}
        def slider(value, low, high, step, label):
            return widgets.FloatSlider(value=value, min=low, max=high, step=step,
                description=label, readout_format='.2f', continuous_update=False,
                style=style, layout=widgets.Layout(width='320px'))
        self.wavelength = slider(2., .5, 4., .1, 'λ (µm)')
        self.period = slider(4., 1., 6., .1, 'd (µm)')
        self.alpha = slider(.3, -.95, .95, .01, 'kᵢₓ / k')
        self.beta = slider(0., -.95, .95, .01, 'kᵢᵧ / k')
        self.order = widgets.Dropdown(options=[1], value=1, description='Order m',
            style=style, layout=widgets.Layout(width='320px'))
        self.cone = widgets.Checkbox(value=False, description='Show cone / planar limit',
            indent=False, layout=widgets.Layout(width='260px'))
        self.detector = widgets.Checkbox(value=True, description='Show detector plane',
            indent=False, layout=widgets.Layout(width='220px'))
        self.detector_distance = slider(3., 1., 6., .25, 'L (µm)')
        self.detector_summary = widgets.HTML()
        self.spatial_controls = widgets.HBox([self.cone, self.detector, self.detector_distance],
            layout=widgets.Layout(flex_flow='row wrap'))
        self.notice = widgets.HTML()
        self.summary = widgets.HTML()
        self.incident_summary = widgets.HTML()
        self.ewald_summary = widgets.HTML()
        self.coordinate_scale = widgets.Dropdown(
            options=[('Direction cosines', 'direction'), ('Spatial frequency (Goodman)', 'frequency'),
                     ('Wavevector', 'wavevector')], value='direction',
            description='Coordinates', style=style, layout=widgets.Layout(width='320px'))
        self.coordinate_summary = widgets.HTML()
        self.normal = widgets.Button(description='Normal incidence', layout=widgets.Layout(width='160px'))
        self.reset = widgets.Button(description='Reset: in-plane', layout=widgets.Layout(width='160px'))
        self.skew = widgets.Button(description='Conical: β = 0.4', layout=widgets.Layout(width='160px'))
        self.cutoff = widgets.Button(description='Grazing: β = 0.6', layout=widgets.Layout(width='160px'))
        self.beyond = widgets.Button(description='Beyond: β = 0.8', layout=widgets.Layout(width='160px'))
        for button, alpha, beta in ((self.normal, 0., 0.), (self.reset, .3, 0.),
                                    (self.skew, .3, .4), (self.cutoff, .3, .6),
                                    (self.beyond, .3, .8)):
            callback = lambda _, alpha=alpha, beta=beta: self.set_example(beta, alpha=alpha)
            button.on_click(callback)
            self._callbacks.append((button, callback))
        parameters = [
            widgets.HBox([self.wavelength, self.period], layout=widgets.Layout(flex_flow='row wrap')),
            widgets.HBox([self.alpha, self.beta], layout=widgets.Layout(flex_flow='row wrap')),
            self.notice]
        self.incident_controls = widgets.VBox([*parameters,
            widgets.HBox([self.normal, self.reset, self.skew],
                         layout=widgets.Layout(flex_flow='row wrap'))])
        self.controls = widgets.VBox([*parameters,
            widgets.HBox([self.reset, self.skew, self.cutoff, self.beyond],
                         layout=widgets.Layout(flex_flow='row wrap'))])
        self._owned = widgets.VBox([self.controls, self.incident_controls, self.order, self.cone,
                                    self.summary, self.incident_summary, self.ewald_summary,
                                    self.spatial_controls, self.detector_summary,
                                    self.coordinate_scale, self.coordinate_summary])
        self._inputs = (self.wavelength, self.period, self.alpha, self.beta, self.order, self.cone,
                        self.detector, self.detector_distance, self.coordinate_scale)
        for control in self._inputs:
            control.observe(self._update, names='value')
        self._update()

    def set_example(self, beta, *, alpha=.3):
        """Set the written example exactly; each preset selects m=+1."""
        if self._closed:
            return
        self._busy = True
        try:
            self.wavelength.value, self.period.value = 2., 4.
            self.alpha.value = alpha
            self.beta.min, self.beta.max = -.95, .95
            self.beta.value = beta
            if 1 not in self.order.options:
                self.order.options = sorted(set(self.order.options) | {1})
            self.order.value = 1
            if beta == 0:
                self.cone.value = False
        finally:
            self._busy = False
        self._update()

    def _update(self, change=None):
        if self._busy or self._closed:
            return
        self._busy = True
        try:
            owner = change['owner'] if change is not None else None
            if owner in (self.cone, self.detector, self.detector_distance):
                if owner is not self.cone:
                    self.detector_summary.value = detector_readout(
                        self.data, self.detector_distance.value, self.detector.value)
                if 'orders' in self._figures:
                    live = self._figures['orders']
                    with live.batch_update():
                        update_spatial_controls(live, self.data, show_cone=self.cone.value,
                            show_detector=self.detector.value, detector_distance=self.detector_distance.value,
                            cone_only=owner is self.cone)
                return
            if owner is self.coordinate_scale:
                self.coordinate_summary.value = direction_scale_readout(self.data, self.coordinate_scale.value)
                if 'harvey' in self._figures:
                    fresh, = representation_figures(self.data, self.coordinate_scale.value, views=('harvey',))
                    self._replace_figure('harvey', fresh)
                return
            selection_only = owner is self.order
            previous_labels = self.data['labels'] if hasattr(self, 'data') else None
            # Keep incidence physical. The alpha control has priority; explicitly
            # report any necessary change of beta rather than showing stale data.
            if not selection_only:
                bound = np.floor(100*np.sqrt(1-self.alpha.value**2)-1e-8)/100
                old_beta = self.beta.value
                beta = float(np.clip(old_beta, -bound, bound))
                self.beta.min, self.beta.max = -bound, bound
                self.beta.value = beta
                self.notice.value = (f'Along-groove component adjusted to {beta:+.2f} '
                    'so kᵢₓ² + kᵢᵧ² < k².' if beta != old_beta else '')
            chosen = self.order.value
            self.data = conical_orders(self.wavelength.value, self.period.value,
                                       self.alpha.value, self.beta.value, selected_order=chosen)
            options = tuple(int(m) for m in self.data['labels'])
            if self.order.options != options:
                self.order.options = options
            self.order.value = chosen
            self.summary.value = selection_readout(self.data)
            self.coordinate_summary.value = direction_scale_readout(
                self.data, self.coordinate_scale.value)
            self.detector_summary.value = detector_readout(
                self.data, self.detector_distance.value, self.detector.value)
            if not selection_only:
                self._set_incident_summary()
            self._set_ewald_summary()
            if selection_only and np.array_equal(previous_labels, self.data['labels']):
                for key, live in self._figures.items():
                    if key == 'incidence':
                        continue
                    with live.batch_update():
                        if key == 'orders':
                            update_spatial_selection(live, self.data, show_detector=self.detector.value,
                                                     detector_distance=self.detector_distance.value)
                        else:
                            update_representation_selection(live, self.data, key, self.coordinate_scale.value)
                    if key == 'orders':
                        self._remember_component_layout()
                return
            requested = tuple(key for key in ('harvey', 'ewald') if key in self._figures)
            representations = dict(zip(requested, representation_figures(
                self.data, self.coordinate_scale.value, views=requested))) if requested else {}
            for key in self._figures:
                if key == 'incidence':
                    if selection_only:
                        continue
                    fresh = incidence_figure(self.data)
                elif key == 'orders':
                    fresh = self._spatial_figure()
                else:
                    fresh = representations[key]
                self._replace_figure(key, fresh)
        finally:
            self._busy = False

    def _replace_figure(self, key, fresh):
        """Retain live traces and send only changed generated properties."""
        live = self._figures[key]
        layout = fresh.layout.to_plotly_json()
        layout_changes = _changed_properties(self._layout_specs[key], layout)
        with live.batch_update():
            for old, new in zip(live.data, fresh.data, strict=True):
                changes = _changed_properties(old.to_plotly_json(), new.to_plotly_json())
                if changes:
                    old.update(changes)
            if layout_changes:
                live.update_layout(layout_changes)
        self._layout_specs[key] = layout

    def _remember_component_layout(self):
        # Selection intentionally changes only the component frame and its title.
        # Keep the generated baseline separate from any user zoom or camera.
        live = self._figures['orders'].layout
        spec = self._layout_specs['orders']
        for axis in ('xaxis', 'yaxis', 'zaxis'):
            spec['scene2'][axis]['range'] = list(live.scene2[axis].range)
        spec['scene2']['zaxis']['title']['text'] = live.scene2.zaxis.title.text
        spec['annotations'][-1]['text'] = live.annotations[-1].text

    def _set_incident_summary(self):
        si = self.data['incident']
        if np.hypot(*si[:2]) < 1e-12:
            lesson = 'Normal incidence: the wavefronts are parallel to the grating; its whole surface has the same incident phase.'
        elif abs(si[1]) < 1e-12:
            lesson = 'Oblique incidence across the grooves: phase changes across openings, but stays constant along each opening.'
        else:
            lesson = 'Conical incidence: the wavefronts also tilt along the grooves; phase now changes along each opening.'
        self.incident_summary.value = (
            f'<b>{lesson}</b><br>'
            f'Incident <b>k</b><sub>i</sub>/k = ({si[0]:+.3f}, {si[1]:+.3f}, {si[2]:+.3f}). '
            'Orange planes are wave crests (uᵢ/A = +1), spaced by λ along <b>k</b><sub>i</sub>; orange lines are their surface traces.<br>'
            'Surface colour: incident field cos(kᵢₓ x + kᵢᵧ y) at t = 0, not intensity. '
            'Red = +1; white = 0; blue = −1, as in the earlier notebooks. '
            'Dotted lines mark opening centres; the 50%-open screen locates the grating.')

    def _spatial_figure(self):
        return spatial_figure(self.data, self.cone.value, self.detector.value,
                              self.detector_distance.value)

    def _set_ewald_summary(self):
        i = self.data['selected_index']
        centre = -self.data['incident']
        text = ('C = −<b>k</b><sub>i</sub>/k = ('+', '.join(f'{v:+.3f}' for v in centre)+'). '
                f'Rod: qₓ/k = {self.data["chosen"]*self.data["spacing"]:+.3f}, qᵧ = 0. ')
        if self.data['valid'][i]:
            q = self.data['components'][i].real-self.data['incident']
            text += '<b>q</b>/k = ('+', '.join(f'{v:+.3f}' for v in q)+'). '
            text += ('At m = 0, <b>q</b> = 0 and the vector triangle collapses.'
                     if self.data['chosen'] == 0 else
                     'C → O is <b>k</b><sub>i</sub>/k; C → Q is <b>k</b><sub>m</sub>/k; O → Q is <b>q</b>/k.')
        else:
            text += 'No real sphere intersection: the outgoing vector triangle is hidden.'
        self.ewald_summary.value = text

    def _view(self, key):
        if self._closed:
            raise RuntimeError('This explorer is closed. Create a new instance.')
        if key not in self._views:
            if key == 'incidence':
                figure = incidence_figure(self.data)
            elif key == 'orders':
                figure = self._spatial_figure()
            else:
                figure, = representation_figures(self.data, self.coordinate_scale.value, views=(key,))
            live = go.FigureWidget(figure)
            self._figures[key] = live
            self._layout_specs[key] = figure.layout.to_plotly_json()
            if key == 'incidence':
                children = [self.incident_controls, self.incident_summary, live]
            else:
                children = [self.controls, self.order, self.summary]
                if key == 'orders':
                    children += [self.spatial_controls, self.detector_summary]
                if key == 'ewald':
                    children += [self.ewald_summary]
                if key == 'harvey':
                    children += [self.coordinate_scale, self.coordinate_summary]
                children += [live]
            self._views[key] = widgets.VBox(children)
            self._owned.children = (*self._owned.children, self._views[key])
        return self._views[key]

    @property
    def incidence_ui(self):
        return self._view('incidence')

    @property
    def orders_ui(self):
        return self._view('orders')

    @property
    def harvey_ui(self):
        return self._view('harvey')

    @property
    def ewald_ui(self):
        return self._view('ewald')

    def close(self):
        if self._closed:
            return
        self._closed = True
        for control in self._inputs:
            control.unobserve(self._update, names='value')
        for button, callback in self._callbacks:
            button.on_click(callback, remove=True)
        self._callbacks.clear()
        _close_widget_tree(self._owned)
        self._figures.clear()
        self._views.clear()
        self._layout_specs.clear()
