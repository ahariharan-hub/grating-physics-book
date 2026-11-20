"""Notebook 7.2A: prescribed order coefficients with existing scalar kinematics.

No scattering solver, finite envelope, propagation algorithm or source ensemble.
Lengths are in micrometres. The grating is z=0; transmission goes into z<0.
"""
from itertools import combinations

import numpy as np

from .diffraction import diffraction_orders, GRAZING_TOLERANCE
from .grating_intro import _close_widget_tree
from .mutual_coherence import fringe_visibility
from .waves import wavenumber
from .diffraction_order_detector_view import TwoOrderDetectorView


def transmitted_orders(wavelength=2., pitch=2., angle_deg=30., azimuth_deg=0.):
    """Adapt existing +z kinematics to the course's signed incidence and -z output.

    Strict transmission excludes the existing helper's grazing directions.
    This is a coordinate adapter, not a new diffraction physics backend.
    """
    angle = float(angle_deg)
    if not np.isfinite(angle) or not -90 < angle < 90:
        raise ValueError('Use a finite signed incidence angle strictly between -90 and 90 degrees.')
    # Signed polar angle retains the original xz convention at azimuth zero.
    phi = float(azimuth_deg)+(0 if angle >= 0 else 180)
    data = diffraction_orders(wavelength, pitch, abs(angle), phi)
    accepted = data['rad'][data['valid']] > GRAZING_TOLERANCE
    labels = data['allowed'][accepted].copy()
    vectors = data['directions'][accepted].copy()*wavenumber(wavelength)
    if abs(np.sin(np.deg2rad(phi))) < 1e-14:
        vectors[:, 1] = 0.  # Suppress roundoff for the exact xz special case.
    vectors[:, 2] *= -1
    return labels, vectors


def selected_vectors(wavelength, pitch, angle_deg, pair, azimuth_deg=0.):
    """Return two transmitted orders with the incident wave's shared ky."""
    values = np.asarray(pair)
    if (values.shape != (2,) or not np.isfinite(values).all()
            or not np.all(values == np.rint(values)) or values[0] == values[1]):
        raise ValueError('Select two distinct integer order labels.')
    labels, vectors = transmitted_orders(wavelength, pitch, angle_deg, azimuth_deg)
    if not all(m in labels for m in values):
        raise ValueError('Both selected orders must propagate strictly into z < 0; grazing is excluded.')
    return np.stack([vectors[np.flatnonzero(labels == m)[0]] for m in values])


class DiffractionOrderInterferenceExplorer:
    """One retained-artist explorer; geometry, coefficients and screen are separate."""

    def __init__(self):
        import ipywidgets as widgets
        self._closed = self._changing = False
        self.updates = self.geometry_evaluations = 0
        self._basis_key = None
        self._controls = []
        def slider(label, value, lower, upper, step):
            control = widgets.FloatSlider(description=label, value=value,
                min=lower, max=upper, step=step, continuous_update=False,
                style={'description_width':'initial'},
                layout=widgets.Layout(width='315px', max_width='100%'))
            self._controls.append(control)
            return control
        self.pitch = slider('Grating pitch d (µm)', 2, 1, 6, .25)
        self.wavelength = slider('Wavelength λ (µm)', 2, 1, 3, .25)
        self.angle = slider('Incidence θᵢ (°)', 30, -60, 60, 5)
        self.azimuth = slider('Incident azimuth φᵢ (°)', 0, -180, 180, 5)
        self.pair = widgets.Dropdown(description='Orders (m, p)',
            options=[], style={'description_width':'initial'},
            layout=widgets.Layout(width='315px', max_width='100%'))
        self.ratio = slider('Amplitude |Cₚ| / |Cₘ|', 1, 0, 2, .1)
        self.phase = slider('Δφ = arg Cₚ − arg Cₘ (°)', 0, -180, 180, 15)
        self.detector = slider('Detector zᴅ (µm)', -3, -8, -.25, .25)
        self._controls.append(self.pair)
        self._geometry_controls = (self.pitch, self.wavelength, self.angle, self.azimuth)
        self.reset_button = widgets.Button(description='Reset', icon='undo')
        self.readout = widgets.HTML()
        self.order_colours=('#2866a3','#b85c17')
        self.detector_view=TwoOrderDetectorView()
        self._default_pair=(-1,0)
        self._refresh_pairs(preferred=self._default_pair)
        for control in self._controls:
            control.observe(self._changed,names='value')
        self.reset_button.on_click(self.reset)
        self.ui=widgets.VBox([
            widgets.HTML('<b>Geometry</b> — only strictly propagating pairs are offered; coefficients are prescribed.'),
            widgets.HBox([self.pitch,self.wavelength,self.angle],layout=widgets.Layout(flex_flow='row wrap')),
            widgets.HBox([self.azimuth],layout=widgets.Layout(flex_flow='row wrap')),
            widgets.HTML('Azimuth is measured from +x toward +y. Signed θᵢ reverses the transverse direction when negative. A 1D grating conserves ky in every order; Δky = 0.'),
            widgets.HBox([self.pair,self.detector,self.reset_button],layout=widgets.Layout(flex_flow='row wrap')),
            widgets.HTML('<b>Prescribed coefficients</b> — |Cₘ| = 1; a phase change changes Cₚ, not source coherence.'),
            widgets.HBox([self.ratio,self.phase],layout=widgets.Layout(flex_flow='row wrap')),
            self.detector_view.figure,self.readout,
            widgets.HTML("<small>Rotate the 3D view by dragging. The XY detector shows the same "
                "extended fields; direction arrows are not beam boundaries. The separate 3D wavevector "
                "construction uses rad/µm and shows Δk = kₚ − kₘ. "
                "Blue and orange sheets mark constant-phase crests of the two outgoing waves. Each direction arrow is normal to its own wavefront; the small corner marks a true 3D right angle (which can look oblique after rotating the view). Their edges only clip the display: these ideal plane waves extend indefinitely in the transmitted half-space. Conical orders share ky, so intensity remains uniform along y. The green dotted line marks y = 0.</small>")])
        self._update()

    @staticmethod
    def _component_number(value):
        """Format shared numerical data; suppress a negative zero only after rounding."""
        return f'{value:+.3f}'.replace('-0.000','+0.000')

    def _readout_html(self,vectors,delta,pair,period,visibility,pitch,ratio):
        """Separate the measured teaching result, component values and assumptions."""
        fmt=self._component_number
        number='text-align:right;font-variant-numeric:tabular-nums;padding:5px 12px'
        rows=[]
        for vector,name,order,colour in zip(vectors,('m','p'),pair,self.order_colours):
            rows.append(f'<tr><th scope="row" style="text-align:left;padding:5px 12px;color:{colour}">'
                f'{name} = {order:+d}</th><td style="{number}">{fmt(vector[0])}</td>'
                f'<td style="{number}">{fmt(vector[1])}</td><td style="{number}">{fmt(vector[2])}</td></tr>')
        rows.append(f'<tr style="background:#f5f1f9;color:#79539e">'
            f'<th scope="row" style="text-align:left;padding:5px 12px">Difference (p − m)</th>'
            f'<td style="{number}">{fmt(delta[0])}</td><td style="{number}">{fmt(delta[1])}</td><td style="{number}">{fmt(delta[2])}</td></tr>')
        motion=('Moving the detector leaves fringe positions unchanged: Δk<sub>z</sub> = 0.'
            if abs(delta[2])<1e-12 else
            'Moving the detector shifts the fringes because Δk<sub>z</sub> ≠ 0; their spacing stays fixed.')
        coherence=('Fully coherent pair; contrast includes amplitude imbalance.' if ratio>0 else
            'C<sub>p</sub> = 0: no fringes. Normalized mutual coherence is undefined.')
        return ('<div style="font-family:system-ui,sans-serif;color:#243444;background:#fff;'
            'border:1px solid #d9e2ec;border-radius:8px;padding:14px 16px;line-height:1.45;max-width:1000px;box-sizing:border-box">'
            '<div style="display:flex;flex-wrap:wrap;gap:18px 32px;align-items:start">'
            '<div style="flex:1;min-width:290px">'
            '<div style="font-size:12px;color:#526575;margin-bottom:5px">'
            'WAVEVECTOR COMPONENTS · rad/µm</div>'
            '<table style="border-collapse:collapse;width:100%;max-width:460px;font-size:14px">'
            '<thead><tr style="border-bottom:1px solid #d9e2ec">'
            '<th scope="col" style="text-align:left;padding:4px 12px">Order</th>'
            '<th scope="col" style="text-align:right;padding:4px 12px">k<sub>x</sub></th>'
            '<th scope="col" style="text-align:right;padding:4px 12px">k<sub>y</sub></th>'
            '<th scope="col" style="text-align:right;padding:4px 12px">k<sub>z</sub></th>'
            '</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
            '<div style="flex:1;min-width:290px">'
            '<div style="display:flex;gap:32px;flex-wrap:wrap;margin-bottom:7px">'
            '<div><div style="font-size:12px;color:#526575">FRINGE SPACING · Λ<sub>x</sub></div>'
            f'<div style="font-size:24px;font-weight:650">{period:.3f} µm</div></div>'
            '<div><div style="font-size:12px;color:#526575">VISIBILITY · V</div>'
            f'<div style="font-size:24px;font-weight:650">{visibility:.3f}</div></div></div>'
            '<div style="font-size:13px;color:#526575">'
            f'G = 2π/d = {2*np.pi/pitch:.3f} rad/µm<br>'
            f'Δk<sub>x</sub> = (n − m)G = {fmt(delta[0])} rad/µm<br>'
            f'Λ<sub>x</sub> = 2π/|Δk<sub>x</sub>| = d/|n − m| = {period:.3f} µm'
            f'<br>Normal fringe spacing = {2*np.pi/np.linalg.norm(delta[:2]):.3f} µm'
            '<br>Δky = 0: shared y phase cancels in intensity.'
            '<br>Interference phase = Δkₓ x + Δk_z zᴅ + Δφ'
            '</div></div></div>'
            f'<div style="border-top:1px solid #e8edf2;margin-top:12px;padding-top:10px;font-size:14px">'
            f'{motion}<br><span style="font-size:12px;color:#526575">{coherence}</span></div>'
            '<details style="font-size:12px;color:#526575;margin-top:8px">'
            '<summary style="cursor:pointer">Model and display notes</summary>'
            '<div style="margin-top:5px">Fixed intensity scale: 0…9. '
            'Ticks indicate grating repetition, not a solved transmission mask. '
            'The fields are ideal extended components, not finite beams or camera DN.</div>'
            '</details></div>')

    def _refresh_pairs(self,preferred=None):
        labels,_=transmitted_orders(self.wavelength.value,self.pitch.value,
            self.angle.value,self.azimuth.value)
        options=[(f'({m:+d}, {n:+d})',(int(m),int(n))) for m,n in combinations(labels,2)]
        current=preferred if preferred is not None else self.pair.value
        legal=[value for _,value in options]
        self.pair.options=options
        self.pair.disabled=not options
        self.pair.value=current if current in legal else (legal[0] if legal else None)

    def _changed(self,change):
        if self._closed or self._changing:return
        if change['owner'] in self._geometry_controls:
            self._changing=True
            try:self._refresh_pairs()
            finally:self._changing=False
        self._update()

    def _update(self):
        if self._closed:return
        self.updates+=1
        wavelength,pitch,angle=self.wavelength.value,self.pitch.value,self.angle.value
        pair=self.pair.value
        vectors=None
        if pair is None:
            self.data=None
            self.readout.value='<b>No pair of strictly propagating transmitted orders.</b> Increase pitch, reduce wavelength, or change incidence angles. Grazing directions are excluded.'
        else:
            key=(wavelength,pitch,angle,pair,self.azimuth.value)
            if key!=self._basis_key:
                self._vectors=selected_vectors(wavelength,pitch,angle,pair,self.azimuth.value)
                self._basis_key=key
                self.geometry_evaluations+=1
            vectors=self._vectors
            delta=vectors[1]-vectors[0]
            period=2*np.pi/abs(delta[0])
            ratio=self.ratio.value
            visibility=fringe_visibility(1,ratio**2)
            self.readout.value=self._readout_html(vectors,delta,pair,period,visibility,pitch,ratio)
            self.data=dict(vectors=vectors.copy(),delta=delta.copy(),period_um=period,
                component_period_um=period,normal_period_um=2*np.pi/np.linalg.norm(delta[:2]),visibility=visibility,pair=pair,
                coefficients=np.array([1,ratio*np.exp(1j*np.deg2rad(self.phase.value))]))
        self.detector_view.update(vectors,wavelength,pitch,angle,pair,
            self.ratio.value,self.phase.value,self.detector.value,self.azimuth.value)
        if self.data is not None:
            self.data['intensity']=self.detector_view.intensity
            self.data['profile']=self.detector_view.intensity[len(self.detector_view.y)//2].copy()

    def reset(self,button=None):
        if self._closed:return
        self._changing=True
        try:
            for control,default in ((self.pitch,2),(self.wavelength,2),(self.angle,30),
                    (self.ratio,1),(self.phase,0),(self.detector,-3),(self.azimuth,0)):
                control.value=default
            self._refresh_pairs(preferred=self._default_pair)
        finally:self._changing=False
        self._update()

    def close(self):
        if self._closed:return
        self._closed=True
        for control in self._controls:control.unobserve(self._changed,names='value')
        self.reset_button.on_click(self.reset,remove=True)
        self.detector_view.close()
        _close_widget_tree(self.ui)
        self.data=None
