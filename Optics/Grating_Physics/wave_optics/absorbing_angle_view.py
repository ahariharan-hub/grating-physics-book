"""Field-first angle experiment using the absorber from lesson 0.7.

The existing oblique scalar solver and ideal opening model supply all physics.
Only angle/thickness changes solve scattering; display controls reuse solutions.
"""
from collections import OrderedDict
from dataclasses import replace
import io
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .amplitude_grating_2d import absorbing_screen_configuration, AbsorbingComparison2DData
from .grating_intro import _close_widget_tree


class AbsorbingAngleExplorer:
    """Same centred air opening and illustrative absorber as 0.7, now tilted.

    Use My=0, fill_y=1 and ky_incident=0. Display limit affects only the order
    plot; maps and profiles always use every retained harmonic. Cache at most
    eight physical cases. Call close() before replacing the explorer.
    """
    def __init__(self, configuration=None, *, display_order_limit=7):
        import ipywidgets as widgets
        self.configuration = configuration or absorbing_screen_configuration(
            fill_y=1., My=0, Mx=24)
        c = self.configuration
        if c.My != 0 or c.fill_y != 1 or c.ky_incident != 0:
            raise ValueError('Use the y-invariant, in-plane incidence configuration.')
        if c.n_inclusion != c.n_incident or c.n_output != c.n_incident:
            raise ValueError('The opening and both exterior indices must agree.')
        self._closed = self._resetting = False
        self._cache = OrderedDict()
        self.solve_count = 0
        self.display_order_limit = display_order_limit
        self.x = np.linspace(-c.period_x/2, c.period_x/2, 512, endpoint=False)
        self.map_x = np.linspace(-1.5*c.period_x, 1.5*c.period_x, 280)
        # Follow the exit, keeping the output region at a fixed distance below it.
        self._defaults = (float(np.rad2deg(np.arcsin(c.kx_incident/(2*np.pi*c.n_incident/c.wavelength)))), c.thickness)
        style = {'description_width': 'initial'}
        self.angle = widgets.FloatSlider(value=self._defaults[0], min=-60., max=60., step=1.,
            description='Incidence (deg)', continuous_update=False, style=style)
        self.thickness = widgets.SelectionSlider(options=sorted(set((.03,.08,.15,.3,.5,.75,1.,c.thickness))),
            value=c.thickness, description='Thickness (um)', continuous_update=False, style=style)
        self.distance = widgets.FloatSlider(value=0., min=0., max=1., step=.02,
            description='Below exit (um)', continuous_update=False, style=style)
        self.quantity = widgets.Dropdown(options=[('Real field Re U','real'),('Intensity |U| squared','intensity'),
            ('Amplitude |U|','amplitude'),('Wrapped phase','phase')], value='real', description='Show', style=style)
        self.compare_ideal = widgets.Checkbox(value=False, description='Compare ideal mask', indent=False)
        self.order = widgets.Dropdown(options=list(range(-min(c.Mx,display_order_limit),min(c.Mx,display_order_limit)+1)),
            value=0, description='Order m', style=style)
        self.reset = widgets.Button(description='Reset experiment')
        self.readout = widgets.HTML()
        self.pictures = [widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1200px')) for _ in range(2)]
        self.tabs = widgets.Tab(children=self.pictures)
        self.tabs.set_title(0, '1. Structure and field')
        self.tabs.set_title(1, '2. Which orders carry the light?')
        self.ui = widgets.VBox([widgets.HBox([self.angle,self.distance,self.reset]),
            widgets.HBox([self.quantity,self.compare_ideal]),
            widgets.Accordion(children=[widgets.HBox([self.thickness,self.order])],selected_index=None),
            self.readout,self.tabs])
        self.ui.children[2].set_title(0,'Optional: thickness and selected order')
        self._controls = (self.angle,self.thickness,self.distance,self.quantity,self.compare_ideal,self.order)
        for w in self._controls: w.observe(self._update,names='value')
        self.tabs.observe(self._update,names='selected_index')
        self.reset.on_click(self._reset)
        self._update()

    def _case(self):
        key = (self.angle.value,self.thickness.value)
        if key not in self._cache:
            c = replace(self.configuration, thickness=key[1],
                kx_incident=2*np.pi*self.configuration.n_incident/self.configuration.wavelength*np.sin(np.deg2rad(key[0])))
            data = AbsorbingComparison2DData(c)
            z = np.linspace(-c.thickness-1., .55, 240)
            field = data.solution.reconstruct(self.map_x,[0.],z)[:,0,:]
            material = c.refractive_index(self.map_x,[0.],z)[:,0,:].imag
            self._cache[key] = (data,z,field,material)
            self.solve_count += 1
            if len(self._cache)>8: self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def _values(self, field):
        if self.quantity.value=='real': return field.real
        if self.quantity.value=='intensity': return abs(field)**2
        if self.quantity.value=='amplitude': return abs(field)
        return np.where(abs(field)>1e-6,np.angle(field),np.nan)

    def _png(self, figure):
        FigureCanvasAgg(figure)
        stream=io.BytesIO(); figure.savefig(stream,format='png',facecolor='white')
        return stream.getvalue()

    def _update(self, change=None):
        if self._closed or self._resetting: return
        self.readout.value='Calculating the selected physical case...'
        self.data,z,field,material = self._case()
        data=self.data; c=data.configuration; s=data.solution
        plane=-c.thickness-self.distance.value
        self.field = s.reconstruct(self.x,[0.],[plane])[0,0]
        self.ideal_field = data.ideal.reconstruct(self.x,[0.],[plane])[0,0]
        self.ideal_representation = 'thin mask propagated from z=0'
        ideal_label = f'Ideal mask: propagated from z=0 (M={c.Mx})'
        self.readout.value=(f'<b>Angle {self.angle.value:+.0f} deg; observation z={plane:.2f} um.</b> '
            f'R={s.reflectance:.3f}, T={s.transmittance:.3f}, A={s.absorption:.3f}; '
            f'R+T+A={s.energy_balance:.6f}.<br>'
            'Real field shows wavefronts; intensity shows where the field is strong. '
            'Above the layer the physical field includes incident + reflected waves. '
            f'Phase change across the clear opening: {c.kx_incident*c.fill_x*c.period_x:+.2f} rad. '
            f'{2*c.Mx+1} harmonics; illustrative material index {c.n_background}.')
        if self.compare_ideal.value:
            self.readout.value += ('<br><b>Ideal profile:</b> '+self.ideal_representation+
                '. The binary boundary is prescribed at z=0 and propagated to this plane.')
        tab=self.tabs.selected_index
        if tab is None:return
        if tab==0:
            fig=Figure(figsize=(11.5,7.2),dpi=100,layout='constrained')
            grid=fig.add_gridspec(2,2,height_ratios=(1.5,1))
            geometry=fig.add_subplot(grid[0,0]); actual=fig.add_subplot(grid[0,1]); profile=fig.add_subplot(grid[1,:])
            extent=[self.map_x[0],self.map_x[-1],z[0],z[-1]]
            geometry.imshow(material,origin='lower',extent=extent,aspect='equal',cmap='Greys',vmin=0,vmax=max(.01,c.n_background.imag))
            geometry.set(title='Same air openings in an absorbing layer')
            angle=np.deg2rad(self.angle.value)
            geometry.annotate('Incident',xy=(0.,.02),xytext=(-.45*np.sin(angle),.02+.45*np.cos(angle)),ha='center',
                arrowprops={'arrowstyle':'->','color':'#1976b3'},color='#1976b3')
            values=self._values(field)
            limits=(-1.5,1.5) if self.quantity.value=='real' else ((-np.pi,np.pi) if self.quantity.value=='phase' else (0.,3. if self.quantity.value=='intensity' else 1.75))
            image=actual.imshow(values,origin='lower',extent=extent,aspect='equal',
                cmap='RdBu_r' if self.quantity.value=='real' else ('twilight' if self.quantity.value=='phase' else 'magma'),
                vmin=limits[0],vmax=limits[1])
            fig.colorbar(image,ax=actual,shrink=.8,label=self.quantity.label)
            actual.set(title='Physical field: fixed colour scale across angles')
            for axis in (geometry,actual):
                axis.axhline(0,color='grey',ls='--',lw=.8)
                axis.axhline(-c.thickness,color='grey',ls='--',lw=.8)
                axis.axhline(plane,color='#18a7b5',lw=1.4)
                axis.set(xlabel='x (um)',ylabel='z (um)')
            profile.plot(self.x,self._values(self.field),label='Physical absorber',color='#1976b3')
            if self.compare_ideal.value:
                profile.plot(self.x,self._values(self.ideal_field),'--',label=ideal_label,color='#df791a')
            profile.set(xlabel='x (um)',ylabel=self.quantity.label,title='Profile on the cyan observation plane (one period)')
            if self.quantity.value=='phase':profile.set_ylim(-3.4,3.4)
            profile.legend(fontsize=9);profile.grid(alpha=.15)
        else:
            fig=Figure(figsize=(11.5,4.4),dpi=100,layout='constrained')
            axes=fig.subplots(1,2)
            coefficients,ideal=data.coefficients(self.distance.value)
            m=s.orders[:,0];keep=abs(m)<=self.display_order_limit
            for ax,physical,prescribed,title in (
                (axes[0],abs(coefficients),abs(ideal),'Complex coefficient magnitude'),
                (axes[1],s.transmission_efficiency,data.ideal.transmission_efficiency,'Transmitted normal-power fraction')):
                ax.plot(m[keep],physical[keep],'o-',label='Physical absorber')
                if self.compare_ideal.value:ax.plot(m[keep],prescribed[keep],'x--',label='Ideal mask')
                ax.axvline(self.order.value,color='#18a7b5',alpha=.6)
                ax.set(xlabel='Order m',ylabel='|C_m|' if ax is axes[0] else 'Power fraction',title=title,xticks=m[keep])
                ax.grid(alpha=.15);ax.legend(fontsize=8)
            j=np.flatnonzero(m==self.order.value)[0]
            self.readout.value+=(f'<br><b>m={self.order.value:+d}: {s.output_kinds[j]}</b>; '
                f'C={coefficients[j]:.4g}, transmitted power fraction={s.transmission_efficiency[j]:.4g}. '
                'Nonzero evanescent amplitudes carry no normal output power.')
        self.pictures[tab].value=self._png(fig)

    def _reset(self, _):
        self._resetting=True
        self.angle.value,self.thickness.value=self._defaults
        self.distance.value=0.;self.quantity.value='real';self.compare_ideal.value=False
        self.order.value=0;self.tabs.selected_index=0
        self._resetting=False;self._update()

    def close(self):
        if self._closed:return
        self._closed=True
        for control in self._controls:control.unobserve(self._update,names='value')
        self.tabs.unobserve(self._update,names='selected_index')
        self.reset.on_click(self._reset,remove=True)
        _close_widget_tree(self.ui)
        self._cache.clear()
