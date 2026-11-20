"""Retained order-propagation explorer for later Chapter 8 worked lessons."""
from io import BytesIO
import numpy as np
import ipywidgets as w
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .field_propagation import prescribed_order_data, component_field
from .grating_intro import _close_widget_tree

BLUE,ORANGE='#2866a3','#b85c17'


def png(figure):
    buffer=BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    return buffer.getvalue()


def masked_phase(field):
    """Phase is omitted below 1e-8 of the unit field amplitude."""
    return np.ma.masked_where(abs(field)<1e-8,np.angle(field))


class FieldPropagationExplorer:
    """Only mode and positive observation distance vary; cached evolution map."""
    def __init__(self):
        self.closed=False
        self.mode=w.Dropdown(options=[('Coherent orders 0 and +1','pair'),
                                      ('Single order +1','single')],description='Input field')
        self.distance=w.FloatSlider(value=2.,min=0.,max=12.,step=.1,
                                    description='s (um)',continuous_update=False)
        self.reset=w.Button(description='Reset')
        self.picture=w.Image(format='png',layout=w.Layout(width='100%'))
        self.readout=w.HTML()
        self.ui=w.VBox([w.HBox([self.mode,self.distance,self.reset]),self.readout,self.picture])
        self.x=np.linspace(-8.,8.,801)
        self.depths=np.linspace(0.,12.,241)
        self._case=None
        self.data=None
        self.figure=None
        self.mode.observe(self.update,names='value')
        self.distance.observe(self.update,names='value')
        self.reset.on_click(self._reset)
        self.update()

    def _reset(self,_):
        self.mode.value='pair'
        self.distance.value=2.

    def update(self,_=None):
        if self.closed:return
        if self._case!=self.mode.value:
            self._case=self.mode.value
            self.data=prescribed_order_data((0,1) if self._case=='pair' else (1,))
            self.reference=component_field(self.data['vectors'],self.data['coefficients'],self.x)
            self.evolution=abs(component_field(self.data['vectors'],self.data['coefficients'],
                             self.x[None,:],s=self.depths[:,None]))**2
        self.observed=component_field(self.data['vectors'],self.data['coefficients'],self.x,
                                     s=self.distance.value)
        self.figure=self.make_figure()
        self.picture.value=png(self.figure)
        kz=self.data['vectors'][:,2].real
        phases=-kz*self.distance.value
        labels=', '.join(f'm={m:+d}: C_ref=1, kz={z:.4f}, phase gain={p:.3f} rad'
                         for m,z,p in zip(self.data['orders'],kz,phases))
        self.readout.value=(f'<b>Prescribed input; no scattering solve.</b> z_ref=0 um; '
                            f'z_obs={-self.distance.value:.1f} um. {labels}. '
                            'Intensity scale 0–4 in both modes; zero-amplitude phase is omitted.')

    def make_figure(self):
        fig=Figure(figsize=(11,7.2),layout='constrained')
        axes=fig.subplots(2,2)
        for field,color,label in [(self.reference,BLUE,'Reference s=0'),
                                  (self.observed,ORANGE,f'Observation s={self.distance.value:.1f} um')]:
            axes[0,0].plot(self.x,abs(field),color=color,label=label)
            axes[0,1].plot(self.x[::4],masked_phase(field)[::4],'.',ms=2.5,color=color,label=label)
            axes[1,1].plot(self.x,abs(field)**2,color=color,label=label)
        axes[0,0].set(title='Complex field: magnitude',ylabel='|U| (unit field amplitude)',ylim=(-.05,2.1))
        axes[0,1].set(title='Complex field: phase (zeros omitted)',ylabel='arg U (rad)',ylim=(-3.4,3.4))
        axes[1,1].set(title='Relative intensity on the two planes',ylabel='I = |U| squared',ylim=(-.1,4.2))
        im=axes[1,0].imshow(self.evolution,origin='upper',extent=[-8,8,12,0],
                          aspect='auto',cmap='viridis',vmin=0,vmax=4)
        axes[1,0].axhline(self.distance.value,color=ORANGE,ls='--',lw=1.5)
        axes[1,0].set(title='Between the planes: relative intensity',ylabel='s (um), toward -z')
        fig.colorbar(im,ax=axes[1,0],label='I = |U| squared')
        for ax in axes.ravel():ax.set(xlabel='x (um)',xlim=(-8,8))
        axes[0,0].legend(fontsize=8)
        for ax in [axes[0,0],axes[0,1],axes[1,1]]:ax.grid(alpha=.2)
        return fig

    def close(self):
        if self.closed:return
        self.closed=True
        self.mode.unobserve(self.update,names='value')
        self.distance.unobserve(self.update,names='value')
        self.reset.on_click(self._reset,remove=True)
        self.picture.value=b''
        _close_widget_tree(self.ui)
        self.figure=self.data=self.reference=self.observed=self.evolution=None


def phase_information_figure(distance=2.):
    """Exact equal-reference-intensity comparison: orders (0,+1) vs (0,-1)."""
    x=np.linspace(-8.,8.,801)
    fig=Figure(figsize=(11,6),layout='constrained')
    axes=fig.subplots(2,2)
    for orders,color,label in [((0,1),BLUE,'A: 0,+1'),((0,-1),ORANGE,'B: 0,-1')]:
        d=prescribed_order_data(orders)
        u0=component_field(d['vectors'],d['coefficients'],x)
        us=component_field(d['vectors'],d['coefficients'],x,s=distance)
        axes[0,0].plot(x,abs(u0)**2,color=color,label=label)
        axes[0,1].plot(x[::4],masked_phase(u0)[::4],'.',ms=2.5,color=color,label=label)
        axes[1,0].plot(x,abs(us)**2,color=color,label=label)
        axes[1,1].plot(x[::4],masked_phase(us)[::4],'.',ms=2.5,color=color,label=label)
    axes[0,0].set(title='Equal reference intensities, s=0',ylabel='I = |U| squared',ylim=(-.1,4.2))
    axes[0,1].set(title='Different reference phases',ylabel='arg U (rad)',ylim=(-3.4,3.4))
    axes[1,0].set(title=f'Different intensities, s={distance:g} um',ylabel='I = |U| squared',ylim=(-.1,4.2))
    axes[1,1].set(title=f'Downstream phases, s={distance:g} um',ylabel='arg U (rad)',ylim=(-3.4,3.4))
    axes[0,0].legend(fontsize=8)
    for ax in axes.ravel():ax.set(xlabel='x (um)',xlim=(-8,8));ax.grid(alpha=.2)
    return fig
