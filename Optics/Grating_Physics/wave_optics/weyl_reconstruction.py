"""Build a spherical field by refining its weighted plane-wave sum.

Section 5 of 8.2 uses one physical source and a flat observation plane.
All lengths are in wavelengths; exp(-i omega t), s=-z, lambda=1.
The target map integrates the azimuth analytically. The interactive sum
independently adds full 3D quadrature modes, displayed in a meridional section.
No grating, collection aperture, reference sphere or lens is introduced.
"""
from functools import lru_cache

import numpy as np
import ipywidgets as w
from matplotlib import rc_context
from matplotlib.figure import Figure
from scipy.special import j0

from .green_kirchhoff import _owned_panel, _png
from .propagation_pictures import BLUE, GREY
from .weyl_cap import K, SOURCE_DISTANCE, _source, _order, reference_field, weyl_modes


BUILD_ORDERS=(None,8,16,24)
BUILD_LABELS=('One plane wave','Coarse sum','Finer sum','Dense sum')


def weyl_field(points, source_x=0., source_distance=SOURCE_DISTANCE, order=48):
    """Full 3D Weyl sum, integrating azimuth to J0 before radial quadrature.

    For mu=kappa/k, the propagating measure is i*k*dmu/(4*pi).
    For kappa=i*beta, the evanescent measure is dbeta/(4*pi).
    Both branches are retained. beta_max=20/s0 matches the existing backend.
    """
    source_x,source_distance=_source(source_x,source_distance);order=_order(order)
    if np.iscomplexobj(points):raise ValueError('Points must be finite real coordinates.')
    p=np.asarray(points,float)
    if p.shape[-1:]!=(3,) or not np.all(np.isfinite(p)) or np.any(p[...,2]<0):
        raise ValueError('Use finite (x,y,s) points with s>=0.')
    shape=p.shape[:-1];p=p.reshape(-1,3)
    nodes,weights=np.polynomial.legendre.leggauss(order)
    mu=(nodes+1)/2;weights=weights/2
    beta_max=20/source_distance;beta=beta_max*mu
    kt_prop=K*np.sqrt(1-mu*mu);kt_ev=np.sqrt(K*K+beta*beta)
    result=np.empty(len(p),complex)
    for start in range(0,len(p),512):
        batch=p[start:start+512]
        rho=np.hypot(batch[:,0]-source_x,batch[:,1])[:,None]
        depth=(batch[:,2]+source_distance)[:,None]
        prop=1j*K*np.sum(weights*j0(rho*kt_prop)*np.exp(1j*K*mu*depth),axis=1)
        ev=beta_max*np.sum(weights*j0(rho*kt_ev)*np.exp(-beta*depth),axis=1)
        result[start:start+512]=(prop+ev)/(4*np.pi)
    return result.reshape(shape)


@lru_cache(maxsize=6,typed=True)
def field_map(source_x=0.):
    """Numerically reconstructed meridional section, not an analytic drawing."""
    source_x,_=_source(source_x,SOURCE_DISTANCE)
    x=np.linspace(-3.2,3.2,97);s=np.linspace(0.,3.5,97)
    ss,xx=np.meshgrid(s,x,indexing='ij')
    points=np.stack([xx,np.zeros_like(xx),ss],axis=-1)
    field=weyl_field(points,source_x)
    phase=np.unwrap(np.unwrap(np.angle(field),axis=0),axis=1)
    # Only the phase contours use this unwrapped array. Values still carry
    # absolute complex phase, including their 1/(4*pi*R) normalization.
    for value in (x,s,ss,xx,field,phase):value.setflags(write=False)
    return dict(x=x,s=s,ss=ss,xx=xx,field=field,phase=phase)


def _step(step):
    if isinstance(step,(bool,np.bool_)) or not isinstance(step,(int,np.integer)) or not 0<=step<=3:
        raise ValueError('Build step must be an integer from 0 through 3.')
    return int(step)


@lru_cache(maxsize=4,typed=True)
def reconstruction_state(step=0):
    """A selected plane wave, then progressively finer full 3D Weyl sums.

    Each quadrature level resamples the continuous integral with its proper
    weights; it is not a prefix of an unchanged finite list of components.
    Counts include both propagating and evanescent modes after the first step.
    """
    step=_step(step);grid=field_map(0.)
    if step==0:
        angle=np.deg2rad(30.)
        kx=K*np.sin(angle);kappa=K*np.cos(angle)
        # A single propagating component selected from the Weyl spectral density.
        coefficient=1j/(2*kappa)*np.exp(1j*kappa*SOURCE_DISTANCE)
        vectors=np.array([[kx,0.,-kappa]],complex)
        coefficients=np.array([coefficient])
    else:
        vectors,coefficients=weyl_modes(order=BUILD_ORDERS[step])
    # Separable evaluation on y=0: still integrates every 3D mode, including ky.
    along_s=np.exp(-1j*grid['s'][:,None]*vectors[None,:,2])
    along_x=coefficients[:,None]*np.exp(1j*vectors[:,0,None]*grid['x'][None,:])
    field=along_s@along_x
    points=np.stack([grid['xx'],np.zeros_like(grid['xx']),grid['ss']],axis=-1)
    target=reference_field(points)
    for value in (vectors,coefficients,field,target):value.setflags(write=False)
    return dict(step=step,grid=grid,vectors=vectors,coefficients=coefficients,
                count=len(coefficients),field=field,target=target,
                error=float(np.max(abs(field-target))))


def reconstruction_figure(data):
    """Phase only, with one stable colour scale and identical spatial geometry."""
    with rc_context({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                     'xtick.labelsize':9,'ytick.labelsize':9}):
        fig=Figure(figsize=(9.,4.8),layout='constrained',facecolor='white')
        current,target=fig.subplots(1,2)
        grid=data['grid'];image=None
        for ax,field in [(current,data['field']),(target,data['target'])]:
            image=ax.pcolormesh(grid['ss'],grid['xx'],np.angle(field),shading='gouraud',
                               cmap='twilight',vmin=-np.pi,vmax=np.pi,rasterized=True)
            ax.axvline(0,color=GREY,lw=1.,ls=':')
            ax.plot(-SOURCE_DISTANCE,0,'*',color=BLUE,ms=10)
            ax.annotate(r'Source $P_0$',(-SOURCE_DISTANCE,0),xytext=(0,-21),
                        textcoords='offset points',ha='center',fontsize=9,color=BLUE)
            ax.text(0,3.3,'Reference plane',ha='center',va='bottom',fontsize=8)
            ax.set(xlim=(-2.1,3.55),ylim=(-3.6,3.6),aspect='equal',
                   xlabel=r'Downstream $s=-z$ / $\lambda$',ylabel=r'$x/\lambda$')
        current.set_title(('Start: one flat plane wave','Coarse sum: interference',
                           'Finer sum: curved fronts','Dense sum: spherical fronts')[data['step']])
        target.set_title('Target: spherical wave from the source')
        bar=fig.colorbar(image,ax=[current,target],shrink=.7,pad=.025,
                         ticks=[-np.pi,0,np.pi])
        bar.ax.set_yticklabels([r'$-\pi$','0',r'$\pi$'])
        bar.set_label('Phase (rad)')
        return fig


def reconstruction_explorer(step=0):
    step=_step(step)
    control=w.SelectionSlider(options=list(zip(BUILD_LABELS,range(4))),value=step,
                              description='Build the sum',continuous_update=False,
                              style={'description_width':'initial'},
                              layout=w.Layout(width='600px',max_width='100%'))
    notes=(
        'Start here: straight stripes mark the flat phase fronts of one plane wave.',
        'The fields now interfere. The combined fronts change shape, but this coarse sampling still leaves large ripples.',
        'The curved stripes now closely follow the spherical target. More directions resolve the cancellation between components.',
        'The dense Weyl sum matches the spherical target. Flat components have reconstructed curved phase fronts.',
    )

    def render(panel,values):
        data=reconstruction_state(**values);panel.data=data
        panel.image.value=_png(reconstruction_figure(data))
        panel.note.value='<div>'+notes[data['step']]+'</div>'
        detail=('The first view isolates one propagating spectral component.' if data['step']==0 else
                f'{data["count"]} weighted quadrature components; propagating and evanescent branches included. '
                f'Maximum complex-field difference from the target: {data["error"]:.2e}.')
        panel.equation.value=(
            '<div><b>Read the colours:</b> matching colours mean matching phase. '
            'Colour does not encode brightness or intensity.</div>'
            '<details><summary>What is being added?</summary><div>'+detail+
            ' Each later step samples the continuous Weyl integral more finely; its quadrature weights are recomputed. '
            'These counts describe the numerical representation.</div></details>')
    return _owned_panel(dict(step=control),render)


