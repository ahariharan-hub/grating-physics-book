"""A 3D Weyl decomposition, thin scalar grating, and object-side reference cap.

Coordinates are (x,y,s), s=-z; exp(-i omega t); wavelength=1. The physical
source moves as P0=(x0,0,-s0); the grating is at s=0. The cap stays fixed,
centred on the reference object point O=(0,0,0) in the grating, radius 6.
It is a collection reference surface, not an additional emitter.
Numerical spectra include both propagating and evanescent components.
"""
from functools import lru_cache
from pathlib import Path
from io import StringIO

import numpy as np
import ipywidgets as w
from matplotlib.figure import Figure
from matplotlib import rc_context

from .field_propagation import component_field
from .green_kirchhoff import _owned_panel, _slider, _png, _complex_html
from .propagation_pictures import BLUE, ORANGE, GREEN, INK, GREY, arrow

K=2*np.pi
SOURCE_DISTANCE=1.5
CAP_CENTER=(0.,0.,0.)
CAP_RADIUS=6.
CAP_ANGLES=np.arange(-40.,41.)
CAP_QUADRATURE_ORDER=64


def _order(n):
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or not 8<=n<=96:
        raise ValueError('Quadrature order must be an integer from 8 to 96.')
    return int(n)


def _period(period):
    if np.iscomplexobj(period) or np.ndim(period) or not np.isfinite(period) or not 1.2<=period<=4:
        raise ValueError('Period must be a finite real scalar from 1.2 to 4 wavelengths.')
    return float(period)


def _mode(grating):
    if not isinstance(grating,(bool,np.bool_)):
        raise ValueError('grating must be Boolean.')
    return bool(grating)


def cap_points(theta_deg, phi_deg=0.):
    """Fixed entrance cap centred at grating origin O; meridional theta."""
    if np.iscomplexobj(theta_deg) or np.iscomplexobj(phi_deg):
        raise ValueError('Cap angles must be finite and real.')
    theta,phi=np.broadcast_arrays(np.asarray(theta_deg,float),np.asarray(phi_deg,float))
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)) or np.any(abs(theta)>40):
        raise ValueError('Use finite cap angles within +/-40 degrees.')
    theta,phi=np.deg2rad(theta),np.deg2rad(phi)
    return np.stack([CAP_RADIUS*np.sin(theta)*np.cos(phi),
                     CAP_RADIUS*np.sin(theta)*np.sin(phi),
                     CAP_RADIUS*np.cos(theta)],axis=-1)+CAP_CENTER


def _source(source_x, source_distance):
    for value, low, high in ((source_x,-1.,1.),(source_distance,1.,2.5)):
        if (isinstance(value,(bool,np.bool_)) or np.iscomplexobj(value) or
                np.ndim(value) or not np.isfinite(value) or not low<=value<=high):
            raise ValueError('Use source x within +/-1 and source distance from 1 to 2.5 wavelengths.')
    return float(source_x),float(source_distance)


def reference_field(points, source_x=0., source_distance=SOURCE_DISTANCE):
    """Unobstructed spherical field from the selected physical source."""
    source_x,source_distance=_source(source_x,source_distance)
    if np.iscomplexobj(points):raise ValueError('Coordinates must be real.')
    points=np.asarray(points,float)
    if points.shape[-1:]!=(3,) or not np.all(np.isfinite(points)):
        raise ValueError('Points must be finite (x,y,s) coordinates.')
    distance=np.linalg.norm(points-np.array([source_x,0,-source_distance]),axis=-1)
    if np.any(distance==0):raise ValueError('The source field is singular at P0.')
    return np.exp(1j*K*distance)/(4*np.pi*distance)


def _one_order(n, shift, source_x, source_distance):
    """Weyl quadrature for one shifted spectrum; all azimuths are integrated.

    For incident propagating modes, kt dkt/kappa=-k dmu, mu=kappa/k.
    For incident evanescent modes, kappa=i beta and kt dkt/kappa=-i dbeta.
    This removes the incoming grazing singularity analytically. Shifted output
    grazing boundaries are split and square-mapped in polar/azimuth variables.
    beta truncation leaves an incident-plane absolute tail <=exp(-20)/(4pi*s0).
    """
    z,weights=np.polynomial.legendre.leggauss(n);t=(z+1)/2;weights=weights/2
    chunks=[];coefficients=[];g=abs(shift);bmax=20/source_distance
    if g==0:
        phi=np.arange(4*n)*2*np.pi/(4*n);dphi=2*np.pi/(4*n)
        for evanescent in (False,True):
            values=bmax*t if evanescent else t
            measure=bmax*weights if evanescent else weights
            radial=np.sqrt(K*K+values**2) if evanescent else K*np.sqrt(1-values**2)
            longitudinal=1j*values if evanescent else K*values
            for radial_q,kap,weight in zip(radial,longitudinal,measure):
                qx=radial_q*np.cos(phi);qy=radial_q*np.sin(phi)
                chunks.append(np.column_stack([qx,qy,np.full(len(phi),-kap)]))
                prefactor=1 if evanescent else 1j*K
                coefficients.append(np.full(len(phi),prefactor*weight*dphi*np.exp(1j*kap*source_distance-1j*qx*source_x)/(8*np.pi**2)))
    else:
        mu_critical=np.sqrt(1-(1-g/K)**2)
        beta_critical=np.sqrt((K+g)**2-K*K)
        for evanescent,edge,limit in [(False,mu_critical,1.),(True,beta_critical,bmax)]:
            # A moved source changes the evanescent truncation. Do not split
            # at a grazing transition that lies beyond the integration range.
            for lower in (True,False) if edge<limit else (None,):
                if lower is None:
                    values=limit*t;measure=limit*weights
                else:
                    span=edge if lower else limit-edge
                    values=edge-span*t*t if lower else edge+span*t*t
                    measure=2*span*t*weights
                for value,weight in zip(values,measure):
                    radial=np.sqrt(K*K+value*value) if evanescent else K*np.sqrt(1-value*value)
                    kap=1j*value if evanescent else K*value
                    cosine=(K*K-radial*radial-g*g)/(2*radial*g)
                    if abs(cosine)<1:
                        critical=np.arccos(cosine)
                        sectors=[(critical-critical*t*t,2*critical*t*weights),
                                 (critical+(np.pi-critical)*t*t,2*(np.pi-critical)*t*weights)]
                    else:sectors=[(np.pi*t,np.pi*weights)]
                    for phi,azimuth_weight in sectors:
                        for sign in (1,-1):
                            qx=radial*np.cos(phi);qy=sign*radial*np.sin(phi)
                            if shift<0:qx=-qx
                            output_qx=qx+shift
                            output_kappa=np.sqrt((K*K-output_qx**2-qy**2).astype(complex))
                            chunks.append(np.column_stack([output_qx,qy,-output_kappa]))
                            prefactor=1 if evanescent else 1j*K
                            coefficients.append(prefactor*weight*azimuth_weight*np.exp(1j*kap*source_distance-1j*qx*source_x)/(8*np.pi**2))
    return np.concatenate(chunks),np.concatenate(coefficients)


@lru_cache(maxsize=8)
def weyl_modes(grating=False,period=2.,order=48,source_x=0.,source_distance=SOURCE_DISTANCE):
    """Full weighted 3D components at s=0; global kz is negative/outgoing.

    The thin amplitude mask is t(x)=0.5+0.5*cos(2pi*x/d). Its three Fourier
    coefficients shift the CONTINUOUS incident spectrum; these are not three
    plane waves illuminating the whole problem. Output kappa is recomputed.
    """
    grating=_mode(grating);period=_period(period);order=_order(order)
    source_x,source_distance=_source(source_x,source_distance)
    vectors=[];coefficients=[]
    for m,c in [(-1,.25),(0,.5),(1,.25)] if grating else [(0,1.)]:
        v,a=_one_order(order,m*K/period,source_x,source_distance)
        vectors.append(v);coefficients.append(c*a)
    v=np.concatenate(vectors);a=np.concatenate(coefficients)
    v.setflags(write=False);a.setflags(write=False)
    return v,a


def reconstructed_field(points,grating=False,period=2.,order=48,source_x=0.,source_distance=SOURCE_DISTANCE):
    """Evaluate the plane-wave sum at arbitrary outgoing 3D points, in batches."""
    if np.iscomplexobj(points):raise ValueError('Coordinates must be real.')
    points=np.asarray(points,float)
    if points.shape[-1:]!=(3,) or not np.all(np.isfinite(points)) or np.any(points[...,2]<0):
        raise ValueError('Use finite (x,y,s) points with s>=0.')
    shape=points.shape[:-1];p=points.reshape(-1,3);out=np.empty(len(p),complex)
    v,a=weyl_modes(grating,period,order,source_x,source_distance)
    for j in range(0,len(p),8):
        x,y,s=p[j:j+8].T
        out[j:j+8]=component_field(v,a,x,y,s)
    return out.reshape(shape)


@lru_cache(maxsize=12)
def cap_profile(grating=False,period=2.,order=CAP_QUADRATURE_ORDER,source_x=0.,source_distance=SOURCE_DISTANCE):
    """Fixed cap and fixed original-source normalization; moving P0 stays visible."""
    points=cap_points(CAP_ANGLES)
    field=reconstructed_field(points,grating,period,order,source_x,source_distance)
    reference=reference_field(points)
    ratio=field/reference
    for array in (points,field,reference,ratio):array.setflags(write=False)
    return dict(points=points,field=field,reference=reference,ratio=ratio)


def selected_orders(angle,grating,period):
    if not np.isfinite(angle) or not -60<=angle<=60:raise ValueError('Use an input angle within +/-60 degrees.')
    period=_period(period);grating=_mode(grating)
    m=np.array([-1,0,1]) if grating else np.array([0])
    alpha=np.sin(np.deg2rad(angle))+m/period
    gamma=np.sqrt((1-alpha**2).astype(complex))
    return dict(m=m,alpha=alpha,gamma=gamma,propagating=abs(alpha)<=1)


def _spectrum_density(kx,ky,grating,period,source_x,source_distance,display_clip=False):
    """Fourier amplitude in radian wavevectors, with inverse measure dkx dky/(2pi)^2.

    The forward transform has no prefactor. Thus this equals the cycle-frequency
    transform evaluated at f=k_transverse/(2pi); the inverse measure changes.
    The image alone regularizes the grazing ring.
    """
    kx,ky=np.broadcast_arrays(kx,ky);spectrum=np.zeros(kx.shape,complex)
    for m,c in [(-1,.25),(0,.5),(1,.25)] if grating else [(0,1.)]:
        incoming_kx=kx-m*2*np.pi/period
        kappa=np.sqrt((K*K-incoming_kx**2-ky**2).astype(complex))
        denominator=np.where(abs(kappa)<.02*K,.02*K,kappa) if display_clip else kappa
        spectrum+=c*1j/(2*denominator)*np.exp(1j*kappa*source_distance-1j*incoming_kx*source_x)
    return spectrum


def _spectrum_image(grating,period,source_x=0.,source_distance=SOURCE_DISTANCE):
    """Normalized kx/k, ky/k image grid; never used for reconstruction quadrature."""
    ax=np.linspace(-2.1,2.1,211);ay=np.linspace(-1.4,1.4,141)
    a,b=np.meshgrid(ax,ay)
    spectrum=_spectrum_density(K*a,K*b,grating,period,source_x,source_distance,display_clip=True)
    return ax,ay,np.log10(np.maximum(abs(spectrum)*2*K,1e-4))


STEP_TITLES=('1. Spherical illumination','2. Plane-wave components',
             '3. Grating diffraction','4. Sum on the entrance cap')
STEP_NOTES=(
    'Move Source x or Source distance. Blue fronts and the complex field across the reference plane change together. '
    'This is the physical illumination from P0; the entrance cap and plane stay fixed.',
    'Select a component angle. Green fronts and the cyan spectrum marker identify one term of the SAME spherical field. '
    'Moving the source changes its complex weight and rotates the arrow. Decomposition is a change of description, not an optical element.',
    'Change Period. Each selected incident component produces shifted orders; the whole input produces shifted CONTINUOUS spectra. '
    'The right plot shows the prescribed thin-mask field at the plane. Move the source to change the illumination of that same grating.',
    'Move Cap point to sample the coherent sum at P. All components contribute, including evanescent ones; Component angle only highlights. '
    'The cap is a fixed observation surface centred at grating origin O, not at P0; no lens or apodization is applied. '
    'Its spherical shape does not impose a spherical wavefront.',
)


def _geometry(ax,stage,grating,period,input_angle,source_x,source_distance,point):
    """An unchanged laboratory geometry; successive steps change the description."""
    source_s=-source_distance
    cap_limit=CAP_CENTER[2]+CAP_RADIUS+.4
    curve=cap_points(CAP_ANGLES)
    if stage==0:
        ss,xx=np.meshgrid(np.linspace(-2.8,cap_limit,210),np.linspace(-4.4,4.4,210))
        distance=np.sqrt((xx-source_x)**2+(ss-source_s)**2)
        field=np.exp(1j*K*distance)/np.maximum(distance,.15)
        ax.pcolormesh(ss,xx,np.ma.masked_where(distance<.15,field.real),
                      shading='auto',cmap='RdBu_r',vmin=-1,vmax=1,alpha=.32,rasterized=True)
    theta=np.linspace(-np.pi/2,np.pi/2,300)
    outer=np.max(np.linalg.norm(curve-[source_x,0.,source_s],axis=1))
    for radius in np.arange(.5,outer+.5,.5):
        xx=source_x+radius*np.sin(theta);ss=source_s+radius*np.cos(theta)
        keep=np.ones_like(ss,dtype=bool) if stage==0 else ss<=0
        ax.plot(np.where(keep,ss,np.nan),xx,color=BLUE,alpha=.4,lw=.8)
    ax.plot(source_s,source_x,'*',color=BLUE,ms=11,zorder=4)
    ax.annotate(r'$P_0$',(source_s,source_x),xytext=(6,-17),textcoords='offset points',color=BLUE)
    ax.axhline(0,color=GREY,lw=.5,ls=':')
    ax.axvline(0,color=INK,lw=2 if grating else 1,ls='-' if grating else ':')
    ax.text(.12,-3.7,'Grating' if grating else 'Reference plane',fontsize=9)
    if grating:
        for xline in np.arange(-4,4.1,period):ax.plot([-.07,.07],[xline,xline],color=INK,lw=2)
    orders=selected_orders(input_angle,grating,period)
    colors={-1:'#8053a1',0:GREEN,1:ORANGE}
    if stage>0:
        ss,xx=np.meshgrid(np.linspace(.04,cap_limit-.2,90),np.linspace(-4.4,4.4,100))
        incident_kx=K*np.sin(np.deg2rad(input_angle))
        phase0=np.pi/2+K*np.cos(np.deg2rad(input_angle))*source_distance-incident_kx*source_x
        for m,alpha,gamma,prop in zip(orders['m'],orders['alpha'],orders['gamma'],orders['propagating']):
            if not prop:continue
            phase=K*(alpha*xx+gamma.real*ss)+phase0
            levels=np.arange(np.floor(phase.min()/(2*np.pi)),np.ceil(phase.max()/(2*np.pi))+1)*2*np.pi
            ax.contour(ss,xx,phase,levels=levels,colors=colors[int(m)],alpha=.25,linewidths=.7)
            arrow(ax,(0,0),(1.25*gamma.real,1.25*alpha),color=colors[int(m)],lw=1.8)
            if grating:ax.text(1.25*gamma.real+.05,1.25*alpha,f'm={m}',fontsize=7,color=colors[int(m)])
    ax.plot(curve[:,2],curve[:,0],color=INK,ls='-' if stage==3 else ':',lw=1.8 if stage==3 else .8)
    ax.plot(CAP_CENTER[2],CAP_CENTER[0],'o',color=INK,ms=3)
    ax.annotate(r'$O$' if stage==3 else r'$Q=O$',(CAP_CENTER[2],CAP_CENTER[0]),
                xytext=(-12,-16),textcoords='offset points',fontsize=9)
    if stage==3:
        ax.plot(point[2],point[0],'o',color=ORANGE,ms=6)
        offset=(-13,7) if point[2]>CAP_RADIUS-.4 else (5,7)
        ax.annotate(r'$P$',(point[2],point[0]),xytext=offset,textcoords='offset points',fontsize=9)
    ax.set(xlim=(-2.8,cap_limit),ylim=(-4.4,4.4),aspect='equal',xlabel=r'$s=-z$ / wavelength',
            ylabel=r'$x$ / wavelength',title='Physical spherical illumination' if stage==0 else 'Inspect selected components')
    ax.text(.02,.98,'Blue: physical illumination' if stage==0 else 'Blue: incident sphere\nFlat fronts: selected component(s)',
            transform=ax.transAxes,va='top',fontsize=9,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
    ax.text(.02,.02,'Cap centred at O; lens not modelled',transform=ax.transAxes,fontsize=8,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
    return orders


def _plane_field_plot(ax,grating,period,source_x,source_distance):
    xi=np.linspace(-4.4,4.4,501);points=np.column_stack([xi,np.zeros_like(xi),np.zeros_like(xi)])
    incident=reference_field(points,source_x,source_distance)*4*np.pi
    mask=.5+.5*np.cos(K*xi/period) if grating else np.ones_like(xi)
    prescribed=mask*incident
    if grating:
        ax.plot(xi,incident.real,color=BLUE,alpha=.6,label=r'Incident: Re $U_i$')
        ax.plot(xi,prescribed.real,color=ORANGE,lw=1.7,label=r'After mask: Re $U_{\rm ref}$')
    else:
        ax.plot(xi,incident.real,color=BLUE,lw=1.6,label='Real part')
        ax.plot(xi,incident.imag,'--',color=ORANGE,lw=1.4,label='Imaginary part')
    ax.set(xlim=(-4.4,4.4),ylim=(-1.05,1.05),xlabel=r'Plane coordinate $\xi$ / wavelength',
           ylabel=r'Field / fixed amplitude $(1/4\pi)$',
           title=r'Prescribe $U_{\rm ref}=tU_i$' if grating else r'Field on the plane: $U_{\rm ref}(Q)$')
    ax.axhline(0,color=GREY,lw=.6);ax.legend(fontsize=9,loc='lower left')
    return dict(plane_x=xi,plane_field=prescribed/(4*np.pi),incident_plane=incident/(4*np.pi),mask=mask)


def cap_figure(grating=False,period=2.,input_angle=20.,cap_angle=0.,
               source_x=0.,source_distance=SOURCE_DISTANCE,stage=3):
    grating=_mode(grating);period=_period(period)
    source_x,source_distance=_source(source_x,source_distance)
    if isinstance(stage,bool) or not isinstance(stage,(int,np.integer)) or stage not in range(4):
        raise ValueError('stage must be 0 through 3.')
    if not np.isfinite(cap_angle) or cap_angle not in CAP_ANGLES:
        raise ValueError('Select an integer cap angle within +/-40 degrees.')
    grating=grating and stage>=2
    orders=selected_orders(input_angle,grating,period)
    point=cap_points(cap_angle)
    profile=cap_profile(grating,period,CAP_QUADRATURE_ORDER,source_x,source_distance) if stage==3 else None
    fig=Figure(figsize=(10.8,4.5),layout='constrained',facecolor='white')
    axes=fig.subplots(1,2 if stage==0 else 3,
                      gridspec_kw={'width_ratios':[1.1,1.4] if stage==0 else [1.05,1.,1.05]})
    geometry=axes[0]
    _geometry(geometry,stage,grating,period,input_angle,source_x,source_distance,point)
    data=dict(profile=profile,point=point,orders=orders,grating=grating,period=period,stage=stage,
              source=np.array([source_x,0.,-source_distance]))
    if stage==0:
        data.update(_plane_field_plot(axes[1],False,period,source_x,source_distance))
    else:
        spectrum=axes[1];ax,ay,image=_spectrum_image(grating,period,source_x,source_distance)
        spectrum.imshow(image,origin='lower',extent=(ax[0],ax[-1],ay[0],ay[-1]),
                        vmin=-3,vmax=1.3,cmap='magma',aspect='equal')
        phi=np.linspace(0,2*np.pi,181);spectrum.plot(np.cos(phi),np.sin(phi),'--',color='white',alpha=.7,lw=1)
        for m,a in zip(orders['m'],orders['alpha']):
            spectrum.plot(a,0,'o',mfc='none',mec='cyan',ms=6)
            spectrum.annotate(f'm={m}' if grating else 'selected',(a,0),
                              xytext=(0,-18 if grating and m==0 else 12),textcoords='offset points',
                              color='cyan',fontsize=8,ha='center')
        spectrum.set(xlabel=r'$k_x/k$',ylabel=r'$k_y/k$',title=r'Continuous spectrum $\widetilde U_{\rm ref}$')
        spectrum.text(.02,.02,'Fixed log-magnitude scale\nDashed: propagating boundary',
                      transform=spectrum.transAxes,color='white',fontsize=8)
        data['spectrum_image']=image
        if stage==1:
            a=np.sin(np.deg2rad(input_angle))
            weight=_spectrum_density(K*a,0.,False,period,source_x,source_distance).item()
            phasor=axes[2];value=2*K*weight
            phasor.axhline(0,color=GREY,lw=.7);phasor.axvline(0,color=GREY,lw=.7)
            arrow(phasor,(0,0),(value.real,value.imag),color=GREEN,lw=2)
            phasor.set(xlim=(-2.25,2.25),ylim=(-2.25,2.25),aspect='equal',xlabel='Real part',ylabel='Imaginary part',
                       title='Selected spectral weight')
            phasor.text(.03,.03,r'Fixed scale: $2k$ times the weight',transform=phasor.transAxes,fontsize=8)
            data['spectral_weight']=weight
        elif stage==2:
            data.update(_plane_field_plot(axes[2],grating,period,source_x,source_distance))
        else:
            cap=axes[2];ratio=profile['ratio'];index=int(cap_angle+40)
            cap.plot(CAP_ANGLES,ratio.real,color=BLUE,lw=1.8,label='Real part')
            cap.plot(CAP_ANGLES,ratio.imag,'--',color=ORANGE,lw=1.5,label='Imaginary part')
            cap.axhline(1,color=GREY,lw=.7,ls=':');cap.axhline(0,color=GREY,lw=.7,ls=':')
            cap.axvline(cap_angle,color=GREEN,lw=.8,ls=':')
            cap.scatter([cap_angle,cap_angle],[ratio[index].real,ratio[index].imag],color=GREEN,s=24,zorder=3)
            cap.set(xlim=(-40,40),ylim=(-1.65,1.65),xlabel='Position on fixed cap (degrees)',
                    ylabel='U(P) / original spherical reference',title='Field sampled on the cap')
            cap.legend(loc='lower left',fontsize=9)
            cap.text(.02,.97,'Reference: initial source, no grating',transform=cap.transAxes,va='top',fontsize=8,color=GREY)
            data.update(selected_field=profile['field'][index],selected_ratio=ratio[index])
    for ax in axes:ax.tick_params(labelsize=9)
    fig.suptitle(STEP_TITLES[stage],fontsize=14,color=INK)
    return fig,data


def weyl_cap_explorer():
    controls=dict(stage=w.ToggleButtons(options=[('1 Sphere',0),('2 Components',1),('3 Diffraction',2),('4 Cap',3)],
                                      value=0,description='Step',layout=w.Layout(width='100%',max_width='100%'),
                                      style={'button_width':'125px'}),
                  source_x=_slider(0.,-1.,1.,.1,'Source x'),
                  source_distance=_slider(SOURCE_DISTANCE,1.,2.5,.1,'Source distance'),
                  grating=w.ToggleButtons(options=[('No grating',False),('With grating',True)],value=True,description='Mask',
                                         layout=w.Layout(max_width='100%'),style={'button_width':'110px'}),
                  period=_slider(2.,1.2,4.,.2,'Period'),
                  input_angle=_slider(20.,-60,60,5,'Component angle'),
                  cap_angle=_slider(0.,-40,40,1,'Cap point'))
    def render(panel,values):
        stage=values['stage']
        keys=['stage','source_x','source_distance']
        if stage>=2:
            keys+=['grating']
            if values['grating']:keys+=['period']
        if stage>0:keys+=['input_angle']
        if stage==3:keys+=['cap_angle']
        panel.children[0].children=tuple(controls[k] for k in keys)
        fig,data=cap_figure(**values);panel.image.value=_png(fig);panel.data=data
        panel.note.value='<p>'+STEP_NOTES[stage]+'</p>'
        source=f'<div><b>Physical source P₀ = (x₀, 0, −s₀):</b> ({values["source_x"]:+.1f}, 0, −{values["source_distance"]:.1f}) wavelengths</div>'
        if stage==0:
            readout='<div><b>Plane field:</b> U_ref(Q) = exp(ik|Q − P₀|) / (4π|Q − P₀|).</div>'
        elif stage==1:
            a=data['spectral_weight']
            readout=('<div><b>Selected complex spectral weight:</b> '+_complex_html(a)+'</div>'+
                     f'<div><b>Magnitude / phase:</b> {abs(a):.4f} / {np.angle(a):+.3f} rad</div>')
        elif stage==2:
            readout=('<div><b>Plane prescription:</b> U_ref = t U_i; t(x) = ½ + ½ cos(2πx/d).</div>' if values['grating'] else
                     '<div><b>Plane prescription:</b> U_ref = U_i; no mask.</div>')
        else:
            field=data['selected_field'];ratio=data['selected_ratio']
            readout=('<div><b>Field at P:</b> '+_complex_html(field)+'</div>'+
                     f'<div><b>Relative to the ORIGINAL spherical reference:</b> amplitude {abs(ratio):.3f}; phase {np.angle(ratio):+.3f} rad</div>')
        panel.equation.value=(source+readout+
            '<details><summary>Numerical model</summary>Full 3D Weyl quadrature, including evanescent components. '
            'The cap and amplitude scales stay fixed as the source moves. Only the spectrum picture clips the grazing singularity. '
            'A thin scalar mask is prescribed; material scattering and a collecting lens are not solved.</details>')
    return _owned_panel(controls,render)


def write_static(path):
    """Supplementary final-step state for a reader without a running kernel."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with rc_context({'svg.fonttype':'path','svg.hashsalt':'weyl_cap_8_3','font.family':'DejaVu Sans'}):
        fig,_=cap_figure(grating=True)
        try:
            stream=StringIO();fig.savefig(stream,format='svg',bbox_inches='tight',pad_inches=.06,
                                         metadata={'Date':None,'Title':'Weyl spectrum and entrance cap'})
            contents='\n'.join(line.rstrip() for line in stream.getvalue().splitlines())+'\n'
            if not path.exists() or path.read_text(encoding='utf8')!=contents:path.write_text(contents,encoding='utf8')
        finally:fig.clear()
    return path
