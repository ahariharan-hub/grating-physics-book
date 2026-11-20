"""Interactive physical pictures for 8.1; distances are in wavelengths.

The patch experiment integrates the full RS I kernel over a prescribed square
aperture field. It is a midpoint quadrature, not a solution for screen scattering.
"""
import numpy as np
import ipywidgets as widgets
from matplotlib.figure import Figure
from .propagation_pictures import BLUE, ORANGE, GREEN, INK


STAGES = ('Point response', 'Boundary choice', 'Patch sum')
CHOICES = ('Kirchhoff', 'RS I', 'RS II')


def _finite_real(value, name, *, positive=False):
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must be finite and real.')
    if not np.all(np.isfinite(array)) or (positive and np.any(array <= 0)):
        raise ValueError(f'{name} must be finite'+(' and positive.' if positive else '.'))
    return array.astype(float)


def _scalar(value, name, *, positive=False):
    array = _finite_real(value, name, positive=positive)
    if array.ndim:
        raise ValueError(f'{name} must be a scalar.')
    return float(array)


def _count(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 3:
        raise ValueError('count must be an integer of at least 3.')
    return int(value)


def green(s, transverse, k=2*np.pi):
    s = _scalar(s, 's', positive=True)
    k = _scalar(k, 'k', positive=True)
    r = np.hypot(s, _finite_real(transverse, 'transverse'))
    return np.exp(1j*k*r)/(4*np.pi*r)


def green_normal(s, transverse, k=2*np.pi):
    """Derivative with respect to Q's normal into the right-hand half-space."""
    s = _scalar(s, 's', positive=True)
    k = _scalar(k, 'k', positive=True)
    r = np.hypot(s, _finite_real(transverse, 'transverse'))
    return s*np.exp(1j*k*r)*(1-1j*k*r)/(4*np.pi*r**3)


def _aperture_parameters(s, width, ramp, count, amplitude):
    s = _scalar(s, 's', positive=True)
    width = _scalar(width, 'width', positive=True)
    ramp = _scalar(ramp, 'ramp')
    count = _count(count)
    amplitude = _scalar(amplitude, 'amplitude')
    if amplitude < 0:
        raise ValueError('amplitude must be nonnegative.')
    return s, width, ramp, count, amplitude


def _row_contributions(s, positions, width, ramp, count, amplitude):
    """Shared full RS I midpoint quadrature, batched over observation points."""
    step = width/count
    q = (np.arange(count)+.5)*step-width/2
    eta, xi = np.meshgrid(q, q, indexing='ij')
    r = np.sqrt(s*s+(positions[..., None, None]-eta)**2+xi**2)
    kernel = s*np.exp(2j*np.pi*r)*(1-2j*np.pi*r)/(2*np.pi*r**3)
    values = amplitude*np.exp(1j*ramp*eta)*kernel*step**2
    return q, values.sum(axis=-1)


def patches(s, y, width, ramp, count=41, amplitude=1.):
    """Square-aperture field amplitude*exp(i*ramp*eta), zero outside.

    Return row coordinates and row contributions, each summed over the second
    transverse coordinate. Both transverse dimensions are integrated.
    """
    y = _scalar(y, 'y')
    s, width, ramp, count, amplitude = _aperture_parameters(s, width, ramp, count, amplitude)
    return _row_contributions(s, np.asarray(y), width, ramp, count, amplitude)


def screen_profile(s, width, ramp, positions=None, count=41, amplitude=1.):
    """Calculated cut U(0,y,s), with I/I_ref=|U|^2 and fixed |U_inc,ref|=1.

    No peak or per-setting renormalization is performed. Batching bounds the
    temporary quadrature arrays. Source and screen coordinates are wavelengths.
    """
    s, width, ramp, count, amplitude = _aperture_parameters(s, width, ramp, count, amplitude)
    positions = np.linspace(-8, 8, 321) if positions is None else _finite_real(positions, 'positions')
    if positions.ndim != 1 or len(positions) < 2 or np.any(np.diff(positions) <= 0):
        raise ValueError('positions must be a strictly increasing 1D array with at least two points.')
    positions = np.array(positions, dtype=float, copy=True)
    field = np.empty(len(positions), dtype=complex)
    for start in range(0, len(positions), 32):
        _, rows = _row_contributions(s, positions[start:start+32], width, ramp, count, amplitude)
        field[start:start+32] = rows.sum(axis=-1)
    intensity = abs(field)**2
    for array in (positions, field, intensity):
        array.setflags(write=False)
    return dict(y=positions, field=field, intensity=intensity,
                parameters=(s, width, ramp, count, amplitude), intensity_reference=1.)


def poisson_arago_field(rho, s=25., radius=5., count=120):
    """Plane-wave disk shadow via Fresnel propagation and scalar Babinet.

    Wavelength and incident amplitude are one. This is an explicitly paraxial
    example, not the full RS I square-aperture calculation used by the widget.
    Angular integration is analytic (J0); radial quadrature is Gauss-Legendre.
    """
    from scipy.special import j0, roots_legendre
    rho = _finite_real(rho, 'rho')
    s = _scalar(s, 's', positive=True)
    radius = _scalar(radius, 'radius', positive=True)
    count = _count(count)
    nodes, weights = roots_legendre(count)
    r = (nodes+1)*radius/2
    weights = weights*radius/2
    integrand = (r*np.exp(1j*np.pi*r*r/s)*weights)*j0(2*np.pi*rho[...,None]*r/s)
    hole = (2*np.pi/(1j*s))*np.exp(2j*np.pi*s+1j*np.pi*rho*rho/s)*integrand.sum(axis=-1)
    return np.exp(2j*np.pi*s)-hole


def _phasor(ax, values, selected=None):
    chain = np.r_[0j, np.cumsum(values)]
    ax.plot(chain.real, chain.imag, '.-', color=BLUE, lw=1, ms=3)
    if selected is not None:
        a,b=chain[selected:selected+2]
        ax.annotate('',(b.real,b.imag),(a.real,a.imag),
                    arrowprops=dict(arrowstyle='->',color=ORANGE,lw=2.5))
    ax.annotate('',(chain[-1].real,chain[-1].imag),(0,0),
                arrowprops=dict(arrowstyle='->',color=GREEN,lw=2))
    ax.axhline(0,color='0.8',lw=.6);ax.axvline(0,color='0.8',lw=.6)
    ax.set(xlabel='Real field',ylabel='Imaginary field',aspect='equal')
    # Keep a square complex plane even when the chain is almost vertical.
    # Independent autoscaling otherwise squeezes it into a thin, unreadable box.
    span=max(float(np.ptp(chain.real)),float(np.ptp(chain.imag))) or 1.
    cx=(chain.real.min()+chain.real.max())/2
    cy=(chain.imag.min()+chain.imag.max())/2
    ax.set(xlim=(cx-.65*span,cx+.65*span),ylim=(cy-.65*span,cy+.65*span))
    return chain[-1]


def picture(stage, s=3., y=.5, width=3., ramp=0., choice='RS I', row=20,
            amplitude=1., profile=None):
    if stage not in STAGES:
        raise ValueError(f'stage must be one of {STAGES}.')
    if choice not in CHOICES:
        raise ValueError(f'choice must be one of {CHOICES}.')
    s = _scalar(s, 's', positive=True)
    y = _scalar(y, 'y')
    width = _scalar(width, 'width', positive=True)
    ramp = _scalar(ramp, 'ramp')
    amplitude = _scalar(amplitude, 'amplitude')
    if amplitude < 0:
        raise ValueError('amplitude must be nonnegative.')
    if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)) or not 0 <= row < 41:
        raise ValueError('row must be an integer from 0 to 40.')
    if stage=='Patch sum':
        if profile is None:
            profile=screen_profile(s,width,ramp,amplitude=amplitude)
        if profile['parameters'] != (s,width,ramp,41,amplitude):
            raise ValueError('The cached screen profile does not match the aperture parameters.')
        if not profile['y'][0] <= y <= profile['y'][-1]:
            raise ValueError('The selected point must lie on the displayed screen cut.')
    fig=Figure(figsize=(11.5,8.3) if stage=='Patch sum' else (11,4.1),layout='constrained')
    if stage=='Patch sum':
        (face,a),(b,detector)=fig.subplots(2,2)
    else:
        a,b=fig.subplots(1,2)
    if stage == 'Point response':
        from matplotlib.patches import Circle
        r=np.hypot(s,y)
        for radius in np.arange(.5,max(9,r+1),.5):
            a.add_patch(Circle((0,0),radius,fill=False,color=BLUE,alpha=.25,lw=.8))
        a.plot(0,0,'o',color=BLUE)
        a.annotate(r'Source $Q=(0,0,0)$',(0,0),xytext=(8,-24),textcoords='offset points',fontsize=9)
        a.plot(s,y,'o',color=ORANGE)
        a.annotate(r'$P=(0,y,s)$',(s,y),xytext=(-8,14),textcoords='offset points',ha='right',fontsize=10)
        a.plot([0,s],[0,y],color=ORANGE)
        extent=max(4.2,abs(y)+.7)
        a.set(xlim=(-.6,8.5),ylim=(-extent,extent),aspect='equal',
              xlabel=r'Propagation distance $s=-z$ / wavelength',ylabel='Transverse y / wavelength',
              title=f'Point-source response: R = {r:.2f} wavelengths')
        g=green(s,y)
        _phasor(b,[g]);b.set_title(f'Point response g (not U): |g| = {abs(g):.4f}\nPhase = {np.angle(g):+.2f} rad')
    elif stage == 'Boundary choice':
        sign={'Kirchhoff':0,'RS I':-1,'RS II':1}[choice]
        extent=max(5.,abs(y)+2)
        q=np.linspace(-extent,extent,501)
        g=green(s,y-q); d=green_normal(s,y-q)
        for lo,hi in [(-extent,-width/2),(width/2,extent)]:
            a.plot([0,0],[lo,hi],color=INK,lw=6,solid_capstyle='butt')
        a.plot([0,0],[-extent,extent],'--',color='0.6',lw=1)
        a.plot(s,y,'o',color=ORANGE)
        a.annotate(r'$P=(0,y,s)$ (+)',(s,y),xytext=(0,14),textcoords='offset points',ha='center',fontsize=9)
        if sign:
            a.plot(-s,y,'o',color=BLUE)
            a.text(-s,y+.3,"P' (+)" if sign>0 else "P' (-)",ha='center')
            a.plot([-s,0],[y,0],':',color=BLUE)
        a.plot([0,s],[0,y],':',color=ORANGE)
        a.plot(0,0,'o',color=INK);a.text(.15,-.5,r'$Q=(0,0,0)$',fontsize=9)
        a.set(xlim=(-8.5,8.5),ylim=(-extent,extent),xlabel=r'Normal coordinate $s=-z$ / wavelength',
              ylabel='Transverse coordinate / wavelength',title='Screen opening; dashed reference plane S')
        a.axvspan(0,8.5,color=GREEN,alpha=.045,zorder=-2)
        a.text(.98,.97,'Unknown U in the outgoing half-space\nHelmholtz equation + radiation condition',
               transform=a.transAxes,ha='right',va='top',fontsize=8.5,color=INK)
        # Mirror distances agree on S, while their normal derivatives oppose.
        b.plot(q,abs((1+sign)*g),color=BLUE,label=r'Response magnitude $|G|$')
        b.plot(q,abs((1-sign)*d)/(2*np.pi),'--',color=ORANGE,
               label=r'Normal-derivative magnitude $|\partial_nG|/k$')
        b.set(xlabel=r'Plane coordinate $\eta$ (eta) / wavelength; $\xi=0$',ylabel='Green-function weight magnitude',
              title={'Kirchhoff':'Kirchhoff: neither boundary weight vanishes',
                     'RS I':r'RS I: $G=0$; prescribe $U$',
                     'RS II':r'RS II: $\partial_nG=0$; prescribe $\partial_nU$'}[choice])
        b.legend(fontsize=9)
    else:
        q,terms=patches(s,y,width,ramp,amplitude=amplitude)
        extent=max(3.8,width/2+.5,abs(y)+.8)
        phase=np.angle(np.exp(1j*ramp*q))
        # A front view makes both integration coordinates visible. The side
        # view shows xi=x=0; its orange phasor represents the entire selected row.
        image=face.imshow(np.tile(phase[:,None],(1,len(q))),origin='lower',
                          extent=(-width/2,width/2,-width/2,width/2),
                          cmap='twilight',vmin=-np.pi,vmax=np.pi,aspect='equal')
        face.axhline(q[row],color=ORANGE,lw=2)
        face.axvline(0,color='white',ls=':',lw=1)
        face.plot(0,q[row],'o',mfc='white',mec=ORANGE,ms=6)
        face.set(xlabel=r'$\xi$ (xi) / wavelength',ylabel=r'$\eta$ (eta) / wavelength')
        face.set_title(f'Reference plane S: amplitude = {amplitude:.2f}\n'+
                       rf'Selected row: $\eta={q[row]:+.2f}\lambda$',fontsize=10)
        fig.colorbar(image,ax=face,orientation='horizontal',pad=.05,shrink=.8,
                     label='Prescribed phase (rad)',ticks=[-np.pi,0,np.pi])
        # The front view already shows the complete sampled boundary field.
        # A dense stack of phase markers here looked like a closed screen.
        # Keep the aperture clear and mark only the selected integration point.
        a.plot(0,q[row],'o',mfc='white',mec=ORANGE,ms=8,mew=2,zorder=4)
        a.plot(s,y,'o',color=GREEN)
        a.annotate(r'$P=(0,y,s)$',(s,y),xytext=(-4,14),textcoords='offset points',ha='right',fontsize=9)
        a.annotate(r'$Q=(0,\eta,0)$',(0,q[row]),xytext=(8,-24),textcoords='offset points',fontsize=9)
        a.plot([0,s],[q[row],y],color=ORANGE)
        a.plot([0,s],[q[row],q[row]],':',color='0.55',lw=1)
        a.annotate('',(s,y),(s,q[row]),arrowprops=dict(arrowstyle='<->',color=BLUE,lw=1))
        a.text(s+.12,(y+q[row])/2,r'$y-\eta$',fontsize=9,color=BLUE,va='center')
        a.annotate('',(s,-.84*extent),(0,-.84*extent),arrowprops=dict(arrowstyle='<->',color=INK,lw=1))
        a.text(s/2,-.95*extent,r'$s$',ha='center',fontsize=10)
        for lo,hi in [(-extent,-width/2),(width/2,extent)]:
            a.plot([0,0],[lo,hi],color=INK,lw=5,solid_capstyle='butt')
        for edge in (-width/2,width/2):
            a.plot([0,.16],[edge,edge],color=INK,lw=2)
        a.annotate('',(-.3,width/2),(-.3,-width/2),
                   arrowprops=dict(arrowstyle='<->',color=GREEN,lw=1))
        a.text(-.44,0,'A',ha='right',va='center',color=GREEN,fontsize=10)
        a.plot([s,s],[-extent,extent],'--',color='0.65',lw=1,zorder=-1)
        a.text(.98,.97,'Observation screen',ha='right',va='top',transform=a.transAxes,fontsize=9,color=INK)
        a.set(xlim=(-.9,9.2),ylim=(-extent,extent),xlabel=r'Propagation distance $s=-z$ / wavelength',
              ylabel='Transverse coordinate / wavelength',title=r'Side view: $x=\xi=0$')
        total=_phasor(b,terms,row)
        b.set_title(f'RS I: blue = row sums, green = U(P)\n|U(P)| = {abs(total):.3f}',fontsize=11)
        detector.plot(profile['y'],profile['intensity'],color=BLUE,lw=1.5,label='Calculated full RS I cut')
        detector.axhline(1,color='0.65',ls=':',lw=1,label='Fixed unit-incident reference')
        detector.plot([y],[abs(total)**2],'o',color=GREEN,ms=7,
                      label=f'P: I/I_ref = {abs(total)**2:.3f}')
        detector.set(xlim=(profile['y'][0],profile['y'][-1]),
                     ylim=(0,max(1.1,1.12*float(profile['intensity'].max()))),
                     xlabel=r'Observation position $y/\lambda$ at $x=0$',
                     ylabel=r'$I/I_{\rm ref}=|U|^2/|U_{\rm inc,ref}|^2$',
                     title='Numerical screen intensity (not measured data)')
        detector.legend(loc='upper center',bbox_to_anchor=(.5,-.23),fontsize=8,frameon=False)
        fig.propagation_data=dict(profile=profile, selected_y=y, selected_field=total,
                                  selected_intensity=abs(total)**2, row_contributions=terms,
                                  row_coordinates=q, selected_row=row, intensity_reference=1.)
    return fig


class PropagationPanel(widgets.VBox):
    """An owned widget tree with explicit, idempotent observer cleanup."""

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        for control in getattr(self, 'controls', {}).values():
            control.unobserve(self._update_callback, names='value')
        seen = set()

        def release(widget):
            if not isinstance(widget, widgets.Widget) or id(widget) in seen:
                return
            seen.add(id(widget))
            for child in getattr(widget, 'children', ()):
                release(child)
            release(getattr(widget, 'layout', None))
            release(getattr(widget, 'style', None))
            if widget is not self:
                widget.close()

        release(self)
        # Fixed-stage explorers do not render their stage selector or unrelated
        # controls. They still own those models and must close their comms.
        for control in getattr(self, 'controls', {}).values():
            release(control)
        self.data=None
        self._profile_cache=None
        super().close()


def propagation_explorer(stage=None):
    """All experiments, or one fixed stage beside its notebook explanation."""
    if stage is not None and stage not in STAGES:
        raise ValueError(f'stage must be one of {STAGES}.')
    fixed_stage = stage
    w = widgets
    from io import BytesIO

    stage=w.ToggleButtons(options=STAGES,value=fixed_stage or STAGES[0],description='Explore')
    if fixed_stage is not None:
        stage.layout.display='none'
        stage.disabled=True
    s=w.FloatSlider(value=3,min=.8,max=8,step=.1,description='Distance',continuous_update=False)
    y=w.FloatSlider(value=.5,min=-8,max=8,step=.05,description='P transverse',continuous_update=False)
    width=w.FloatSlider(value=3,min=.8,max=6,step=.1,description='Opening',continuous_update=False)
    ramp=w.FloatSlider(value=0,min=-4,max=4,step=.1,description='Phase slope',continuous_update=False)
    choice=w.Dropdown(options=['Kirchhoff','RS I','RS II'],value='RS I',description='Response')
    row=w.IntSlider(value=20,min=0,max=40,description='Select row',continuous_update=False)
    amplitude=w.FloatSlider(value=1,min=0,max=2,step=.1,description='Amplitude',continuous_update=False)
    # PNG has a portable image/png MIME type across notebook widget frontends.
    # Image(format='svg') may produce image/svg instead of image/svg+xml.
    caption=w.HTML();out=w.Image(format='png',layout=w.Layout(width='100%',max_width='100%'))
    controls=dict(stage=stage,s=s,y=y,width=width,ramp=ramp,choice=choice,row=row,amplitude=amplitude)
    notes={
        'Point response':'Equation link: g = exp(ikR)/(4 pi R). The green arrow is the point-source response g, not the propagated aperture field U. Horizontal and vertical components are its real and imaginary parts. Move P: distance changes magnitude and phase. Blue circles mark half-wavelength steps.',
        'Boundary choice':(
            'Move Distance or P transverse to change the geometry. '
            'Blue is |G|; dashed orange is |normal derivative G| / k, scaled for comparable units. '
            'These are weight magnitudes, not the real and imaginary parts of U(P). '
            'Choose RS I: blue vanishes on the entire plane, removing the unknown-slope term. '
            'Choose RS II: orange vanishes instead. Kirchhoff retains both weights. '
            'Mirror probes are mathematical devices, not physical reflections.'),
        'Patch sum':'Xi (\u03be) and eta (\u03b7) label source-plane coordinates. Each blue phasor combines one complete row of xi patches; orange selects a row; green adds all rows. The side view marks only the selected Q; all rows are still integrated. The field is prescribed as zero outside the square opening. This propagates the chosen field; it does not solve scattering by the material screen.'}


    def update(*_):
        if getattr(panel, '_closed', False):
            return
        active=stage.value
        width.layout.display='none' if active=='Point response' else ''
        choice.layout.display='' if active=='Boundary choice' else 'none'
        ramp.layout.display=row.layout.display='' if active=='Patch sum' else 'none'
        amplitude.layout.display='' if active=='Patch sum' else 'none'
        caption.value='<p>'+notes[active]+'</p>'
        if active=='Patch sum':
            caption.value=('<p><b>RS I patch sum:</b> each contribution is U_ref &times; K<sub>RS1</sub> &times; patch area. '
                'Move P transverse: the green sum and screen marker change together. '
                'Select row changes highlighting only. Intensity uses a fixed unit incident reference; '
                'the curve is never normalized to its own peak.</p>')+caption.value
        if active=='Boundary choice':
            assumptions={
                'Kirchhoff':'Opening: U = incident U and normal derivative U = incident normal derivative. Screen: both set to zero. This overprescribes the boundary data; the exact Kirchhoff theorem itself is consistent.',
                'RS I':'Opening: prescribe U = incident U. Screen: prescribe U = 0. The normal derivative of U is determined by the outgoing solution, not independently set to zero.',
                'RS II':'Opening: prescribe the incident normal derivative of U. Screen: prescribe normal derivative U = 0. The field U is determined by the outgoing solution, not independently set to zero.'}
            caption.value='<p><b>Assumed plane data: '+choice.value+'</b><br>'+assumptions[choice.value]+'</p>'+caption.value
        values={key:control.value for key,control in controls.items()}
        if active=='Patch sum':
            signature=(s.value,width.value,ramp.value,41,amplitude.value)
            if panel._profile_cache is None or panel._profile_cache['parameters']!=signature:
                panel._profile_cache=screen_profile(s.value,width.value,ramp.value,amplitude=amplitude.value)
            values['profile']=panel._profile_cache
        fig=picture(**values)
        panel.data=getattr(fig,'propagation_data',None)
        try:
            stream=BytesIO();fig.savefig(stream,format='png',dpi=130)
            out.value=stream.getvalue()
        finally:
            # Figures are unmanaged: release artists after each serialization.
            fig.clear()
    # Do not depend on CSS display:none for fixed-stage navigation. Some
    # notebook frontends showed the disabled selector despite that style.
    if fixed_stage=='Point response':
        children=[w.HBox([s,y]),caption,out]
    elif fixed_stage=='Boundary choice':
        children=[w.HBox([s,y]),w.HBox([width,choice]),caption,out]
    elif fixed_stage=='Patch sum':
        children=[w.HBox([s,y]),w.HBox([width,ramp]),w.HBox([row,amplitude]),caption,out]
    else:
        children=[stage,w.HBox([s,y]),w.HBox([width,choice]),w.HBox([ramp,row]),
                  w.HBox([amplitude]),caption,out]
    panel=PropagationPanel(children)
    panel._profile_cache=None
    panel.data=None
    panel.controls=controls
    panel._update_callback=update
    for control in controls.values():control.observe(update,names='value')
    try:
        update()
    except Exception:
        panel.close()
        raise
    return panel
