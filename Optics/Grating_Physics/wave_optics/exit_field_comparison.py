"""Interactive thick-exit -> equivalent object-plane comparison for lesson 9.3."""
from collections import OrderedDict
from dataclasses import replace
import io
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from .amplitude_grating_2d import absorbing_screen_configuration, AbsorbingComparison2DData
from .grating_intro import _close_widget_tree


def exit_to_object_comparison(solution, x):
    """Back-propagate all solved propagating modes; compare the same ideal modes.

    The material solve and physical exit reconstruction retain all solved modes.
    Only the object-plane comparison is restricted to propagating modes.
    Input amplitude is one at z=0; the equivalent outgoing field is not the
    actual incident-side total field returned by solution.reconstruct(x,0).
    """
    c=solution.configuration
    if c.My!=0 or c.ky_incident!=0 or c.fill_y!=1:
        raise ValueError('Use the y-invariant in-plane geometry.')
    if c.n_incident!=1 or c.n_output!=1 or c.n_inclusion!=1:
        raise ValueError('This comparison uses air exteriors and an air opening.')
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('x must be a finite nonempty one-dimensional grid.')
    m=solution.orders[:,0]
    keep = solution.output_kinds == 'propagating'
    basis=np.exp(1j*solution.kx[:,None]*x)
    beta=solution.beta_output[keep]
    with np.errstate(over='raise',invalid='raise'):
        inverse=np.exp(-1j*beta*c.thickness)
        equivalent_coefficients=solution.transmission[keep]*inverse
    if not np.isfinite(equivalent_coefficients).all():
        raise ValueError('The selected inverse propagation exceeds floating-point range.')
    # Binary t is defined at the top/object plane, with no incident h-phase factor.
    ideal_top_coefficients=c.fill_x*np.sinc(m[keep]*c.fill_x)
    wrapped=(x/c.period_x+.5)%1-.5
    aperture=(wrapped>=-c.fill_x/2)&(wrapped<c.fill_x/2)
    return dict(x=x,keep=keep,orders=m[keep],
        kx=solution.kx[keep], beta=solution.beta_output[keep],
        angles_deg=np.rad2deg(np.arcsin(np.clip(solution.kx[keep]/(2*np.pi/c.wavelength),-1.,1.))),
        exit_coefficients=solution.transmission[keep],
        efficiencies=solution.transmission_efficiency[keep],
        exit_field=solution.transmission@basis,
        equivalent_coefficients=equivalent_coefficients,
        equivalent_field=equivalent_coefficients@basis[keep],
        ideal_coefficients=ideal_top_coefficients,
        ideal_field=ideal_top_coefficients@basis[keep],
        ideal_exact=aperture*np.exp(1j*c.kx_incident*x),
        inverse_gain=float(np.max(abs(inverse))) if len(inverse) else 1.,
        forward_coefficients=equivalent_coefficients*np.exp(1j*beta*c.thickness))


def _geometry_and_orders(fig, grid, result, c, selected_index):
    """One periodic cell, incident/selected unit waves, all considered directions."""
    geometry=fig.add_subplot(grid[0,0])
    right=grid[0,1].subgridspec(2,1,height_ratios=(1.7,1.),hspace=.05)
    fan=fig.add_subplot(right[0]); table_axis=fig.add_subplot(right[1])
    d,h=c.period_x,c.thickness
    k0=2*np.pi/c.wavelength
    j=selected_index; selected=int(result['orders'][j])
    angle=np.deg2rad(result['angles_deg'][j])
    theta_i=np.arcsin(c.kx_incident/k0)
    xx=np.linspace(-d/2,d/2,260)
    zz=np.linspace(-h-.8*d,.65*d,260)
    X,Z=np.meshgrid(xx,zz)
    wave=np.full(X.shape,np.nan)
    above=Z>=0;below=Z<=-h
    beta_i=np.sqrt(k0*k0-c.kx_incident**2)
    wave[above]=np.cos(c.kx_incident*X[above]-beta_i*Z[above])
    if abs(result['exit_coefficients'][j])>1e-10:
        wave[below]=np.cos(result['kx'][j]*X[below]-result['beta'][j].real*(Z[below]+h)
                           +np.angle(result['exit_coefficients'][j]))
    geometry.imshow(wave,origin='lower',extent=[-d/2,d/2,zz[0],zz[-1]],
                    cmap='RdBu_r',vmin=-1,vmax=1,aspect='equal',interpolation='bilinear')
    opening=c.fill_x*d
    for start in (-d/2,opening/2):
        geometry.add_patch(Rectangle((start,-h),(d-opening)/2,h,facecolor='#222222',zorder=3))
    geometry.text(0,-h/2,'Air opening',ha='center',va='center',fontsize=8)
    geometry.axhline(0,color='#c96a12',ls='--',lw=1.5,zorder=4)
    geometry.axhline(-h,color='#1676a5',ls=':',lw=1.5,zorder=4)
    geometry.text(-.48*d,.02*d,r'$z_{obj}=0$',va='bottom',fontsize=8,
                  bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1),zorder=5)
    geometry.text(-.48*d,-h-.02*d,r'$z_{ref}=z_e=-h$',va='top',fontsize=8,
                  bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1),zorder=5)
    geometry.annotate('',xy=(0,.03*d),xytext=(-.42*d*np.sin(theta_i),.03*d+.42*d*np.cos(theta_i)),
        arrowprops=dict(arrowstyle='->',color='#222222',lw=2))
    geometry.annotate('',xy=(.5*d*np.sin(angle),-h-.05*d-.5*d*np.cos(angle)),xytext=(0,-h-.05*d),
        arrowprops=dict(arrowstyle='->',color='#27834a',lw=2))
    geometry.annotate('',xy=(-.48*d,.55*d),xytext=(.48*d,.55*d),
        arrowprops=dict(arrowstyle='<->',color='#333333',lw=1))
    geometry.text(0,.57*d,f'One period d={d:g} um',ha='center',fontsize=8)
    geometry.set(xlabel='x (um)',ylabel='z (um)',title=f'Geometry and unit wavefronts: selected m={selected:+d}')
    geometry.text(.5,-.18,'Incident wave above; selected outgoing wave below.\nUnit amplitudes here show direction, not order strength.',
        transform=geometry.transAxes,ha='center',va='top',fontsize=8)
    # Equal-radius rays; this is a direction diagram, not a power plot.
    fan.axhline(0,color='#777777',lw=1.5)
    fan.plot([0,0],[0,-1.12],':',color='#999999',lw=1)
    fan.annotate('',xy=(0,.03),xytext=(-.5*np.sin(theta_i),.5*np.cos(theta_i)),
                 arrowprops=dict(arrowstyle='->',color='#222222',lw=1.5))
    for order,theta in zip(result['orders'],np.deg2rad(result['angles_deg'])):
        active=order==selected; color='#27834a' if active else '#8b99ad'
        fan.annotate('',xy=(np.sin(theta),-np.cos(theta)),xytext=(0,0),
                     arrowprops=dict(arrowstyle='->',color=color,lw=2.2 if active else 1.))
        fan.text(1.13*np.sin(theta),-1.13*np.cos(theta),f'{order:+d}',ha='center',va='center',
                 color=color,fontsize=9)
    fan.text(-1.26,.05,'Exit plane',fontsize=8)
    fan.text(.97,.04,'+x',fontsize=8,ha='right')
    fan.text(.035,-1.22,'-z',fontsize=8)
    fan.set(xlim=(-1.3,1.3),ylim=(-1.35,.58),aspect='equal',title='All considered orders; selection in green')
    fan.axis('off')
    table_axis.axis('off')
    rows=[[f'{m:+d}',f'{angle:+.1f}',f'{power:.4f}']
          for m,angle,power in zip(result['orders'],result['angles_deg'],result['efficiencies'])]
    table=table_axis.table(cellText=rows,colLabels=['Order m','Angle (deg)','Thick power fraction'],
                           loc='center',cellLoc='center',colWidths=[.22,.3,.48])
    table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1.,1.15)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#dde1e5');cell.set_linewidth(.5)
        if row==0:cell.set_facecolor('#eef1f4')
        elif int(result['orders'][row-1])==selected:cell.set_facecolor('#e2f1e6')


def spatial_phase(field, mode='wrapped', amplitude_floor=1e-6):
    """Phase along x, unwrapped only within connected nonzero segments.

    Each unwrapped segment starts at its first valid wrapped phase. No phase
    fitting or alignment between models is performed. Undefined phase remains NaN.
    """
    if mode not in ('wrapped','unwrapped'):
        raise ValueError('Phase mode must be wrapped or unwrapped.')
    field=np.asarray(field,dtype=complex)
    values=np.angle(field)
    values=np.where(np.isclose(abs(values),np.pi,atol=1e-12,rtol=0),np.pi,values)
    valid=np.isfinite(field)&(abs(field)>amplitude_floor)
    values=np.where(valid,values,np.nan)
    if mode=='unwrapped':
        indices=np.flatnonzero(valid)
        for segment in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
            if len(segment):values[segment]=np.unwrap(values[segment])
        return values
    # Break the plotted line at wrapping jumps rather than drawing vertical edges.
    jumps=np.r_[False,abs(np.diff(values))>np.pi]
    return np.where(jumps,np.nan,values)


def comparison_png(result, configuration, *, show_exact=False, selected_order=0, phase_mode='wrapped'):
    fig=Figure(figsize=(11.5,11.1),dpi=105,layout='constrained')
    FigureCanvasAgg(fig)
    grid=fig.add_gridspec(3,2,height_ratios=(1.55,1.,1.),hspace=.12)
    j=int(np.flatnonzero(result['orders']==selected_order)[0])
    _geometry_and_orders(fig,grid,result,configuration,j)
    axes=np.array([[fig.add_subplot(grid[row,col]) for col in range(2)] for row in (1,2)])
    x=result['x']
    mode=np.exp(1j*result['kx'][j]*x)
    exit_mode=result['exit_coefficients'][j]*mode
    object_mode=result['equivalent_coefficients'][j]*mode
    def phase(field): return spatial_phase(field,phase_mode)
    for col,transform in enumerate((np.abs,phase)):
        axes[0,col].plot(x,transform(result['exit_field']),color='#1976b3',label='Physical exit: all solved orders')
        axes[0,col].plot(x,transform(exit_mode),'-.',color='#27834a',lw=1.,label=f'm={selected_order:+d}: thick contribution')
        axes[0,col].set_title(r'1. Physical exit at $z_{ref}=z_e=-h$')
        axes[1,col].plot(x,transform(result['equivalent_field']),color='#1976b3',label='Thick mask: all propagating orders')
        axes[1,col].plot(x,transform(result['ideal_field']),'--',color='#df791a',label='Ideal thin mask: same propagating orders')
        axes[1,col].plot(x,transform(object_mode),'-.',color='#27834a',lw=1.,label=f'm={selected_order:+d}: thick contribution')
        if show_exact:
            axes[1,col].plot(x,transform(result['ideal_exact']),':',color='#777777',lw=1.2,label='Exact ideal boundary (all harmonics)')
        axes[1,col].set_title(r'2. Compare at the object plane $z_{obj}=0$')
        for row in range(2):
            ax=axes[row,col]
            ax.set(xlabel='x (um)',ylabel='Amplitude |U|' if col==0 else f'{phase_mode.capitalize()} phase (rad)')
            ax.grid(alpha=.15)
            if col==1 and phase_mode=='wrapped':ax.set_ylim(-3.4,3.4)
            ax.legend(fontsize=7.,loc='best')
    buffer=io.BytesIO();fig.savefig(buffer,format='png')
    return buffer.getvalue()


def coefficient_comparison(result, phase_floor=1e-6):
    """Compare complex coefficients at z_obj=0 on the same propagating labels."""
    thick=result['equivalent_coefficients']
    thin=result['ideal_coefficients']
    valid=(abs(thick)>phase_floor)&(abs(thin)>phase_floor)
    delta=np.full(thick.shape,np.nan)
    delta[valid]=np.angle(thick[valid]*np.conj(thin[valid]))
    delta=np.where(np.isclose(abs(delta),np.pi,atol=1e-12,rtol=0),np.pi,delta)
    return dict(orders=result['orders'],thick=thick,thin=thin,
        magnitude_thick=abs(thick),magnitude_thin=abs(thin),
        difference=thick-thin,phase_difference=delta,phase_valid=valid)


def order_comparison_png(data, selected_order):
    fig=Figure(figsize=(11.5,4.2),dpi=105,layout='constrained')
    FigureCanvasAgg(fig)
    magnitude,phase=fig.subplots(1,2)
    m=data['orders']
    magnitude.vlines(m-.07,0,data['magnitude_thick'],color='#1976b3',lw=1.5)
    magnitude.plot(m-.07,data['magnitude_thick'],'o',color='#1976b3',label='Thick: back-propagated to object plane')
    magnitude.vlines(m+.07,0,data['magnitude_thin'],color='#df791a',lw=1.5)
    magnitude.plot(m+.07,data['magnitude_thin'],'x',color='#df791a',label='Ideal thin mask at object plane')
    magnitude.set(title=r'Coefficient magnitudes at $z_{obj}=0$',ylabel=r'$|C_m|$',ylim=(0,None))
    magnitude.legend(fontsize=8)
    phase.axhline(0,color='#999999',lw=.8)
    phase.plot(m,data['phase_difference'],'o',color='#444444',label='Thick minus thin, modulo 2 pi')
    valid_phase=data['phase_difference'][data['phase_valid']]
    extent=min(3.5,max(.1,1.2*float(np.max(abs(valid_phase))))) if len(valid_phase) else .1
    phase.set(title='Relative phase of each order',ylabel='Wrapped phase difference (rad)',
              ylim=(-extent,extent))
    if extent>np.pi:
        phase.set(yticks=[-np.pi,0,np.pi],yticklabels=['-pi','0','+pi'])
    phase.text(.5,-.23,'Phase omitted if either coefficient magnitude is <= 1e-6.',
               transform=phase.transAxes,ha='center',fontsize=8)
    for ax in (magnitude,phase):
        ax.axvline(selected_order,color='#27834a',alpha=.6,lw=1.5)
        ax.set(xlabel='Order m',xticks=m,xlim=(m[0]-.55,m[-1]+.55))
        ax.grid(alpha=.15)
    stream=io.BytesIO();fig.savefig(stream,format='png')
    return stream.getvalue()


def selected_order_html(data, selected_order):
    j=int(np.flatnonzero(data['orders']==selected_order)[0])
    def phase_text(c):return f'{np.angle(c):+.5f}' if abs(c)>1e-6 else 'undefined'
    rows=''.join(f'<tr><td>{name}</td><td>{c.real:+.6f}</td><td>{c.imag:+.6f}</td>'
                 f'<td>{abs(c):.6f}</td><td>{phase_text(c)}</td></tr>'
                 for name,c in [('Thick equivalent',data['thick'][j]),('Ideal thin mask',data['thin'][j])])
    phase=f'{data["phase_difference"][j]:+.5f} rad' if data['phase_valid'][j] else 'undefined'
    difference=data['difference'][j]
    return (f'<b>Selected order m={selected_order:+d}; both coefficients at z_obj=0.</b>'
        '<table style="border-collapse:collapse;width:100%;text-align:right">'
        '<tr><th>Model</th><th>Re C</th><th>Im C</th><th>|C|</th><th>arg C (rad)</th></tr>'
        +rows+'</table>'
        f'Complex difference (thick - thin): {difference:.6g}; '
        f'|difference|={abs(difference):.6g}; phase difference: {phase}.')


def observation_comparison(result, solution, distance=0.):
    """Forward the two object-plane predictions; verify against the solved output.

    The comparison and independent direct reconstruction use the same solved
    propagating subspace. The ideal prescription is at z=0, not at z=-h.
    """
    distance=float(distance)
    if not np.isfinite(distance) or distance<0:
        raise ValueError('Distance below the exit must be finite and nonnegative.')
    c=solution.configuration
    z_ref=-c.thickness-distance
    transfer=np.exp(1j*result['beta']*(c.thickness+distance))
    thick_coefficients=result['equivalent_coefficients']*transfer
    thin_coefficients=result['ideal_coefficients']*transfer
    # Query the backend at the observation plane, independently of the object replay.
    direct_coefficients=solution.at_reference_plane(z_ref)[result['keep']]
    basis=np.exp(1j*result['kx'][:,None]*result['x'])
    thick=thick_coefficients@basis
    thin=thin_coefficients@basis
    direct=direct_coefficients@basis
    reference_norm=np.linalg.norm(direct)
    replay_error=float(np.linalg.norm(thick-direct)/max(reference_norm,1e-15))
    thin_error=float(np.linalg.norm(thin-direct)/max(reference_norm,1e-15))
    return dict(x=result['x'],distance=distance,z_ref=z_ref,
        thick_field=thick,thin_field=thin,direct_field=direct,
        thick_coefficients=thick_coefficients,thin_coefficients=thin_coefficients,
        direct_coefficients=direct_coefficients,replay_error=replay_error,thin_error=thin_error)


def observation_comparison_png(data, phase_mode='wrapped'):
    fig=Figure(figsize=(12,4.3),dpi=105,layout='constrained')
    FigureCanvasAgg(fig)
    amplitude,phase,intensity=fig.subplots(1,3)
    x=data['x']
    for ax,transform,label in (
        (amplitude,np.abs,'Amplitude |U|'),
        (phase,lambda u:spatial_phase(u,phase_mode),f'{phase_mode.capitalize()} phase (rad)'),
        (intensity,lambda u:abs(u)**2,'Relative intensity |U| squared')):
        ax.plot(x,transform(data['thick_field']),color='#1976b3',lw=2,label='Thick via object plane')
        ax.plot(x,transform(data['thin_field']),'--',color='#df791a',label='Ideal thin mask from z=0')
        ax.plot(x,transform(data['direct_field']),'x',markevery=40,ms=4,color='#555555',label='Thick direct from exit')
        ax.set(xlabel='x (um)',ylabel=label)
        ax.grid(alpha=.15)
    amplitude.set_title('Amplitude')
    phase.set_title('Phase')
    intensity.set_title('Intensity')
    if phase_mode=='wrapped':phase.set_ylim(-3.4,3.4)
    amplitude.legend(fontsize=7.5,loc='best')
    fig.suptitle(f'Common observation plane: z_ref={data["z_ref"]:.3f} um; distance below exit={data["distance"]:.3f} um',fontsize=11)
    stream=io.BytesIO();fig.savefig(stream,format='png')
    return stream.getvalue()


class ExitFieldComparisonExplorer:
    """Solve on angle/thickness/harmonic changes; reuse for projection controls."""
    def __init__(self, configuration=None):
        import ipywidgets as widgets
        self.configuration=configuration or absorbing_screen_configuration(fill_y=1.,My=0,Mx=24)
        c=self.configuration
        # Validate the supported geometry before constructing the controls.
        if c.My!=0 or c.fill_y!=1 or c.ky_incident!=0 or c.n_incident!=1 or c.n_output!=1 or c.n_inclusion!=1:
            raise ValueError('Use a y-invariant absorbing screen with air openings and air exteriors.')
        self._cache=OrderedDict();self.solve_count=0;self._closed=False;self._updating=False
        self.x=np.linspace(-c.period_x/2,c.period_x/2,768,endpoint=False)
        style={'description_width':'initial'}
        self.angle=widgets.FloatSlider(value=float(np.rad2deg(np.arcsin(c.kx_incident/(2*np.pi/c.wavelength)))),
            min=-45,max=45,step=1,description='Incidence (deg)',continuous_update=False,style=style)
        self.thickness=widgets.SelectionSlider(options=sorted(set((.03,.08,.15,.3,.5,.75,1.,c.thickness))),
            value=c.thickness,description='Thickness h (um)',continuous_update=False,style=style)
        self.solve_cutoff=widgets.Dropdown(options=sorted(set((8,12,24,32,c.Mx))),value=c.Mx,
            description='Solve cutoff M',style=style)
        self.order=widgets.Dropdown(options=[0],value=0,description='Show order m',style=style)
        self.phase_mode=widgets.Dropdown(options=[('Wrapped','wrapped'),('Unwrapped','unwrapped')],
            value='wrapped',description='Spatial phase',style=style,layout=widgets.Layout(width='250px'))
        self.exact=widgets.Checkbox(value=False,description='Show exact ideal boundary',indent=False)
        self.image=widgets.Image(format='png',layout=widgets.Layout(width='100%',max_width='1250px'))
        self.readout=widgets.HTML()
        self.orders_image=widgets.Image(format='png',layout=widgets.Layout(width='100%',max_width='1250px'))
        self.orders_readout=widgets.HTML()
        self.distance=widgets.FloatSlider(value=0.,min=0.,max=5.,step=.05,
            description='Distance below exit (um)',continuous_update=False,style=style,
            layout=widgets.Layout(width='550px'))
        self.observation_image=widgets.Image(format='png',layout=widgets.Layout(width='100%',max_width='1250px'))
        self.observation_readout=widgets.HTML()
        self.tabs=widgets.Tab(children=[self.image,widgets.VBox([self.orders_image,self.orders_readout]),
            widgets.VBox([self.distance,self.observation_readout,self.observation_image])],selected_index=0)
        self.tabs.set_title(0,'Geometry and spatial fields')
        self.tabs.set_title(1,'Compare orders')
        self.tabs.set_title(2,'Observation plane')
        self.options=widgets.Accordion(children=[widgets.HBox([self.solve_cutoff,self.exact])],selected_index=None)
        self.options.set_title(0,'Optional: solve cutoff and exact ideal boundary')
        self.ui=widgets.VBox([widgets.HBox([self.thickness,self.angle,self.order,self.phase_mode],layout=widgets.Layout(flex_flow='row wrap')),
            self.options,self.readout,self.tabs])
        self._controls=(self.angle,self.thickness,self.order,self.phase_mode,self.solve_cutoff,self.exact,self.distance)
        for control in self._controls:control.observe(self._update,names='value')
        self.tabs.observe(self._update,names='selected_index')
        self._update()

    def _update(self, change=None):
        if self._closed or self._updating:return
        key=(self.angle.value,self.thickness.value,self.solve_cutoff.value)
        if key not in self._cache:
            c=replace(self.configuration,thickness=self.thickness.value,Mx=self.solve_cutoff.value,
                kx_incident=2*np.pi/self.configuration.wavelength*np.sin(np.deg2rad(self.angle.value)))
            self._cache[key]=AbsorbingComparison2DData(c).solution
            self.solve_count+=1
            if len(self._cache)>6:self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        self.solution=self._cache[key]
        self.result=exit_to_object_comparison(self.solution,self.x)
        c=self.solution.configuration;r=self.result
        self._updating=True
        try:
            previous=self.order.value
            self.order.options=[(f'{m:+d} ({theta:+.1f} deg)',int(m)) for m,theta in zip(r['orders'],r['angles_deg'])]
            self.order.value=previous if previous in r['orders'] else 0
        finally:
            self._updating=False
        j=int(np.flatnonzero(r['orders']==self.order.value)[0])
        self.readout.value=(f'<b>Exit: z_ref=z_e={-c.thickness:g} um. Object plane: z_obj=0.</b> '
            f'R={self.solution.reflectance:.3f}, T={self.solution.transmittance:.3f}, A={self.solution.absorption:.3f}.<br>'
            f'Object-plane curves use all solved propagating orders: {", ".join(str(m) for m in r["orders"])}. '
            f'Selected m={self.order.value:+d}: angle {r["angles_deg"][j]:+.1f} deg, '
            f'exit coefficient {r["exit_coefficients"][j]:.4g}, '
            f'object coefficient {r["equivalent_coefficients"][j]:.4g}. '
            'Selecting an order highlights its contribution; the field sums still use every considered order. '
            'No peak normalization or phase fitting is applied. '
            'Unwrapped phase follows x within each nonzero region; phase is hidden where |U| is below 1e-6.')
        is_fields=self.tabs.selected_index==0
        self.phase_mode.disabled=self.tabs.selected_index not in (0,2)
        self.distance.disabled=self.tabs.selected_index!=2
        self.exact.disabled=not is_fields
        if is_fields:
            self.image.value=comparison_png(r,c,show_exact=self.exact.value,selected_order=self.order.value,phase_mode=self.phase_mode.value)
        elif self.tabs.selected_index==1:
            self.readout.value=(f'<b>Order comparison at z_obj=0; h={c.thickness:g} um; '
                f'incidence {self.angle.value:+g} deg.</b> '
                f'All {len(r["orders"])} propagating orders are compared using the same incident amplitude and phase. '
                'No peak normalization or phase fitting is applied.')
            self.order_result=coefficient_comparison(r)
            self.orders_image.value=order_comparison_png(self.order_result,self.order.value)
            self.orders_readout.value=selected_order_html(self.order_result,self.order.value)
        elif self.tabs.selected_index==2:
            self.observation_result=observation_comparison(r,self.solution,self.distance.value)
            observed=self.observation_result
            self.readout.value=(f'<b>Compare both predictions at z_ref={observed["z_ref"]:.3f} um.</b> '
                f'Both use all {len(r["orders"])} solved propagating orders. '
                'The direct check uses the same propagating orders, not the complete near field.')
            self.observation_readout.value=(
                f'Thick replay check, relative complex-field difference: {observed["replay_error"]:.2e}. '
                'This checks the change of reference plane, not scattering convergence.<br>'
                f'Ideal thin vs direct thick, relative complex-field difference: {observed["thin_error"]:.4g}. '
                'No amplitude scaling or phase fitting is applied.<br>'
                f'Selected m={self.order.value:+d} at z_ref: '
                f'C thick={observed["thick_coefficients"][j]:.5g}; '
                f'C thin={observed["thin_coefficients"][j]:.5g}.')
            self.observation_image.value=observation_comparison_png(observed,self.phase_mode.value)

    def close(self):
        if self._closed:return
        self._closed=True
        for control in self._controls:control.unobserve(self._update,names='value')
        self.tabs.unobserve(self._update,names='selected_index')
        _close_widget_tree(self.ui);self._cache.clear()
