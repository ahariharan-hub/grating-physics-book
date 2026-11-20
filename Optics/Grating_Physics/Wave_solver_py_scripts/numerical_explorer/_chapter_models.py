"""Headless experiment adapters using wave_optics numerical APIs.

These functions assemble teaching experiments; no Qt or pyplot is imported.
All lengths are um unless the lesson explicitly uses medium wavelengths.
"""
from functools import lru_cache
import numpy as np
from wave_optics.waves import wavevector, plane_wave_phasor, field_from_phasor
from wave_optics.diffraction import diffraction_orders
from wave_optics.scalar_propagation import propagate_field, transfer_function


def curves(title, x, *series, xlabel='x (um)', ylabel='Field'):
    return dict(kind='curves', title=title, x=np.asarray(x), series=series,
                xlabel=xlabel, ylabel=ylabel)


def image(title, x, y, values, signed=False, ylabel='s = -z (um)'):
    return dict(kind='image', title=title, x=x, y=y, values=values,
                signed=signed, xlabel='x (um)', ylabel=ylabel)


def complex_curves(title, x, u, xlabel='x (um)'):
    return curves(title, x, ('Re U', u.real), ('Im U', u.imag), ('|U|', abs(u)), xlabel=xlabel)


def _orders(p):
    return diffraction_orders(p['wavelength'], p['period'], p['angle'], p['azimuth'],
                              orders=np.arange(-8, 9), selected_order=p['order'])


@lru_cache(maxsize=12)
def _absorber(thickness, loss, duty, cutoff):
    from wave_optics.absorbing_grating import AbsorbingGratingConfiguration, solve_absorbing_grating
    return solve_absorbing_grating(AbsorbingGratingConfiguration(
        thickness=thickness, kappa=loss, clear_width=1.5*duty, M=cutoff))


def compute(spec, p):
    """Compute one lesson, returning named plot panels and an explicit readout."""
    model = spec.model
    lam, d, s = p['wavelength'], p['period'], p['distance']
    k = 2*np.pi/lam
    phase = np.deg2rad(p['phase'])
    x = np.linspace(-8, 8, 801)
    panels = []
    status = ''
    values = {}
    if model == 'helmholtz':
        from wave_optics.scalar_theory import helmholtz_trial
        v = helmholtz_trial(x, wavelength=lam, spatial_ratio=p['ratio'])
        panels = [complex_curves('Trial field', x, v['field']),
                  curves('Helmholtz residual / k²', x, ('Re residual / k²', v['residual'].real/k**2))]
        status = f"max |residual|/k² = {np.max(abs(v['residual']))/k**2:.4g}"
        values = v
    elif model == 'families':
        from wave_optics.scalar_theory import scalar_solution
        from wave_optics.helmholtz_families import helmholtz_family, gaussian_beam
        a = np.linspace(-5, 5, 181); z = np.linspace(.05, 10, 181)
        xx, zz = np.meshgrid(a, z); points = np.stack((xx, np.zeros_like(xx), zz), -1)
        names = ('plane', 'standing', 'spherical', 'cylindrical', 'bessel', 'multipole', 'evanescent', 'Gaussian (paraxial)')
        name = names[p['family']]
        fn = scalar_solution if p['family'] < 3 else helmholtz_family
        u = gaussian_beam(points, wavelength=lam) if p['family']==7 else fn(points, wavelength=lam, kind=name)
        panels = [image(f'{name}: real field', a, z, field_from_phasor(u, p['time'], 2*np.pi), True, 'z (um), +z view'),
                  image(f'{name}: amplitude', a, z, abs(u), ylabel='z (um), +z view')]
        status = 'Source points excluded; amplitudes of different families use their backend normalizations.'
    elif model == 'boundary':
        from wave_optics.scalar_theory import boundary_family
        v = boundary_family(x, wavelength=lam, backward_fraction=p['backward'], phase_deg=p['phase'])
        panels = [complex_curves('Field (boundary at z=0)', x, v['field'], 'z (um)'),
                  complex_curves('Normal derivative', x, v['derivative'], 'z (um)')]
        status = f"U(0) = exp(i phi); U'(0) = {1j*k*(1-2*p['backward'])*np.exp(1j*phase):.3g}"
    elif model in ('material', 'coefficients', 'mask_compare'):
        sol = _absorber(p['thickness'], p['loss'], p['duty'], p['cutoff'])
        c = sol.configuration
        depth = s if model == 'coefficients' else 0.
        reference = np.exp(2j*np.pi*c.thickness/c.wavelength)
        co = sol.at_reference_plane(-c.thickness-depth)/reference
        a = np.linspace(-c.period, c.period, 801)
        u = sol.reconstruct(a, [-c.thickness-depth])[0]/reference
        ideal = p['duty']*np.sinc(sol.orders*p['duty'])
        panels = [complex_curves('Physical output / incident exit reference', a, u),
                  curves('Order magnitudes', sol.orders, ('Physical |C|', abs(co)),
                         ('Ideal exit mask |t_m|', abs(ideal)), xlabel='Order m', ylabel='Magnitude'),
                  curves('Coefficient phase (tiny orders omitted)', sol.orders,
                         ('Physical phase', np.where(abs(co)>1e-6, np.angle(co), np.nan)), xlabel='Order m', ylabel='Phase (rad)'),
                  curves('Transmitted normal flux', sol.orders, ('Efficiency', sol.transmission_efficiency), xlabel='Order m', ylabel='Incident power fraction')]
        if model == 'mask_compare':
            mask = (abs((a/c.period+.5)%1-.5) < p['duty']/2).astype(float)
            panels[0]['series'] += (('Ideal binary mask at exit', mask),)
        status = (f'lambda0 = 0.633 um; pitch = 1.5 um; normal incidence; '
                  f'R={sol.reflectance:.5f}, T={sol.transmittance:.5f}, A={sol.absorption:.5f}; '
                  f'R+T+A={sol.energy_balance:.6f}. M={p["cutoff"]}; flux balance does not prove convergence. '
                  f'Output depth={depth:g} um. Ideal mask coefficients are at the exit.')
        values = dict(coefficients=co, solution=sol, reference=reference)
    elif model in ('wave', 'wave3d', 'detector'):
        vector = wavevector(lam, p['angle'], p['azimuth'] if model != 'detector' else 0.)
        if model == 'detector':
            points = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
            u = plane_wave_phasor(points, vector, p['amplitude'], phase)
            panels = [curves('Instantaneous field and phasor amplitude', x,
                             ('Real field', field_from_phasor(u, p['time'], 2*np.pi)), ('|U|', abs(u))),
                      curves('Detector intensity', x, ('|U|²', abs(u)**2), ylabel='Relative intensity')]
        else:
            a = np.linspace(-5, 5, 161); b = np.linspace(0, 10, 161)
            aa, bb = np.meshgrid(a, b)
            points = (np.stack((aa, bb-5, np.zeros_like(aa)), -1) if model == 'wave3d'
                      else np.stack((aa, np.zeros_like(aa), bb), -1))
            u = plane_wave_phasor(points, vector, phase=phase)
            panels = [image('Real field at selected time', a, b-5 if model == 'wave3d' else b,
                            field_from_phasor(u, p['time'], 2*np.pi), True,
                            'y (um), z=0' if model == 'wave3d' else 'z (um), +z view'),
                      dict(kind='vectors', title='Wavevector direction (+z)', vectors=np.array([vector/k]), labels=['k/k'])]
        status = f'k = ({vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}) rad/um; exp(-i omega t).'
        values = dict(field=u, vector=vector)
    elif model == 'beam':
        from wave_optics.helmholtz_families import gaussian_beam
        if p['width']/lam < 1.5:
            raise ValueError('Use beam width / wavelength >= 1.5 for this paraxial illustration.')
        points = np.column_stack((x, np.zeros_like(x), np.full_like(x, s)))
        u = gaussian_beam(points, wavelength=lam, waist_ratio=p['width']/lam)
        fx = np.fft.fftshift(np.fft.fftfreq(len(x), x[1]-x[0]))
        spectrum = abs(np.fft.fftshift(np.fft.fft(np.exp(-(x/p['width'])**2))))
        panels = [curves('Transverse intensity', x, ('Plane wave', np.ones_like(x)), ('Gaussian beam', abs(u)**2), ylabel='Relative intensity'),
                  curves('Finite input angular spectrum (sampled window)', fx, ('Magnitude / peak', spectrum/spectrum.max()), xlabel='fx (cycles/um)', ylabel='Normalized magnitude')]
        status = 'Gaussian waist at z=0; peak scaling follows the paraxial backend. Spectrum is cropped to the sampled input window.'
    elif model == 'array':
        from wave_optics.order_array import opening_positions, far_field_amplitude
        if p['width'] > d:
            raise ValueError('Slit width must not exceed the period.')
        positions = opening_positions(d, p['count'])
        angles = np.linspace(-85, 85, 1801)
        envelope = np.sinc(p['width']*np.sin(np.deg2rad(angles))/lam)
        af = far_field_amplitude(angles, positions, lam, 0.)
        a = np.linspace(positions.min()-d, positions.max()+d, 2001)
        mask = np.any(abs(a[:, None]-positions) <= p['width']/2, axis=1)
        panels = [curves('Prescribed finite openings', a, ('Transmission', mask.astype(float))),
                  curves('Far-field shape, normal incidence', angles, ('Single slit', abs(envelope)**2),
                         ('Slit envelope × array', abs(envelope*af)**2), xlabel='Angle (deg)', ylabel='Normalized intensity')]
        status = 'Fraunhofer shape with unit central maximum; excludes absolute power, material scattering and finite-distance propagation.'
    elif model == 'surface':
        mask = (abs((x/d+.5)%1-.5)<p['duty']/2).astype(float)
        u = mask*plane_wave_phasor(x[:, None], [k*np.sin(np.deg2rad(p['angle']))])
        panels = [curves('Surface amplitude', x, ('Binary mask', mask)), complex_curves('Transmitted surface field', x, u)]
        status = 'Prescribed thin mask; clear fraction is independent of illumination angle.'
    elif model in ('orders', 'coordinates', 'conical', 'lattice', 'pupil'):
        from wave_optics.lattice_grating import lattice_orders, collect_orders
        if model in ('lattice', 'pupil'):
            labels = np.array([(m,n) for m in range(-6,7) for n in (range(-6,7) if model=='lattice' else [0])])
            inc = wavevector(lam, p['angle'], p['azimuth'])[:2]/k
            v = lattice_orders(np.diag([d, p['pitch_y']]), labels, wavelength0=lam, incident=inc)
            if model == 'pupil':
                v = collect_orders(v, p['na']); xy = v['sigma']; selected = v['collected']
            else:
                xy = v['components'][:, :2].real; selected = v['status']=='propagating'
            panels = [dict(kind='scatter', title='Sigma collection coordinates' if model=='pupil' else '2D reciprocal lattice: direction cosines',
                           xy=xy, selected=selected, labels=[f'{m},{n}' for m,n in labels], circle=True)]
            status = f'{np.sum(selected)} '+('collected' if model=='pupil' else 'propagating')+' candidates; geometry alone supplies no amplitudes. Outgoing -z, air.'
            values = v
        else:
            v = _orders(p); xy = np.column_stack((v['a'],v['b']))
            panels = [dict(kind='scatter', title='Transverse direction cosines', xy=xy,
                           selected=v['valid'], labels=[str(m) for m in v['labels']], circle=True,
                           highlight=int(np.flatnonzero(v['labels']==p['order'])[0]))]
            if model in ('conical', 'orders'):
                vectors = v['directions'].copy(); vectors[:,2] *= -1
                panels.append(dict(kind='vectors', title='Outgoing order directions (-z)', vectors=vectors,
                                   labels=[str(m) for m in v['allowed']]))
            else:
                panels += [curves('Transverse wavevectors', v['labels'], ('kx', k*v['a']), ('ky', k*v['b']), xlabel='Order m', ylabel='rad/um'),
                           curves('Spatial frequencies', v['labels'], ('fx', v['a']/lam), ('fy', v['b']/lam), xlabel='Order m', ylabel='cycles/um')]
                for panel in panels[1:]: panel['highlight']=p['order']
            status = f'Delta kx={2*np.pi/d:.4g} rad/um; Delta alpha={lam/d:.4g}; ky/k={v["incident"][1]:.4g}. Candidate labels -8..8.'
            values = v
    elif model in ('matching', 'paths'):
        ai = np.sin(np.deg2rad(p['angle'])); ao = ai+p['order']*lam/d
        a = np.linspace(0,d,401)
        cycles = (p['component']-ai)*d/lam
        trial = np.exp(1j*k*(p['component']-ai)*a)
        matched = np.exp(2j*np.pi*p['order']*a/d)
        if model == 'matching':
            panels = [curves('Relative phase along one period', a, ('Trial cycles', (p['component']-ai)*a/lam),
                             ('Selected order cycles', p['order']*a/d), ylabel='Unwrapped cycles'),
                      dict(kind='phasors', title='Relative phasor over one period', paths=[('Trial',trial),('Selected order',matched)])]
        else:
            panels = [curves('Projected optical path across surface displacement', a,
                             ('Incident projection', ai*a), ('Selected outgoing projection', ao*a),
                             ('Difference', (ao-ai)*a), ylabel='Path projection (um)')]
        status = f'Selected alpha_out={ao:.4g}; path difference={p["order"]*lam:.4g} um = {p["order"]} wavelengths. '+('No real outgoing ray.' if abs(ao)>1 else 'Real outgoing direction.')
        if model == 'matching': status += f' Trial mismatch: {cycles:.3f} cycles per pitch.'
        values = dict(alpha_out=ao, path_difference=(ao-ai)*d)
    elif model == 'evanescent':
        from wave_optics.evanescent import evanescent_order, order_phasor, decay_envelope
        v = evanescent_order(lam, d, p['angle'], p['order'])
        a=np.linspace(-3,3,241); depth=np.linspace(0,s,201)
        u=order_phasor(a[None,:], -depth[:,None], v)
        panels = [image('Selected order: real field', a, depth, field_from_phasor(u,p['time'],2*np.pi), True),
                  curves('Amplitude and normal flux', depth, ('Amplitude envelope',decay_envelope(depth,v)),
                         ('Normal flux / unit plane wave', np.full_like(depth,v['q'].real/k)), xlabel='s = -z (um)', ylabel='Relative value')]
        status = f"{v['status']}; q=-kz={v['q']:.4g} rad/um; decay length={v['decay_length']:.4g} um. Unit surface amplitude prescribed; no scattering solve."
        values = dict(v, field=u)
    elif model == 'averaging':
        tau=np.linspace(0,p['cycles'],1601)
        u=field_from_phasor(p['amplitude']*np.exp(1j*phase),tau,2*np.pi)
        # Exact integral from t=0; avoids a time-step-dependent detector result.
        mean=np.empty_like(tau);mean[0]=u[0]**2
        t=tau[1:];mean[1:]=p['amplitude']**2*(.5+(np.sin(4*np.pi*t-2*phase)+np.sin(2*phase))/(8*np.pi*t))
        panels=[curves('Instantaneous real field',tau,('u',u),xlabel='t / T'),
                curves('Exposure-averaged mean square',tau,('Mean square',mean),('Infinite exposure',np.full_like(tau,p['amplitude']**2/2)),xlabel='Exposure / T',ylabel='Mean square')]
        status=f'Final mean square={mean[-1]:.5g}; long-exposure relative phasor intensity A²={p["amplitude"]**2:.5g}. Optical time is normalized; wavelength does not change this curve.'
        values=dict(mean=mean, time=tau)
    elif model == 'interference':
        from wave_optics.evanescent import evanescent_order, order_phasor
        from wave_optics.mutual_coherence import averaged_intensity, fringe_visibility
        m=p['order']
        if m==0: raise ValueError('Choose a nonzero second order; the first order is m=0.')
        v=evanescent_order(1.,d,0.,m)
        if v['status']!='propagating': raise ValueError('Choose a period larger than |m| um so both selected orders propagate.')
        u0=np.exp(2j*np.pi*s)*np.ones_like(x)
        u1=p['balance']*np.exp(1j*phase)*order_phasor(x,-np.full_like(x,s),v)
        delta=(v['kx']*x+(v['q'].real-2*np.pi)*s+phase)
        intensity=averaged_intensity(1.,p['balance']**2,p['coherence'],delta)
        vis=fringe_visibility(1.,p['balance']**2,p['coherence'])
        panels=[curves('Detector intensity',x,('Partially coherent total',intensity),('Intensity sum',np.full_like(x,1+p['balance']**2)),ylabel='Relative intensity'),
                curves('Prescribed component real fields',x,('Order 0',u0.real),('Second order',u1.real))]
        status=f'lambda=1 um; orders 0 and {m}; fringe period={d/abs(m):.4g} um; visibility={vis:.4g}. Coefficients prescribed.'
        values=dict(intensity=intensity, visibility=vis, fringe_period=d/abs(m))
    elif model == 'overlap':
        from wave_optics.gaussian_order_beams import GaussianOrderConfiguration,GaussianOrderExperiment
        exp=GaussianOrderExperiment(GaussianOrderConfiguration(waist=p['waist']))
        a=np.linspace(-20,25,901)
        v=exp.sample(a,np.array([s]),coefficients=(1.,p['balance']*np.exp(1j*phase)))
        panels=[curves('Finite-beam detector intensities',a,('Coherent sum',v['intensity'][0]),
                       ('Intensity sum',v['intensities'][:,0].sum(axis=0)),ylabel='Relative intensity'),
                curves('Individual beams',a,('Order 0',v['intensities'][0,0]),('Order 1',v['intensities'][1,0]),ylabel='Relative intensity')]
        status=f'Line amplitude-overlap diagnostic={float(v["overlap"][0]):.4g}; not mutual coherence or full-plane overlap. Beam directions fixed; lambda=1 um, pitch=4 um.'
        values=v
    elif model == 'green':
        # Same outgoing Helmholtz Green response as the Chapter 8 backend.
        from wave_optics.propagation_interactive import green
        u=green(s,x,k=k)
        panels=[complex_curves('Outgoing point Green response on a detector line',x,u),
                curves('Unwrapped path phase',x,('kR / 2pi',np.hypot(s,x)/lam),ylabel='Cycles')]
        status='G=exp(ikR)/(4 pi R); the response has inverse-length units. No aperture field or intensity normalization is prescribed.'
    elif model == 'rs':
        from wave_optics.propagation_interactive import patches,screen_profile
        a=np.linspace(-8,8,161);ramp=phase/p['width']
        v=screen_profile(s,p['width'],ramp,positions=a,count=p['patches'])
        q,contributions=patches(s,0.,p['width'],ramp,count=p['patches'])
        path=np.r_[0j,np.cumsum(contributions)]
        panels=[curves('Square-aperture RS I detector cut',a,('Intensity',v['intensity']),ylabel='Relative intensity'),
                dict(kind='phasors',title='On-axis row contributions: cumulative complex sum',paths=[('Patch rows',path)]),
                complex_curves('On-axis contributions by boundary row',q,contributions,'Boundary eta (wavelengths)')]
        status=f'{p["patches"]} × {p["patches"]} midpoint patches; lambda=1; on-axis field={path[-1]:.4g}. Increase patch count to check quadrature.'
        values=dict(v,on_axis=path[-1])
    elif model == 'weyl':
        from wave_optics.weyl_reconstruction import weyl_field
        from wave_optics.weyl_cap import reference_field
        a=np.linspace(-p['width'],p['width'],161)
        points=np.column_stack((a,np.zeros_like(a),np.full_like(a,s)))
        u=weyl_field(points,order=p['weyl_order']);ref=reference_field(points)
        error=np.linalg.norm(u-ref)/np.linalg.norm(ref)
        panels=[curves('Spherical field reconstructed by Weyl quadrature',a,('Reference |G|',abs(ref)),('Weyl |G|',abs(u)),ylabel='Amplitude'),
                curves('Complex-field error',a,('Re difference',(u-ref).real),('Im difference',(u-ref).imag))]
        status=f'Relative complex-field L2 error={error:.4g}; lambda=1; source=(0,0,-1.5) in (x,y,s). Increase quadrature order and inspect complex error.'
        values=dict(field=u,reference=ref,error=error)
    elif model == 'kernels':
        from wave_optics.propagation_kernels import kernel_values
        v=kernel_values(x,p['width']/2,distance=s,wavelength=lam)
        names=('Full RS I','Fresnel','Fraunhofer')
        panels=[curves('Reduced spatial-kernel amplitude',x,*[(name,abs(u)) for name,u in zip(names,v['reduced'])],ylabel='Reduced amplitude'),
                curves('Unwrapped phase error against full RS I',x,*[(name,u) for name,u in zip(names[1:],v['phase_error'][1:])],ylabel='Phase error (rad)')]
        status='Source patch xi=width/2, eta=0. Same carrier removed from all methods; section of a 3D kernel, not a full aperture integral.'
        values=v
    elif model == 'multipliers':
        u=np.linspace(-1.25,1.25,801);fx=u/lam
        ha=transfer_function(fx,0.,lam,s,'asm');hf=transfer_function(fx,0.,lam,s,'fresnel')
        chosen=[transfer_function(p['component']/lam,0.,lam,s,m) for m in ('asm','fresnel')]
        panels=[curves('Fourier multiplier magnitude',u,('ASM',abs(ha)),('Fresnel',abs(hf)),xlabel='kx / k',ylabel='|H|'),
                dict(kind='phasors',title=f'Selected kx/k={p["component"]:g}: carrier removed',paths=[(n,np.array([0j,complex(h)*np.exp(-1j*k*s)])) for n,h in zip(('ASM','Fresnel'),chosen)])]
        status=f'Selected |H_ASM|={abs(complex(chosen[0])):.4g}; |H_F|={abs(complex(chosen[1])):.4g}. No incident spectrum prescribed.'
        values=dict(asm=ha,fresnel=hf)
    elif model == 'aperture':
        n=p['samples'];dx=p['window']/n
        if dx>lam/4: raise ValueError('Increase FFT samples or reduce window: require dx <= wavelength/4.')
        a=(np.arange(n)-n//2)*dx;xx,yy=np.meshgrid(a,a)
        boundary=((abs(xx)<=p['width']/2)&(abs(yy)<=p['width']/2)).astype(complex)
        asm=propagate_field(boundary,dx,dx,lam,s,'asm')['field']
        fresnel=propagate_field(boundary,dx,dx,lam,s,'fresnel')['field']
        error=np.linalg.norm(asm-fresnel)/np.linalg.norm(asm)
        border=np.zeros((n,n),bool);border[:8]=True;border[-8:]=True;border[:,:8]=True;border[:,-8:]=True
        edge=float(np.sum(abs(asm[border])**2)/np.sum(abs(asm)**2))
        panels=[image('Prescribed square aperture |U0|',a,a,abs(boundary),ylabel='y (um)'),
                image('ASM output intensity',a,a,abs(asm)**2,ylabel='y (um)'),
                curves('Detector centre line',a,('ASM',abs(asm[n//2])**2),('Fresnel',abs(fresnel[n//2])**2),ylabel='Relative intensity'),
                curves('Output centre-line real field',a,('ASM',asm[n//2].real),('Fresnel',fresnel[n//2].real))]
        status=f'dx={dx:.4g} um; full-window complex-field difference={error:.4g}; border power fraction={edge:.3g}. Border check alone does not exclude wraparound; vary window and samples independently.'
        values=dict(field=asm,boundary=boundary,fresnel=fresnel,error=error,edge_fraction=edge)
    else:
        raise ValueError(f'Unknown lesson model {model!r}')
    return dict(panels=panels,status=status,values=values)


def draw(result, figure, axes=None, colorbars=None):
    """Render static panels or reuse a playground's axes without removing controls."""
    from matplotlib.patches import Circle
    standalone = axes is None
    if standalone:
        figure.clear()
    panels=result['panels']; columns=min(2,len(panels)); rows=(len(panels)+columns-1)//columns
    for i,p in enumerate(panels):
        projection='3d' if p['kind']=='vectors' else 'rectilinear'
        if standalone:
            ax=figure.add_subplot(rows,columns,i+1,projection=projection)
        else:
            ax=axes[i]
            if ax.name != projection:
                subplot=ax.get_subplotspec();ax.remove()
                ax=figure.add_subplot(subplot,projection=projection);axes[i]=ax
        ax.set_title(p['title'], fontsize=10)
        kind=p['kind']
        if kind=='curves':
            for name,values in p['series']: ax.plot(p['x'],values,label=name,lw=1.7)
            if 'highlight' in p: ax.axvline(p['highlight'],color='black',ls=':',label='Selected order')
            ax.set(xlabel=p['xlabel'],ylabel=p['ylabel']);ax.legend(fontsize=8);ax.grid(alpha=.2)
        elif kind=='image':
            data=p['values']; limit=max(float(np.nanmax(abs(data))),1e-12)
            artist=ax.imshow(data,origin='lower',extent=[p['x'][0],p['x'][-1],p['y'][0],p['y'][-1]],
                             aspect='equal',cmap='RdBu_r' if p['signed'] else 'viridis',
                             vmin=-limit if p['signed'] else 0,vmax=limit)
            ax.set(xlabel=p['xlabel'],ylabel=p['ylabel'])
            bar=figure.colorbar(artist,ax=ax,shrink=.8)
            if colorbars is not None: colorbars.append(bar)
        elif kind=='vectors':
            for v,label in zip(p['vectors'],p['labels']):
                ax.quiver(0,0,0,*v,arrow_length_ratio=.12);ax.text(*v,label,fontsize=8)
            ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),xlabel='kx/k',ylabel='ky/k',zlabel='kz/k')
            ax.set_box_aspect((1,1,1))
        elif kind=='scatter':
            xy=p['xy']; chosen=p['selected']
            for mask,color,label in ((chosen,'#0072B2','Propagating / collected'),(~chosen,'#D55E00','Other candidates')):
                ax.scatter(xy[mask,0],xy[mask,1],c=color,label=label,s=24)
            if p['circle']: ax.add_patch(Circle((0,0),1,fill=False,color='black',lw=1))
            if 'highlight' in p:
                selected_xy=xy[p['highlight']]
                ax.scatter(*selected_xy,s=100,facecolors='none',edgecolors='black',linewidths=1.6,label='Selected order')
            # Limit annotation clutter while retaining every candidate in the plot.
            for v,label in zip(xy,p['labels']):
                if np.max(abs(v))<1.7: ax.annotate(label,v,xytext=(3,3),textcoords='offset points',fontsize=7)
            ax.set_aspect('equal');ax.set(xlabel='Transverse x coordinate',ylabel='Transverse y coordinate');ax.legend(fontsize=8);ax.grid(alpha=.2)
        elif kind=='phasors':
            for label,path in p['paths']:
                ax.plot(path.real,path.imag,'o-',ms=2,label=label)
                ax.plot([path[-1].real],[path[-1].imag],'o',ms=6)
            ax.axhline(0,c='gray',lw=.5);ax.axvline(0,c='gray',lw=.5)
            ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='Real',ylabel='Imaginary');ax.legend(fontsize=8)
    if not standalone:
        for ax in axes[len(panels):]: ax.set_visible(False)
