#!/usr/bin/env python3
"""Explore Appendix 1's scalar Fourier Modal Method (FMM).

Edit the parameter block below, then run:
    python scalar_fmm_grating.py

Requires NumPy, SciPy and Matplotlib. Uses the existing wave_optics backend;
keep this script within the repository. No files are written.
All lengths are micrometres; wavelength is the VACUUM wavelength lambda0.
Normal incidence, exp(-i omega t), transmission toward negative z.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


# EDIT THESE PARAMETERS, then run the script.
# Supported geometry: one rectangular, lossless scalar grating layer,
# periodic along x, illuminated at normal incidence.
wavelength = 0.193       # Vacuum wavelength lambda0 (um)
period = 0.6             # Grating pitch (um)
thickness = 0.130        # Layer thickness h (um)
duty_cycle = 2/3         # Ridge width / period; 0 < fraction < 1
n_incident = 1.0         # Refractive index above the grating
n_output = 1.0           # Refractive index below the grating
n_ridge = 1.5           # Ridge refractive index (real, positive)
n_groove = 1.0           # Groove refractive index (real, positive)
M =  24                  # Retain orders -M,...,+M, including evanescent harmonics
output_depth = thickness+0.01        # Distance below the grating exit (um)
convergence_cutoffs = [] # Optional: e.g. [2, 8, 24, 48]; highest M is a reference
interactive = True      # False: calculate and print without opening a window
# Plot limits only crop the view; every retained order still enters the solve.
# None = automatic limits; (lower, upper) fixes a range. Either bound can be None.
propagation_depth_wavelengths = 20.  # Large-view depth / vacuum wavelength
propagation_half_width_wavelengths = 10.  # Large-view half-width / vacuum wavelength
propagation_order = 1         # Order shown separately in the large propagation view
show_field_directions = True  # Overlay propagating order directions on the field map
order_xlim = (-7.5, 7.5)       # Shared by magnitude, phase and diffraction efficiency
magnitude_ylim = (0., None)
phase_ylim = (-np.pi-.15, np.pi+.15)   # Radians
efficiency_ylim = (0., None)
output_xlim = None            # Detector profile x, in um
field_signed_ylim = None     # Re/Im effective transmission at exit
field_amplitude_ylim = (0., None)
field_phase_ylim = (-np.pi-.15, np.pi+.15)
output_ylim = (0., None)      # Detector profile relative intensity
thin_fft_samples = 4096  # Samples per period; increase to check mask FFT convergence
plot_linewidth = 2.6          # Curve thickness; no physical parameter is changed
plot_marker_size = 6.5
fmm_color = '#0072B2'          # Blue: physical FMM
mask_color = '#D55E00'         # Orange: ideal binary mask / FFT
fmm_imag_color = '#009E73'     # Green: FMM imaginary part
mask_imag_color = '#CC79A7'    # Purple: ideal imaginary part
plot_backend = 'QtAgg'   # Separate interactive window (Spyder); None uses IDE default


def load_solver():
    """Locate the course package relative to the script, independent of cwd."""
    for base in Path(__file__).resolve().parents:
        for candidate in (base, base / 'Optics' / 'Grating_Physics'):
            if (candidate / 'wave_optics' / 'scalar_grating.py').is_file():
                sys.path.insert(0, str(candidate))
                from wave_optics.scalar_grating import ScalarGratingConfiguration, solve_scalar_grating
                return ScalarGratingConfiguration, solve_scalar_grating
    raise FileNotFoundError('Keep this script in the Py_Code_Projects repository, near Grating_Physics/wave_optics.')


Configuration, solve = load_solver()


def incident_exit_reference(configuration):
    """Unit incident plane wave continued to z=-h, with exp(-i omega t)."""
    c=configuration
    return np.exp(2j*np.pi/c.wavelength*c.n_incident*c.thickness)


def wrapped_phase(values,floor=1e-6,*,break_wraps=False):
    """Principal phase; zeros are undefined, equivalent +/-pi shown as +pi."""
    values=np.asarray(values)
    phase=np.angle(values)
    # Canonical endpoints avoid numerical +/-pi flips for real Fourier coefficients.
    phase=np.where(np.isclose(abs(phase),np.pi,atol=1e-12,rtol=0),np.pi,phase)
    phase=np.where(np.isclose(phase,0.,atol=1e-12,rtol=0),0.,phase)
    phase=np.where(abs(values)>floor,phase,np.nan)
    if break_wraps:
        phase[np.flatnonzero(abs(np.diff(phase))>np.pi)+1]=np.nan
    return phase


def style_profile_axes(axes):
    """Readable curves and discrete stems; never alter the plotted data."""
    for ax in axes:
        for line in ax.lines:
            line.set_linewidth(plot_linewidth)
            if line.get_marker() not in (None,'None','', ' '):
                line.set_markersize(plot_marker_size)
                line.set_markeredgewidth(1.4)
        for container in ax.containers:
            if hasattr(container,'stemlines'):
                container.stemlines.set_linewidth(plot_linewidth*.8)
        ax.tick_params(labelsize=9)


def output_profile(solution, depth, x):
    """Return full and propagating-only fields AFTER the same scattering solve."""
    z = -solution.configuration.thickness - depth
    coefficients = solution.at_reference_plane(z)
    basis = np.exp(1j * solution.kx[:, None] * x)
    propagating = (solution.beta_output.real > 0) & (solution.beta_output.imag == 0)
    full = coefficients @ basis
    selected = coefficients[propagating] @ basis[propagating]
    return z, coefficients, propagating, full, selected


def thin_fft_comparison(solution, depth, x, samples=None):
    """FFT of the prescribed complex exit mask, propagated with the same M.

    Ideal binary amplitude: groove=1, ridge=0, using the existing binary-mask
    backend. A common incident carrier references the mask to exit z=-h.
    No material scattering, phase fitting or power normalization. Equal exterior indices only.
    FFT sample convergence is separate from FMM basis convergence.
    """
    from wave_optics.binary_transmission import binary_transmission_field, binary_transmission_coefficients
    from wave_optics.amplitude_grating import field_fourier_coefficients

    c = solution.configuration
    if c.n_incident != c.n_output:
        raise ValueError('The binary amplitude-mask comparison requires equal exterior indices.')
    samples = thin_fft_samples if samples is None else samples
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples <= 2*c.M:
        raise ValueError('thin_fft_samples must be an integer greater than 2*M (at least 2).')
    if samples < 2:
        raise ValueError('thin_fft_samples must be at least 2.')
    # Midpoint sampling over one period; helper accounts for the coordinate origin.
    grid = -c.period/2 + (np.arange(samples)+.5)*c.period/samples
    carrier=incident_exit_reference(c)
    mask=carrier*binary_transmission_field(grid,c.period,c.duty_cycle,0.,1.)
    exit_coefficients = field_fourier_coefficients(mask, grid, c.period, solution.orders)
    coefficients = exit_coefficients*np.exp(1j*solution.beta_output*depth)
    field = coefficients @ np.exp(1j*solution.kx[:, None]*np.asarray(x))
    efficiency = solution.beta_output.real/(2*np.pi/c.wavelength*c.n_incident)*abs(exit_coefficients)**2
    analytic=carrier*binary_transmission_coefficients(solution.orders,c.duty_cycle,0.,1.)
    error=float(np.max(abs(exit_coefficients-analytic)))
    return coefficients, field, efficiency, error


def convergence(configuration, limits, current=None):
    """Compare common propagating EXIT coefficients; highest M is only a reference."""
    limits = sorted(set(limits))
    if not limits:
        raise ValueError('Provide at least one convergence cutoff.')
    if any(M < 0 for M in limits):
        raise ValueError('Convergence cutoffs must be nonnegative.')
    solutions = {M: current if current is not None and current.configuration == replace(configuration, M=M)
                 else solve(replace(configuration, M=M)) for M in limits}
    reference = solutions[limits[-1]]
    propagating = (reference.beta_output.real > 0) & (reference.beta_output.imag == 0)
    orders = reference.orders[propagating]
    if any(M < np.max(np.abs(orders), initial=0) for M in limits):
        raise ValueError('Each convergence cutoff must include all reference propagating orders.')
    target = reference.transmission[orders + reference.configuration.M]
    scale = max(np.max(abs(target), initial=0), np.finfo(float).eps)
    rows = []
    print('\nBasis convergence: complex transmitted coefficients at z=-h; same physical inputs.')
    print('Highest M is a numerical reference, not an exact solution. Power balance alone is insufficient.')
    print(' M     N      max |delta C| / max |C_ref|        |R+T-1|')
    for M, trial in solutions.items():
        error = float(np.max(abs(trial.transmission[orders + M] - target), initial=0) / scale)
        rows.append((M, 2*M+1, error, abs(trial.power_balance - 1)))
        print(f'{M:3d}  {2*M+1:4d}              {error:.6e}                 {rows[-1][3]:.3e}')
    return rows


def describe(solution, depth):
    c = solution.configuration
    print(f'\nScalar FMM: lambda0={c.wavelength:g} um; d={c.period:g} um; h={c.thickness:g} um')
    print(f'Indices: incident={c.n_incident:g}, output={c.n_output:g}, ridge={c.n_ridge:g}, groove={c.n_groove:g}')
    print(f'Ridge fraction={c.duty_cycle:g}; M={c.M}; N={2*c.M+1}; output plane z={-c.thickness-depth:g} um')
    print('F_m(z): coupled Fourier harmonics inside the layer; C_m(z_ref)=F_m(z_ref).')
    print('Q is the coupling matrix, not an eigenmode. Its eigenvectors mix harmonics into layer modes.')
    print(f'R={sum(solution.reflection_efficiency):.8f}; T={sum(solution.transmission_efficiency):.8f}; R+T={solution.power_balance:.10f}')
    print(f'Boundary residual={solution.boundary_residual:.3e}; scaled matching condition={solution.condition_number:.3e}')
    print('Fields are relative scalar units. Efficiencies include normal-flux factors.')
    print('Displayed exit coefficients and transmission use the same continued-incident reference exp(i*k_i*h).')
    if c.n_incident == c.n_output:
        print('FFT reference: binary amplitude mask, groove=1 and ridge=0, at z=-h.')
        print('FMM still solves the specified dielectric layer; the binary mask is a different ideal object.')
        print(f'FFT samples per period={thin_fft_samples}; retained orders match FMM. Check both sampling and M convergence.')
    else:
        print('Thin mask / FFT unavailable: existing comparison requires equal exterior indices.')


def overlay_field_directions(solution, ax, x_bounds, z_bounds):
    """Directions at the interfaces, not rays delimiting separate order beams."""
    import matplotlib.patheffects as pe
    c = solution.configuration
    x_center = .5*(x_bounds[0]+x_bounds[1])
    width = x_bounds[1]-x_bounds[0]
    artists = []
    for side, sign, index, beta, kinds, power, origin_z, space, color in (
            ('R',1,c.n_incident,solution.beta_incident,solution.incident_kinds,
             solution.reflection_efficiency,0.,z_bounds[1],'lime'),
            ('T',-1,c.n_output,solution.beta_output,solution.output_kinds,
             solution.transmission_efficiency,-c.thickness,-c.thickness-z_bounds[0],'cyan')):
        allowed=np.flatnonzero(kinds=='propagating')
        # Less room above the layer: keep only the strongest three reflected arrows.
        # Figure 2 retains every direction, including weak orders.
        limit=3 if side=='R' else 7
        if allowed.size>limit:
            allowed=allowed[np.argsort(power[allowed])[-limit:]]
        length=min(.22*width,.63*space)
        if length<=0:
            continue
        k_medium=2*np.pi/c.wavelength*index
        for j in allowed:
            dx=solution.kx[j]/k_medium
            dz=sign*beta[j].real/k_medium
            tip=(x_center+length*dx,origin_z+length*dz)
            arrow=ax.annotate('',xy=tip,xytext=(x_center,origin_z),
                arrowprops=dict(arrowstyle='-|>',color=color,lw=1.3,
                                linestyle='--' if power[j]<1e-6 else '-',mutation_scale=9))
            arrow.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.7,foreground='black'),pe.Normal()])
            label=ax.annotate(f'{side}{int(solution.orders[j]):+d}',xy=tip,
                xytext=(0 if dx==0 else (8 if dx>0 else -8),(4 if sign>0 else -12) if dx==0 else 4*sign),
                textcoords='offset points',ha='center' if dx==0 else ('left' if dx>0 else 'right'),
                va='bottom' if sign>0 else 'top',
                color=color,fontsize=6.5,
                bbox=dict(facecolor='black',edgecolor='none',alpha=.55,pad=.3))
            artists.extend((arrow,label))
    # Small displaced arrow shows the normally incident direction without hiding R0.
    if z_bounds[1]>0:
        x_inc=x_bounds[0]+.12*width
        artists.append(ax.annotate('',xy=(x_inc,.12*z_bounds[1]),
            xytext=(x_inc,.80*z_bounds[1]),
            arrowprops=dict(arrowstyle='-|>',color='white',lw=1.5)))
    return artists


def draw(solution, depth, figure, axes, colorbars):
    """Reuse one figure; removing old colorbars prevents repeated-apply axes growth."""
    for colorbar in colorbars:
        colorbar.remove()
    colorbars.clear()
    for ax in axes:
        ax.clear()
    c = solution.configuration
    # One periodic cell: the groove is centered at d/2, with ridge halves at the edges.
    # Keep the backend coordinate origin and complex-coefficient phases unchanged.
    x = np.linspace(0., c.period, 401)
    z = np.linspace(-c.thickness - max(depth, c.wavelength), .5*c.wavelength, 241)
    field = solution.reconstruct(x, z)
    extent = [x[0], x[-1], z[0], z[-1]]
    image = axes[0].imshow(c.refractive_index(x, z), origin='lower', aspect='auto', extent=extent, cmap='viridis')
    colorbars.append(figure.colorbar(image, ax=axes[0], label='Real material index'))
    axes[0].set(title=r'Material $n(x,z)$: one period', xlabel='x (um)', ylabel='z (um)')
    image = axes[1].imshow(abs(field)**2, origin='lower', aspect='auto', extent=extent, cmap='inferno')
    colorbars.append(figure.colorbar(image, ax=axes[1], label='Relative |U| squared'))
    for interface in (0, -c.thickness):
        axes[1].axhline(interface, color='cyan', lw=.6)
    axes[1].set(title=r'Full field $|U(x,z)|^2$ + directions', xlabel='x (um)', ylabel='z (um)')
    axes[1].set_aspect('equal',adjustable='box')
    if show_field_directions:
        overlay_field_directions(solution,axes[1],(x[0],x[-1]),(z[0],z[-1]))
    z_ref, coefficients, propagating, full, selected = output_profile(solution, depth, x)
    phase = np.where(abs(coefficients) > 1e-6, np.angle(coefficients), np.nan)
    draw_order_coefficients(solution,figure,(axes[2],axes[3]))
    axes[4].bar(solution.orders-.18, solution.reflection_efficiency, width=.36,color='0.5',label='FMM reflected')
    axes[4].bar(solution.orders+.18, solution.transmission_efficiency, width=.36,color=fmm_color,label='FMM transmitted')
    axes[4].set(title=r'Diffraction efficiencies $R_m$, $T_m$', xlabel='Order m', ylabel='Diffraction efficiency')
    axes[4].legend(fontsize=8)
    axes[5].plot(x, abs(full)**2,color=fmm_color,label='FMM full |U| squared')
    axes[5].plot(x, abs(selected)**2,'--',color='0.4',alpha=.8,label='Propagating part (same solve)')
    axes[5].set(title=r'Output intensity $I(x,z_D)=|U(x,z_D)|^2$', xlabel='x (um)', ylabel='Relative intensity')
    if c.n_incident == c.n_output:
        thin_c, thin_field, thin_efficiency, fft_error = thin_fft_comparison(solution, depth, x)
        axes[4].plot(solution.orders, thin_efficiency, 's', mfc='none',color=mask_color,
                     label='Binary mask / FFT transmitted')
        axes[4].legend(fontsize=7)
        axes[5].plot(x,abs(thin_field)**2,'--',color=mask_color, label='Binary amplitude / FFT')
        axes[5].set_title(r'Output intensity $|U(x,z_D)|^2$'+f' (z={z_ref:.3g} um)',fontsize=10)
        axes[5].text(.02,.02,f'Mask FFT coefficient error: {fft_error:.2g}',
                     transform=axes[5].transAxes, va='bottom', fontsize=7)
    else:
        axes[5].text(.02,.98,'Thin mask comparison needs equal exterior indices',
                     transform=axes[5].transAxes, va='top', fontsize=7)
    axes[5].legend(fontsize=7)
    if order_xlim is not None:
        axes[4].set_xlim(order_xlim)
    if efficiency_ylim is not None:
        axes[4].set_ylim(efficiency_ylim)
    if output_xlim is not None:
        axes[5].set_xlim(output_xlim)
    if output_ylim is not None:
        axes[5].set_ylim(output_ylim)
    draw_transmission_profiles(solution,(axes[6],axes[7],axes[8]))
    style_profile_axes((axes[4],axes[5]))
    figure.suptitle(f'Scalar FMM | M={c.M}, N={2*c.M+1} | R+T={solution.power_balance:.8f}\n'
                   r'Exit reference: $a_m=C_m/U_{\mathrm{inc,ref}}$, $t_{\mathrm{eff}}=U_{\mathrm{exit}}/U_{\mathrm{inc,ref}}$',fontsize=11)
    figure.canvas.draw_idle()
    return coefficients, phase


def draw_order_coefficients(solution, figure, axes):
    """Compare exit coefficients by order, sharing the incident phase reference."""
    for ax in axes:
        ax.clear()
    c = solution.configuration
    mask = np.ones(solution.orders.shape,dtype=bool)
    orders = solution.orders[mask]
    carrier=incident_exit_reference(c)
    fmm = solution.transmission[mask]/carrier
    marker,stems,_=axes[0].stem(orders,abs(fmm),linefmt='-',markerfmt='o',basefmt=' ',
                              label='FMM dielectric grating')
    marker.set_color(fmm_color);stems.set_color(fmm_color)
    axes[1].plot(orders,wrapped_phase(fmm),'o',color=fmm_color,
                 label='FMM dielectric grating')
    if c.n_incident == c.n_output:
        coefficients, _, _, error = thin_fft_comparison(solution,0.,[0.])
        fft = coefficients[mask]/carrier
        marker, _, _ = axes[0].stem(orders,abs(fft),linefmt='--',markerfmt='s',
                                    basefmt=' ',label='FFT binary amplitude mask')
        marker.set_markerfacecolor('none')
        marker.set_color(mask_color)
        axes[0].containers[-1].stemlines.set_color(mask_color)
        # Do not attach phase to a sampling residual at a true mask zero.
        axes[1].plot(orders,wrapped_phase(fft,max(1e-6,5*error)),
                     's',color=mask_color,mfc='none',label='FFT binary amplitude mask')
    else:
        axes[0].text(.02,.98,'Thin mask comparison requires equal exterior indices',
                     transform=axes[0].transAxes,va='top',fontsize=8)
    axes[0].set(title=r'Order coefficients $a_m=C_m/U_{\mathrm{inc,ref}}$',xlabel='Order m',ylabel=r'$|a_m|$')
    axes[1].set(title=r'Order phase $\arg a_m$ at $z=-h$',xlabel='Order m',ylabel=r'$\arg a_m$ (rad)',
                yticks=[-np.pi,0,np.pi],yticklabels=[r'$-\pi$','0',r'$\pi$'])
    for ax,limits in zip(axes,(magnitude_ylim,phase_ylim)):
        if order_xlim is not None:
            ax.set_xlim(order_xlim)
        if limits is not None:
            ax.set_ylim(limits)
        ax.legend(fontsize=7);ax.grid(True,alpha=.2)
    style_profile_axes(axes)


def draw_order_directions(solution, figure, axes):
    """Show actual outgoing wavevectors; arrows are directions, not beam edges."""
    for ax in axes:
        ax.clear()
    c = solution.configuration
    ax, table_ax = axes
    rows = []
    for side, sign, index, beta, kinds, power, color in (
            ('R', 1, c.n_incident, solution.beta_incident, solution.incident_kinds,
             solution.reflection_efficiency, 'tab:red'),
            ('T', -1, c.n_output, solution.beta_output, solution.output_kinds,
             solution.transmission_efficiency, 'tab:blue')):
        mask = kinds == 'propagating'
        k_medium = 2*np.pi/c.wavelength*index
        count = np.count_nonzero(mask)
        for j in np.flatnonzero(mask):
            dx = solution.kx[j]/k_medium
            dz = sign*beta[j].real/k_medium
            angle = np.degrees(np.arctan2(dx,abs(dz)))
            eta = float(power[j])
            m = int(solution.orders[j])
            # Equal arrow lengths encode orientation only; weak powers use dashed lines.
            ax.annotate('',xy=(dx,dz),xytext=(0,0),arrowprops=dict(
                arrowstyle='-|>',color=color,lw=1.5,linestyle='--' if eta<1e-6 else '-'))
            if count <= 11:
                ax.text(1.11*dx,1.11*dz,f'{side}{m:+d}',color=color,
                        ha='center',va='center',fontsize=9)
            rows.append((side,m,dx,dz,angle,eta))
    ax.axhline(0,color='0.5',lw=2)
    ax.axvline(0,color='0.8',lw=.7)
    circle=np.linspace(0,2*np.pi,361)
    ax.plot(np.cos(circle),np.sin(circle),color='0.85',lw=.8)
    ax.annotate('',xy=(-.07,.12),xytext=(-.07,.75),
                arrowprops=dict(arrowstyle='-|>',color='black',lw=2))
    ax.text(-.14,.77,'Incident',ha='right',fontsize=9)
    ax.text(.98,.96,'Reflected (+z)',transform=ax.transAxes,ha='right',color='tab:red')
    ax.text(.98,.04,'Transmitted (-z)',transform=ax.transAxes,ha='right',color='tab:blue')
    ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),aspect='equal',
           xlabel=r'$k_x/k_{\mathrm{medium}}$',ylabel=r'$k_z/k_{\mathrm{medium}}$',
           title=r'Outgoing wavevectors $\mathbf{k}_m$')
    ax.text(.5,-.17,'Direction diagram; interfaces collapsed to z=0.\n'
            'Arrows are not beam boundaries. Dashed: power < 1e-6.',
            transform=ax.transAxes,ha='center',fontsize=8)
    # Avoid unreadable tables for very large periods; all directions remain drawn.
    visible = rows if len(rows)<=20 else sorted(rows,key=lambda row:row[5],reverse=True)[:20]
    cells=[[f'{side}{m:+d}',f'{angle:+.2f}',f'{eta:.5f}']
           for side,m,dx,dz,angle,eta in visible]
    table_ax.axis('off')
    table=table_ax.table(cellText=cells,colLabels=['Order','Angle (deg)','Normal power'],
                         cellLoc='center',bbox=[.06,.20,.90,.72])
    table.auto_set_font_size(False);table.set_fontsize(9)
    table_ax.set_title(r'Order angles $\theta_m$ from outgoing normal toward $+x$')
    if len(rows)>20:
        table_ax.text(.5,.95,'Table: 20 strongest orders',ha='center',fontsize=8)
    e_r=np.count_nonzero(solution.incident_kinds=='evanescent')
    e_t=np.count_nonzero(solution.output_kinds=='evanescent')
    grazing=np.count_nonzero(solution.incident_kinds=='grazing')+np.count_nonzero(solution.output_kinds=='grazing')
    table_ax.text(.05,.10,f'Evanescent harmonics: {e_r} above, {e_t} below.\n'
                  'They decay normally; no real propagation angle is assigned.\n'
                  f'Grazing components: {grazing}; zero normal power.',fontsize=9,va='top')
    figure.suptitle(f'FMM scattering | lambda0={c.wavelength:g} um | d={c.period:g} um | '
                   f'n_i={c.n_incident:g}, n_t={c.n_output:g}',fontsize=11)
    figure.canvas.draw_idle()
    return rows


def draw_transmission_profiles(solution,axes):
    """Like-for-like complex transmission at exit z=-h, relative to one carrier.

    The ideal mask has t=0/1. FMM supplies t_eff=U_out/U_inc_ref, where
    U_inc_ref is the incident plane wave continued to the same exit plane.
    This fixed reference removes no fitted or material-dependent phase.
    """
    from wave_optics.binary_transmission import binary_transmission_field
    for ax in axes:
        ax.clear()
    c=solution.configuration
    x=np.linspace(0.,c.period,401)
    z_ref=-c.thickness
    field=solution.reconstruct(x,[z_ref])[0]
    incident_reference=incident_exit_reference(c)
    effective=field/incident_reference
    ideal=binary_transmission_field(x,c.period,c.duty_cycle,0.,1.).real

    phase=wrapped_phase(effective,break_wraps=True)
    mask_phase=np.where(ideal>0,0.,np.nan)
    if c.duty_cycle in (0.,1.):
        mask_x=np.array([0.,c.period]);mask_y=np.full(2,1-c.duty_cycle)
    else:
        edge=.5*c.duty_cycle*c.period
        mask_x=np.array([0.,edge,edge,c.period-edge,c.period-edge,c.period])
        mask_y=np.array([0.,0.,1.,1.,0.,0.])
    axes[0].plot(x,effective.real,color=fmm_color,label='FMM Re t_eff')
    axes[0].plot(x,effective.imag,color=fmm_imag_color,label='FMM Im t_eff')
    axes[0].plot(mask_x,mask_y,'--',color=mask_color,label='Ideal Re t (0/1)')
    axes[0].plot([0,c.period],[0,0],'--',color=mask_imag_color,label='Ideal Im t (0)')
    axes[0].set(title=r'Exit $t(x)$ versus $t_{\mathrm{eff}}(x)$',ylabel='Re / Im transmission')
    axes[1].plot(x,abs(effective),color=fmm_color,label='FMM |t_eff|')
    axes[1].plot(mask_x,mask_y,'--',color=mask_color,label='Ideal |t| (0/1)')
    axes[1].set(title=r'Magnitude $|t(x)|$, $|t_{\mathrm{eff}}(x)|$',ylabel='Amplitude ratio')
    axes[2].plot(x,phase,color=fmm_color,label='FMM arg t_eff')
    axes[2].plot(x,mask_phase,'--',color=mask_color,label='Ideal arg t (open only)')
    axes[2].set(title=r'Phase $\arg t(x)$, $\arg t_{\mathrm{eff}}(x)$',ylabel='Phase relative to incident (rad)',
                yticks=[-np.pi,0,np.pi],yticklabels=[r'$-\pi$','0',r'$\pi$'])
    for i,(ax,limits) in enumerate(zip(axes,(field_signed_ylim,field_amplitude_ylim,field_phase_ylim))):
        ax.set_xlabel('x (um)')
        ax.set_xlim((0,c.period) if output_xlim is None else output_xlim)
        if limits is not None:
            ax.set_ylim(limits)
        ax.grid(True,alpha=.25)
        ax.legend(fontsize=6.5,ncol=2 if i==0 else 1)
    style_profile_axes(axes)
    return dict(x=x,z_ref=z_ref,field=field,incident_reference=incident_reference,
                effective_transmission=effective,ideal_transmission=ideal,
                phase=phase,ideal_phase=mask_phase)


def draw_propagation_view(solution, figure, axes, colorbars):
    """Large, equal-scale signed-field view of transmitted propagating orders.

    Reuses exit coefficients from one FMM solve. No finite-beam envelope or
    aperture is introduced; the repeated grating still produces plane waves.
    """
    import matplotlib.patheffects as pe
    for bar in colorbars:
        bar.remove()
    colorbars.clear()
    for ax in axes:
        ax.clear()
    c=solution.configuration
    depth=float(propagation_depth_wavelengths)
    half_width=float(propagation_half_width_wavelengths)
    if not np.isfinite(depth) or not np.isfinite(half_width) or min(depth,half_width)<=0:
        raise ValueError('Propagation-view depth and half-width must be finite and positive.')
    # At least 16 samples per shortest propagating wavelength in either direction.
    ns=max(401,int(np.ceil(16*depth*c.n_output))+1)
    nx=max(401,int(np.ceil(32*half_width*c.n_output))+1)
    if ns*nx>8_000_000:
        raise ValueError('Propagation-view grid too large; reduce its wavelength ranges.')
    distance=np.linspace(0,depth*c.wavelength,ns)
    x=np.linspace(-half_width*c.wavelength,half_width*c.wavelength,nx)
    mask=solution.output_kinds=='propagating'
    orders=solution.orders[mask]
    coefficients=solution.transmission[mask]
    transverse=np.exp(1j*solution.kx[mask,None]*x)
    longitudinal=np.exp(1j*distance[:,None]*solution.beta_output[mask])
    field=(longitudinal*coefficients)@transverse
    selected_order=propagation_order if propagation_order in orders else 0
    if selected_order not in orders:
        selected_order=int(orders[0])
    selected_index=int(np.flatnonzero(orders==selected_order)[0])
    selected=coefficients[selected_index]*np.outer(longitudinal[:,selected_index],transverse[selected_index])
    extent=[0,depth,-half_width,half_width]
    amplitude=max(float(np.max(abs(field.real))),1e-12)
    image=axes[0].imshow(field.real.T,origin='lower',extent=extent,aspect='equal',
                         cmap='RdYlBu_r',vmin=-amplitude,vmax=amplitude,interpolation='nearest')
    colorbars.append(figure.colorbar(image,ax=axes[0],shrink=.8,label='Re U / incident amplitude'))
    axes[0].set_title(r'All orders: $\operatorname{Re}U=\operatorname{Re}\sum_m U_m$')
    selected_amplitude=abs(coefficients[selected_index])
    if selected_amplitude>1e-6:
        normalized=selected.real/selected_amplitude
    else:
        normalized=np.zeros_like(selected.real)
    image=axes[1].imshow(normalized.T,origin='lower',extent=extent,aspect='equal',
                         cmap='RdYlBu_r',vmin=-1,vmax=1,interpolation='nearest')
    colorbars.append(figure.colorbar(image,ax=axes[1],shrink=.8,label=r'$\operatorname{Re}U_m/|C_m|$'))
    suffix='' if selected_amplitude>1e-6 else ' (negligible amplitude; blank)'
    axes[1].set_title(rf'Order $m={selected_order:+d}$: '+r'$\operatorname{Re}U_m/|C_m|$'+
                      rf' ($|C_m|={selected_amplitude:.3g}$)'+suffix)
    k=2*np.pi/c.wavelength*c.n_output
    for ax,subset in ((axes[0],range(len(orders))),(axes[1],[selected_index])):
        subset=list(subset)
        if len(subset)>11:
            subset=sorted(subset,key=lambda j:abs(coefficients[j]),reverse=True)[:11]
        for j in subset:
            dx=solution.kx[mask][j]/k
            ds=solution.beta_output[mask][j].real/k
            length=.86*min(depth/ds,half_width/abs(dx) if abs(dx)>1e-14 else np.inf)
            tip=(length*ds,length*dx)
            arrow=ax.annotate('',xy=tip,xytext=(0,0),arrowprops=dict(
                arrowstyle='-|>',color='white',lw=1.5,mutation_scale=12,
                linestyle='--' if abs(coefficients[j])<1e-6 else '-'))
            arrow.arrow_patch.set_path_effects([pe.Stroke(linewidth=3,foreground='black'),pe.Normal()])
            ax.annotate(f'm={int(orders[j]):+d}',xy=tip,xytext=(4,4),textcoords='offset points',
                        color='white',fontsize=8,bbox=dict(facecolor='black',alpha=.65,edgecolor='none',pad=1))
        ax.axvline(0,color='black',lw=3)
        ax.set(xlabel=r'Output distance $s=(-z-h)/\lambda_0$',ylabel=r'Transverse $x/\lambda_0$',
               xlim=(0,depth),ylim=(-half_width,half_width))
    figure.suptitle(r'Propagating field $\operatorname{Re}[U(x,z)e^{-i\omega t}]$',fontsize=12)
    if not hasattr(figure,'_propagation_footer'):
        figure._propagation_footer=figure.text(.5,.015,
            'Propagating orders only; evanescent harmonics remain in the near-field plot.\n'
            'Arrows mark wavevectors, not beam boundaries. Infinite periodic plane waves overlap throughout this view.',
            ha='center',fontsize=9)
    figure.canvas.draw_idle()
    return dict(x=x,distance=distance,field=field,selected=selected,orders=orders,
                selected_order=selected_order)


class PropagationPlayer:
    """Animate cached complex fields; order/time controls never run the solver."""
    def __init__(self,solution,figure,axes,data):
        from matplotlib.widgets import Slider,Button
        self.figure,self.axes=figure,axes
        self.busy=False
        self.closed=False
        self.running=False
        maximum=max(1,int(np.max(abs(data['orders']))))
        self.order_slider=Slider(figure.add_axes([.15,.14,.30,.025]),'Order m',
                                 -maximum,maximum,valinit=data['selected_order'],valstep=1)
        self.time_slider=Slider(figure.add_axes([.65,.14,.25,.025]),'Time t/T',0,1,valinit=0)
        self.play=Button(figure.add_axes([.44,.075,.12,.04]),'Play')
        self.order_id=self.order_slider.on_changed(self.select_order)
        self.time_id=self.time_slider.on_changed(self.refresh)
        self.play_id=self.play.on_clicked(self.toggle)
        self.timer=figure.canvas.new_timer(interval=80)
        self.timer.add_callback(self.tick)
        self.close_id=figure.canvas.mpl_connect('close_event',self.close)
        self.set_solution(solution,data)

    def set_solution(self,solution,data):
        self.solution,self.data=solution,data
        self.busy=True
        maximum=max(1,int(np.max(abs(data['orders']))))
        self.order_slider.valmin,self.order_slider.valmax=-maximum,maximum
        self.order_slider.ax.set_xlim(-maximum,maximum)
        selected=int(self.order_slider.val)
        if selected not in data['orders']:
            selected=data['selected_order']
        self.order_slider.set_val(selected)
        self.busy=False
        self.select_order(selected)

    def select_order(self,value):
        if self.busy or self.closed:
            return
        orders=self.data['orders']
        order=int(orders[np.argmin(abs(orders-int(round(value))))])
        if order != self.order_slider.val:
            self.busy=True
            self.order_slider.set_val(order)
            self.busy=False
        c=self.solution.configuration
        j=int(np.flatnonzero(self.solution.orders==order)[0])
        coefficient=self.solution.transmission[j]
        self.data['selected_order']=order
        self.data['selected']=coefficient*np.exp(
            1j*(self.data['distance'][:,None]*self.solution.beta_output[j]
                +self.solution.kx[j]*self.data['x'][None,:]))
        self.selected_amplitude=abs(coefficient)
        # Replace only the selected-order direction arrow and its label.
        ax=self.axes[1]
        for artist in list(ax.texts):
            artist.remove()
        k=2*np.pi/c.wavelength*c.n_output
        ds=self.solution.beta_output[j].real/k
        dx=self.solution.kx[j]/k
        depth=propagation_depth_wavelengths
        width=propagation_half_width_wavelengths
        length=.86*min(depth/ds,width/abs(dx) if abs(dx)>1e-14 else np.inf)
        tip=(length*ds,length*dx)
        import matplotlib.patheffects as pe
        arrow=ax.annotate('',xy=tip,xytext=(0,0),arrowprops=dict(
                         arrowstyle='-|>',color='white',lw=2,mutation_scale=15))
        arrow.arrow_patch.set_path_effects([pe.Stroke(linewidth=4,foreground='black'),pe.Normal()])
        ax.annotate(f'm={order:+d}',xy=tip,xytext=(4,4),textcoords='offset points',
                    color='white',fontsize=9,bbox=dict(facecolor='black',alpha=.65,edgecolor='none',pad=1))
        ax.set_title(rf'Order $m={order:+d}$: '+r'$\operatorname{Re}(U_m e^{-i\omega t})/|C_m|$')
        self.refresh()

    def refresh(self,value=None):
        if self.busy or self.closed:
            return
        phase=np.exp(-2j*np.pi*self.time_slider.val)
        self.axes[0].images[0].set_data((self.data['field']*phase).real.T)
        if self.selected_amplitude>1e-6:
            selected=(self.data['selected']*phase).real/self.selected_amplitude
        else:
            selected=np.zeros_like(self.data['selected'].real)
            self.axes[1].set_title(rf"Order $m={self.data['selected_order']:+d}$: negligible $|C_m|$")
        self.axes[1].images[0].set_data(selected.T)
        self.figure.canvas.draw_idle()

    def tick(self):
        if not self.closed:
            self.time_slider.set_val((self.time_slider.val+.025)%1.)

    def toggle(self,event=None):
        if self.closed:
            return
        self.running=not self.running
        if self.running:
            self.timer.start()
        else:
            self.timer.stop()
        self.play.label.set_text('Pause' if self.running else 'Play')
        self.figure.canvas.draw_idle()

    def reset(self):
        if not self.closed:
            self.timer.stop()
            self.running=False
            self.play.label.set_text('Play')
            self.time_slider.set_val(0.)
            self.order_slider.set_val(propagation_order)

    def close(self,event=None):
        self.timer.stop()
        self.running=False
        self.closed=True
        self.order_slider.disconnect(self.order_id)
        self.time_slider.disconnect(self.time_id)
        self.play.disconnect(self.play_id)
        self.figure.canvas.mpl_disconnect(self.close_id)


class Explorer:
    """Matplotlib controls change a pending configuration; Apply performs one solve."""
    def __init__(self, solution, depth):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
        self.solution, self.initial, self.depth = solution, solution.configuration, depth
        self.initial_depth = depth
        self.figure, grid = plt.subplots(3, 3, figsize=(16, 11))
        self.figure.subplots_adjust(bottom=.27, hspace=.65, wspace=.5, top=.90)
        self.axes, self.colorbars = grid.ravel(), []
        c = self.initial
        definitions = [('lambda0', c.wavelength, .2, 1.5, None),
                       ('n_ridge', c.n_ridge, 1., max(3., c.n_ridge), None),
                       ('Ridge fraction', c.duty_cycle, 0., 1., None),
                       ('h (um)', c.thickness, .02, max(2., c.thickness), None),
                       ('M', c.M, 0, max(64, c.M), 1),
                       ('Output depth (um)', depth, 0., max(3., depth), None)]
        self.sliders = []
        for i, (label, value, low, high, step) in enumerate(definitions):
            low, high = min(low, value), max(high, value)
            axis = self.figure.add_axes([.13 + (i%2)*.47, .18-(i//2)*.055, .29, .022])
            self.sliders.append(Slider(axis, label, low, high, valinit=value, valstep=step))
        self.apply = Button(self.figure.add_axes([.36,.015,.12,.04]), 'Apply / solve')
        self.reset = Button(self.figure.add_axes([.51,.015,.12,.04]), 'Reset')
        self.status = self.figure.text(.5,.205,'Adjust parameters, then Apply. No solver runs while dragging.',ha='center',fontsize=9)
        self.apply_id = self.apply.on_clicked(self.update)
        self.reset_id = self.reset.on_clicked(self.restore)
        self.close_id = self.figure.canvas.mpl_connect('close_event', self.close)
        draw(solution, depth, self.figure, self.axes, self.colorbars)
        self.direction_figure, self.direction_axes = plt.subplots(1,2,figsize=(10,6))
        self.direction_figure.subplots_adjust(left=.08,right=.98,bottom=.19,top=.85,wspace=.25)
        draw_order_directions(solution,self.direction_figure,self.direction_axes)
        self.propagation_figure,self.propagation_axes=plt.subplots(1,2,figsize=(13,7))
        self.propagation_figure.canvas.manager.set_window_title('FMM large-range propagation (Figure 3)')
        self.propagation_figure.subplots_adjust(left=.07,right=.98,bottom=.26,top=.87,wspace=.30)
        self.propagation_colorbars=[]
        propagation_data=draw_propagation_view(solution,self.propagation_figure,self.propagation_axes,self.propagation_colorbars)
        self.propagation_player=PropagationPlayer(solution,self.propagation_figure,self.propagation_axes,propagation_data)

    def update(self, event=None):
        wavelength, ridge, fraction, thickness, cutoff, depth = [s.val for s in self.sliders]
        configuration = replace(self.initial, wavelength=wavelength, n_ridge=ridge,
                                duty_cycle=fraction, thickness=thickness, M=int(cutoff))
        try:
            candidate = self.solution if configuration == self.solution.configuration else solve(configuration)
            draw(candidate, depth, self.figure, self.axes, self.colorbars)
            if configuration != self.solution.configuration:
                import matplotlib.pyplot as plt
                if plt.fignum_exists(self.direction_figure.number):
                    draw_order_directions(candidate,self.direction_figure,self.direction_axes)
                if plt.fignum_exists(self.propagation_figure.number):
                    propagation_data=draw_propagation_view(candidate,self.propagation_figure,self.propagation_axes,self.propagation_colorbars)
                    self.propagation_player.set_solution(candidate,propagation_data)
            self.solution, self.depth = candidate, depth
            self.status.set_text(f'Applied M={configuration.M}: {2*configuration.M+1} harmonics. Check convergence for your observable.')
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            self.status.set_text(str(error))
        self.figure.canvas.draw_idle()

    def restore(self, event=None):
        for slider in self.sliders:
            slider.reset()
        self.update()
        self.propagation_player.reset()

    def close(self, event=None):
        self.apply.disconnect(self.apply_id)
        self.reset.disconnect(self.reset_id)
        self.figure.canvas.mpl_disconnect(self.close_id)
        import matplotlib.pyplot as plt
        plt.close(self.direction_figure)
        self.propagation_player.close()
        plt.close(self.propagation_figure)


if __name__ == '__main__':
    if not np.isfinite(output_depth) or output_depth < 0:
        raise ValueError('output_depth must be finite and nonnegative')
    import matplotlib
    if not interactive:
        matplotlib.use('Agg')
    elif plot_backend is not None:
        # Spyder's inline Plots pane does not support Matplotlib slider interaction.
        matplotlib.use(plot_backend)
    import matplotlib.pyplot as plt

    configuration = Configuration(
        wavelength=wavelength, period=period, thickness=thickness,
        duty_cycle=duty_cycle, n_incident=n_incident, n_output=n_output,
        n_ridge=n_ridge, n_groove=n_groove, M=M)
    solution = solve(configuration)
    if convergence_cutoffs:
        convergence(configuration, convergence_cutoffs, solution)
    describe(solution, output_depth)

    if interactive:
        # Spyder reruns share globals: close this script's previous owned windows.
        previous_explorer = globals().get('explorer')
        if hasattr(previous_explorer,'direction_figure') and hasattr(previous_explorer,'solution'):
            previous_explorer.close()
            plt.close(previous_explorer.figure)
        explorer = Explorer(solution, output_depth)
        # Keep callbacks and the figure alive if an IDE returns from show early.
        # Explorer.close is already registered with the window's close event.
        plt.show(block=True)
    else:
        figure, axes = plt.subplots(3, 3, figsize=(16, 11), layout='constrained')
        draw(solution, output_depth, figure, axes.ravel(), [])
        plt.close(figure)
        direction_figure, direction_axes = plt.subplots(1,2,figsize=(10,6),layout='constrained')
        draw_order_directions(solution,direction_figure,direction_axes)
        plt.close(direction_figure)
        propagation_figure,propagation_axes=plt.subplots(1,2,figsize=(13,7))
        propagation_figure.subplots_adjust(bottom=.16,top=.87)
        draw_propagation_view(solution,propagation_figure,propagation_axes,[])
        plt.close(propagation_figure)
