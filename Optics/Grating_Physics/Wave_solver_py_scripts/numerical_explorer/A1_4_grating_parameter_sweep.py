"""Material-grating parameter sweep using the existing scalar/TE FMM backend.
Normal incidence, real indices, one rectangular layer. Lengths um, lambda0 vacuum.
Each point is a scattering solve, not a thin-mask rescaling. Invalid/singular
points are left as gaps. Fixed M must be checked separately for convergence.
No main() function, imaging calculation, or automatic file saving.
"""
from pathlib import Path
import sys
from dataclasses import replace
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _playground import np,positive,integer,phase,run,BLUE,ORANGE,GREEN
from wave_optics.scalar_grating import ScalarGratingConfiguration,solve_scalar_grating

# EDIT PARAMETERS
wavelength=.193
period=.6
thickness=.1
ridge_fraction=2/3
n_incident=1.
n_output=1.
n_ridge=1.5
n_groove=1.
M=24
sweep_parameter='thickness'  # thickness, wavelength, period, duty_cycle, n_ridge, n_output
sweep_min=.02
sweep_max=.5
points=31
orders=(0,1,2,3)
interactive=True
plot_backend='QtAgg'


@lru_cache(maxsize=128)
def solve_cached(configuration):
    return solve_scalar_grating(configuration)


def compute(p):
    name=p['sweep_parameter']
    if name not in ('thickness','wavelength','period','duty_cycle','n_ridge','n_output'):
        raise ValueError('Unknown sweep_parameter')
    low=float(p['sweep_min']);high=float(p['sweep_max'])
    if not np.isfinite(low+high) or high<=low:raise ValueError('Use finite sweep_min < sweep_max')
    count=integer(p['points'],'points',3);cutoff=integer(p['M'],'M',0)
    if count>201:raise ValueError('Use at most 201 sweep points')
    labels=np.asarray(p['orders'])
    if labels.ndim!=1 or not labels.size or not np.issubdtype(labels.dtype,np.integer) or np.max(abs(labels))>cutoff:
        raise ValueError('orders must be integer labels retained by M')
    c=ScalarGratingConfiguration(wavelength=p['wavelength'],period=p['period'],thickness=p['thickness'],
        duty_cycle=p['ridge_fraction'],n_incident=p['n_incident'],n_output=p['n_output'],
        n_ridge=p['n_ridge'],n_groove=p['n_groove'],M=cutoff)
    values=np.linspace(low,high,count)
    coefficients=np.full((count,len(labels)),np.nan+1j*np.nan,dtype=complex)
    efficiency=np.full((count,len(labels)),np.nan);totals=np.full((count,3),np.nan)
    failures=[]
    for j,value in enumerate(values):
        try:
            trial=replace(c,**{name:float(value)})
            solution=solve_cached(trial)
            carrier=np.exp(2j*np.pi/trial.wavelength*trial.n_incident*trial.thickness)
            coefficients[j]=solution.transmission[labels+cutoff]/carrier
            efficiency[j]=solution.transmission_efficiency[labels+cutoff]
            R=float(np.sum(solution.reflection_efficiency));T=float(np.sum(solution.transmission_efficiency))
            totals[j]=R,T,R+T
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            failures.append((j,str(error)))
    if len(failures)==count:raise ValueError('All sweep points failed; check physical parameters and grazing thresholds')
    phases=np.stack([phase(coefficients[:,j],break_wraps=True) for j in range(len(labels))],axis=1)
    units=' (um)' if name in ('thickness','wavelength','period') else ''
    return dict(values=values,coefficients=coefficients,efficiency=efficiency,phases=phases,
        totals=totals,orders=labels,failures=failures,xlabel=name+units,
        caption=f'Sweep {name}; {count} independent configurations | fixed M={cutoff}, normal incidence',
        status=f'{len(failures)} invalid/singular points shown as gaps. Order phases use the same continued-incident exit reference; check M convergence and sweep-point density separately.')


def draw(data,figure,axes,colorbars):
    from matplotlib import colormaps
    colors=colormaps['viridis'](np.linspace(.1,.85,len(data['orders'])))
    for j,(order,color) in enumerate(zip(data['orders'],colors)):
        label=f'm={order:+d}'
        axes[0].plot(data['values'],abs(data['coefficients'][:,j]),color=color,label=label)
        axes[1].plot(data['values'],data['phases'][:,j],'o-',ms=3,color=color,label=label)
        axes[2].plot(data['values'],data['efficiency'][:,j],color=color,label=label)
    axes[0].set(title=r'Exit magnitudes $|a_m|$',ylabel='Magnitude')
    axes[1].set(title=r'Exit phases $\arg a_m$',ylabel='Phase (rad)',ylim=(-3.3,3.3))
    axes[2].set(title=r'Transmitted diffraction efficiencies $T_m$',ylabel='Normal power fraction')
    for j,(label,color) in enumerate(zip(['R','T','R+T'],['0.5',BLUE,ORANGE])):
        axes[3].plot(data['values'],data['totals'][:,j],color=color,label=label)
    axes[3].set(title='Lossless flux balance is not a convergence test',ylabel='Incident power fraction')
    for ax in axes:
        ax.set_xlabel(data['xlabel']);ax.legend(fontsize=8);ax.grid(True,alpha=.2)


parameters=dict(wavelength=wavelength,period=period,thickness=thickness,ridge_fraction=ridge_fraction,
    n_incident=n_incident,n_output=n_output,n_ridge=n_ridge,n_groove=n_groove,M=M,
    sweep_parameter=sweep_parameter,sweep_min=sweep_min,sweep_max=sweep_max,points=points,orders=orders)

if __name__=='__main__':
    ranges={'thickness':(.005,1.),'wavelength':(.1,.8),'period':(.2,3.),'duty_cycle':(0.,1.),'n_ridge':(1.,3.),'n_output':(1.,2.)}
    low,high=ranges[sweep_parameter]
    controls=[('sweep_min','Sweep start',low,high,None),('sweep_max','Sweep end',low,high,None),
              ('points','Points',5,61,1),('M','Fourier M',4,48,1)]
    experiment=run(parameters,controls,compute,draw,(2,2),'Physical grating parameter sweep',interactive=interactive,plot_backend=plot_backend)
