"""Basis convergence for complex scattering amplitudes and a near-field ROI.
The highest M is a numerical reference, NOT an exact solution. Flux balance and
boundary residual do not establish basis convergence. Existing normal-incidence
lossless scalar/TE backend. Lengths um; wavelength is vacuum. No file exports.
"""
from pathlib import Path
import sys
from functools import lru_cache
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _playground import np,positive,integer,run,BLUE,ORANGE,GREEN
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
cutoffs=(3,5,8,12,20,32)
reference_M=48
target_order=1
output_depth=.03
interactive=True
plot_backend='QtAgg'


@lru_cache(maxsize=96)
def solve_cached(configuration):
    return solve_scalar_grating(configuration)


def compute(p):
    maximum=integer(p['reference_M'],'reference_M',1)
    raw=list(p['cutoffs'])
    if not raw:raise ValueError('Provide at least one lower cutoff')
    validated=[integer(v,'cutoff',0) for v in raw]
    limits=sorted({v for v in validated if v<maximum}|{maximum})
    if len(limits)<2:raise ValueError('reference_M must exceed at least one lower cutoff')
    target=int(p['target_order'])
    if target!=p['target_order'] or abs(target)>min(limits):raise ValueError('Every cutoff must include target_order')
    depth=positive(p['output_depth'],'output_depth',True)
    c=ScalarGratingConfiguration(wavelength=p['wavelength'],period=p['period'],thickness=p['thickness'],
        duty_cycle=p['ridge_fraction'],n_incident=p['n_incident'],n_output=p['n_output'],
        n_ridge=p['n_ridge'],n_groove=p['n_groove'],M=maximum)
    solutions=[solve_cached(replace(c,M=m)) for m in limits]
    ref=solutions[-1];labels=ref.orders[ref.output_kinds=='propagating']
    reference=ref.transmission[labels+maximum]
    scale=max(float(np.linalg.norm(reference)),np.finfo(float).eps)
    x=np.linspace(-c.period/2,c.period/2,401);z=-c.thickness-depth
    profiles=[s.reconstruct(x,[z])[0] for s in solutions]
    field_scale=max(float(np.linalg.norm(profiles[-1])),np.finfo(float).eps)
    errors=[];field_errors=[];coefficients=[];balances=[];residuals=[]
    carrier=np.exp(2j*np.pi/c.wavelength*c.n_incident*c.thickness)
    for trial,profile in zip(solutions,profiles):
        present=abs(labels)<=trial.configuration.M
        comparison=np.zeros(labels.size,dtype=complex)
        comparison[present]=trial.transmission[labels[present]+trial.configuration.M]
        errors.append(np.linalg.norm(comparison-reference)/scale)
        field_errors.append(np.linalg.norm(profile-profiles[-1])/field_scale)
        coefficients.append(trial.transmission[target+trial.configuration.M]/carrier)
        balances.append(abs(trial.power_balance-1));residuals.append(trial.boundary_residual)
    table=' M   relative complex-order error   relative field error   |R+T-1|\n'
    table+='\n'.join(f'{m:3d}       {e:.4e}                  {f:.4e}           {b:.3e}' for m,e,f,b in zip(limits,errors,field_errors,balances))
    return dict(configuration=c,limits=np.array(limits),errors=np.array(errors),field_errors=np.array(field_errors),
                coefficients=np.array(coefficients),balances=np.array(balances),residuals=np.array(residuals),
                x=x,profiles=np.array(profiles),target=target,labels=labels,table=table,
                caption=f'Compare M against numerical reference M={maximum} | same exit coefficients and ROI z={z:.4g} um',
                status='Missing propagating labels at low M are treated as zero. Highest-M errors are zero by definition; verify further increases.')


def draw(data,figure,axes,colorbars):
    m=data['limits'];floor=1e-16
    axes[0].semilogy(m[:-1],np.maximum(data['errors'][:-1],floor),'o-',color=BLUE,label='Complex propagating coefficients')
    axes[0].semilogy(m[:-1],np.maximum(data['field_errors'][:-1],floor),'s--',color=ORANGE,label='Complex field ROI')
    axes[0].axvline(m[-1],color='0.5',ls=':',lw=1,label=f'Reference M={m[-1]} (not an error point)')
    axes[0].set(title='Relative L2 difference from highest M',xlabel='Fourier cutoff M',ylabel='Relative difference')
    axes[1].semilogy(m,np.maximum(data['balances'],floor),'o-',color=BLUE,label='|R+T-1|')
    axes[1].semilogy(m,np.maximum(data['residuals'],floor),'s--',color=ORANGE,label='Matching residual')
    axes[1].set(title='Checks that do NOT prove basis convergence',xlabel='Fourier cutoff M',ylabel='Residual / balance error')
    axes[2].plot(m,data['coefficients'].real,'o-',color=BLUE,label='Re a_m')
    axes[2].plot(m,data['coefficients'].imag,'s--',color=ORANGE,label='Im a_m')
    axes[2].set(title=f'Complex transmitted coefficient: m={data["target"]:+d}',xlabel='Fourier cutoff M',ylabel='Exit coefficient / incident reference')
    for j,color in zip(sorted(set([0,len(m)//2,len(m)-1])),[ORANGE,GREEN,BLUE]):
        axes[3].plot(data['x'],abs(data['profiles'][j])**2,color=color,label=f'M={m[j]}')
    axes[3].set(title='Same output-plane intensity, all retained harmonics',xlabel='x (um)',ylabel='Relative intensity')
    for ax in axes:ax.legend(fontsize=8);ax.grid(True,alpha=.2)


parameters=dict(wavelength=wavelength,period=period,thickness=thickness,ridge_fraction=ridge_fraction,
    n_incident=n_incident,n_output=n_output,n_ridge=n_ridge,n_groove=n_groove,cutoffs=cutoffs,
    reference_M=reference_M,target_order=target_order,output_depth=output_depth)

if __name__=='__main__':
    controls=[('wavelength','lambda0 (um)',.1,.6,None),('thickness','h (um)',.02,.5,None),
              ('n_ridge','Ridge n',1,3,None),('target_order','Target m',-3,3,1),
              ('reference_M','Reference M',16,64,1),('output_depth','ROI depth (um)',0,.5,None)]
    experiment=run(parameters,controls,compute,draw,(2,2),'FMM basis convergence: coefficients and near field',interactive=interactive,plot_backend=plot_backend)
