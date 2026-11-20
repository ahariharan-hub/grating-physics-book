"""Appendix 1.3: complex coefficient and near-field convergence.
Highest M is a numerical reference, not an exact solution. Lengths um.
"""
from dataclasses import replace
import numpy as np
from pathlib import Path
import sys
_script_directory = str(Path(__file__).resolve().parent)
if _script_directory not in sys.path:
    sys.path.insert(0, _script_directory)
from _common import pyplot,finish
from wave_optics.scalar_grating import ScalarGratingConfiguration,solve_scalar_grating

# EDIT PARAMETERS
wavelength = .633
period = 1.5
thickness = .3
ridge_fraction = .5
n_ridge = 1.5
cutoffs = (2,4,8,12)
reference_M = 20
output_depth = .03
show_plot = True


def compute():
    if not cutoffs or any(m<0 or m>=reference_M or int(m)!=m for m in cutoffs):
        raise ValueError('Use integer cutoffs 0 <= M < reference_M.')
    if output_depth<0: raise ValueError('Output depth must be nonnegative.')
    config=ScalarGratingConfiguration(wavelength=wavelength,period=period,thickness=thickness,duty_cycle=ridge_fraction,n_ridge=n_ridge,M=reference_M)
    reference=solve_scalar_grating(config)
    labels=reference.orders[reference.output_kinds=='propagating']
    ref=reference.transmission[reference.output_kinds=='propagating']
    x=np.linspace(-period/2,period/2,401);zref=-thickness-output_depth
    ref_field=reference.reconstruct(x,[zref])[0]
    rows=[]
    for m in cutoffs:
        solution=solve_scalar_grating(replace(config,M=m))
        lookup=dict(zip(solution.orders,solution.transmission))
        co=np.array([lookup.get(int(label),0j) for label in labels])
        err=np.linalg.norm(co-ref)/max(np.linalg.norm(ref),1e-15)
        field=solution.reconstruct(x,[zref])[0]
        near=np.linalg.norm(field-ref_field)/max(np.linalg.norm(ref_field),1e-15)
        rows.append((m,err,near,solution.power_balance))
    return dict(table=np.array(rows),reference=reference)


def plot(data):
    plt=pyplot(show_plot);fig,ax=plt.subplots(figsize=(8,5),layout='constrained');t=data['table']
    ax.semilogy(t[:,0],np.maximum(t[:,1],1e-16),'o-',label='Complex propagating coefficients')
    ax.semilogy(t[:,0],np.maximum(t[:,2],1e-16),'s-',label='Near-field ROI')
    ax.set(xlabel='Fourier cutoff M',ylabel='Relative L2 difference',title=f'Reference M={reference_M}, not exact truth');ax.legend();return fig


if __name__=='__main__':
    result=compute();print('M, coefficient difference, near-field difference, R+T');print(result['table'])
    figure=plot(result);finish(figure,show_plot)
