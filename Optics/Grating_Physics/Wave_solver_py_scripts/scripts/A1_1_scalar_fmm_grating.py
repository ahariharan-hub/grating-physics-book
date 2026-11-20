"""Appendix 1.1: normal-incidence lossless scalar/TE FMM solve.
Vacuum wavelength and lengths um; layer -h<z<0; exp(-i omega t).
"""
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
n_groove = 1.
n_incident = 1.
n_output = 1.
M = 12  # Retain -M..M; increase to check convergence.
output_depth = .2
show_plot = True


def compute():
    if output_depth<0: raise ValueError('Output depth must be nonnegative.')
    config=ScalarGratingConfiguration(wavelength,period,thickness,ridge_fraction,n_incident,n_output,n_ridge,n_groove,M)
    solution=solve_scalar_grating(config)
    x=np.linspace(-period,period,801)
    zref=-thickness-output_depth
    return dict(solution=solution,x=x,field=solution.reconstruct(x,[zref])[0],coefficients=solution.at_reference_plane(zref))


def plot(data):
    plt=pyplot(show_plot);fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');s=data['solution']
    axes[0].plot(data['x'],abs(data['field'])**2);axes[0].set(xlabel='x (um)',ylabel='Relative intensity',title='Output plane')
    axes[1].stem(s.orders,abs(data['coefficients']));axes[1].set(xlabel='Order m',ylabel='|C_m|')
    axes[2].stem(s.orders,s.transmission_efficiency);axes[2].set(xlabel='Order m',ylabel='Transmitted power fraction')
    return fig


if __name__=='__main__':
    result=compute();print('R+T =',result['solution'].power_balance,'; check M convergence separately.')
    figure=plot(result);finish(figure,show_plot)
