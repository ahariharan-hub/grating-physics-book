"""Appendix 1.2: physical absorber versus ideal amplitude mask.
Normal incidence, air exteriors, illustrative n+i*kappa; vacuum wavelength um.
"""
import numpy as np
from pathlib import Path
import sys
_script_directory = str(Path(__file__).resolve().parent)
if _script_directory not in sys.path:
    sys.path.insert(0, _script_directory)
from _common import pyplot,finish
from wave_optics.absorbing_grating import AbsorbingGratingConfiguration,solve_absorbing_grating
from wave_optics.amplitude_grating import IdealAmplitudeGrating

# EDIT PARAMETERS
wavelength = .633
period = 1.5
opening_width = .75
thickness = .3
absorber_index = 1.5
kappa = .8
M = 12
output_depth = .2
show_plot = True


def compute():
    if output_depth<0: raise ValueError('Output depth must be nonnegative.')
    config=AbsorbingGratingConfiguration(wavelength,period,opening_width,thickness,absorber_index,kappa,M=M)
    solution=solve_absorbing_grating(config);ideal=IdealAmplitudeGrating(config)
    x=np.linspace(-period,period,801);zref=-thickness-output_depth
    return dict(solution=solution,ideal=ideal,x=x,physical=solution.reconstruct(x,[zref])[0],mask=ideal.reconstruct(x,[zref])[0])


def plot(data):
    plt=pyplot(show_plot);fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');s=data['solution']
    axes[0].plot(data['x'],abs(data['physical'])**2,label='Physical absorber');axes[0].plot(data['x'],abs(data['mask'])**2,'--',label='Ideal mask')
    axes[0].set(xlabel='x (um)',ylabel='Relative intensity',title='Same output plane');axes[0].legend()
    axes[1].bar(['R','T','A'],[s.reflectance,s.transmittance,s.absorption]);axes[1].set_ylabel('Incident power fraction')
    return fig


if __name__=='__main__':
    result=compute();print('R+T+A =',result['solution'].energy_balance,'; M =',M)
    figure=plot(result);finish(figure,show_plot)
