"""Appendix 1.4: independent material solves across thickness; lengths um.
Normal incidence, scalar/TE, lossless air exteriors. Invalid points stay gaps.
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
ridge_fraction = .5
n_ridge = 1.5
M = 12
minimum_thickness = .05
maximum_thickness = .6
points = 21
selected_order = 1
show_plot = True


def compute():
    if points<3 or not 0<minimum_thickness<maximum_thickness or abs(selected_order)>M:
        raise ValueError('Use >=3 points, positive ordered thickness limits, and a retained order.')
    values=np.linspace(minimum_thickness,maximum_thickness,points)
    config=ScalarGratingConfiguration(wavelength=wavelength,period=period,duty_cycle=ridge_fraction,n_ridge=n_ridge,M=M)
    coefficients=np.full(points,np.nan+1j*np.nan);efficiency=np.full(points,np.nan);failures=[]
    for i,h in enumerate(values):
        try:
            solution=solve_scalar_grating(replace(config,thickness=float(h)))
            index=int(np.flatnonzero(solution.orders==selected_order)[0])
            coefficients[i]=solution.transmission[index]/np.exp(2j*np.pi*h/wavelength)
            efficiency[i]=solution.transmission_efficiency[index]
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            failures.append((float(h),str(error)))
    return dict(thickness=values,coefficients=coefficients,efficiency=efficiency,failures=failures)


def plot(data):
    plt=pyplot(show_plot);fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,values,title in zip(axes,[abs(data['coefficients']),np.where(abs(data['coefficients'])>1e-6,np.angle(data['coefficients']),np.nan),data['efficiency']],['Coefficient magnitude','Phase (rad), incident exit reference','Transmitted power fraction']):
        ax.plot(data['thickness'],values);ax.set(xlabel='Thickness (um)',title=title)
    return fig


if __name__=='__main__':
    result=compute();print('Failed points:',result['failures'],'; check M and sweep density independently.')
    figure=plot(result);finish(figure,show_plot)
