"""Appendix 2.1: finite prescribed openings propagated with scalar ASM.
No finite-material scattering. Vacuum wavelength um; downstream s=-z.
"""
import numpy as np
from pathlib import Path
import sys
_script_directory = str(Path(__file__).resolve().parent)
if _script_directory not in sys.path:
    sys.path.insert(0, _script_directory)
from _common import pyplot,finish
from wave_optics.order_array import opening_positions
from wave_optics.scalar_propagation import transfer_function

# EDIT PARAMETERS
wavelength = .633
medium_index = 1.
period = 3.
opening_width = 1.
openings = 5
beam_waist = 8.
window = 64.
samples = 2048
maximum_depth = 30.
depth_samples = 121
show_plot = True


def compute():
    lam=wavelength/medium_index;dx=window/samples
    positions=opening_positions(period,openings)
    if lam<=0 or dx>lam/4 or maximum_depth<=0 or not 0<opening_width<=period or beam_waist<=0:
        raise ValueError('Check wavelength, sampling, depth, opening width and beam waist.')
    if max(abs(positions))+opening_width/2>=window*.4:raise ValueError('Increase FFT padding around the finite grating.')
    x=(np.arange(samples)-samples//2)*dx
    mask=np.any(abs(x[:,None]-positions)<=opening_width/2,axis=1)
    boundary=mask*np.exp(-(x/beam_waist)**2)
    spectrum=np.fft.fft(boundary);fx=np.fft.fftfreq(samples,dx)
    s=np.linspace(0,maximum_depth,depth_samples)
    field=np.array([np.fft.ifft(spectrum*transfer_function(fx,0.,lam,float(depth),'asm')) for depth in s])
    return dict(x=x,depth=s,boundary=boundary,field=field)


def plot(data):
    plt=pyplot(show_plot);fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    x=data['x'];im=axes[0].imshow(abs(data['field'])**2,origin='lower',extent=[x[0],x[-1],0,maximum_depth],aspect='equal')
    axes[0].set(xlabel='x (um)',ylabel='s=-z (um)',title='Finite ideal openings under Gaussian illumination');fig.colorbar(im,ax=axes[0])
    axes[1].plot(x,abs(data['field'][-1])**2);axes[1].set(xlabel='x (um)',ylabel='Output relative intensity')
    return fig


if __name__=='__main__':
    result=compute();print('Prescribed boundary propagated by ASM. Vary FFT window and samples independently.')
    figure=plot(result);finish(figure,show_plot)
