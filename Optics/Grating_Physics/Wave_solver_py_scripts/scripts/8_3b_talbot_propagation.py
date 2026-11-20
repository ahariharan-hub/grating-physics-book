"""8.3b: periodic ideal amplitude mask, exact ASM and Fresnel recurrence.
Talbot self-imaging is free propagation, not objective imaging. Lengths um.
"""
import numpy as np
from pathlib import Path
import sys
_script_directory = str(Path(__file__).resolve().parent)
if _script_directory not in sys.path:
    sys.path.insert(0, _script_directory)
from _common import pyplot,finish
from wave_optics.binary_transmission import binary_transmission_field
from wave_optics.scalar_propagation import transfer_function

# EDIT PARAMETERS
wavelength = .5  # Vacuum wavelength
medium_index = 1.
period = 4.
clear_fraction = .5
samples = 512
maximum_talbot_periods = 1.25
depth_samples = 121
show_plot = True


def compute():
    lam=wavelength/medium_index;dx=period/samples
    if lam<=0 or maximum_talbot_periods<=0 or samples<32 or depth_samples<2:
        raise ValueError('Check wavelength, Talbot range and sampling.')
    x=(np.arange(samples)-samples//2)*dx
    boundary=binary_transmission_field(x,period,clear_fraction,1.,0.)
    spectrum=np.fft.fft(boundary);fx=np.fft.fftfreq(samples,dx)
    talbot=2*period**2/lam;ratio=np.linspace(0,maximum_talbot_periods,depth_samples)
    fields={}
    for method in ('asm','fresnel'):
        fields[method]=np.array([np.fft.ifft(spectrum*transfer_function(fx,0.,lam,float(r*talbot),method)) for r in ratio])
    return dict(x=x,ratio=ratio,boundary=boundary,talbot=talbot,fields=fields)


def plot(data):
    plt=pyplot(show_plot);fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,method in zip(axes,['asm','fresnel']):
        im=ax.imshow(abs(data['fields'][method])**2,origin='lower',aspect='auto',extent=[-period/2,period/2,0,maximum_talbot_periods])
        ax.set(xlabel='x (um)',ylabel='s / z_T',title=method.upper());fig.colorbar(im,ax=ax)
    return fig


if __name__=='__main__':
    result=compute();print('Paraxial Talbot distance (um):',result['talbot'],'; ASM need not give exact revival. Check samples.')
    figure=plot(result);finish(figure,show_plot)
