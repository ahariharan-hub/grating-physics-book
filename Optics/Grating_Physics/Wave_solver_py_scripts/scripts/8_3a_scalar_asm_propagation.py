"""8.3a: prescribed 2D aperture, exact scalar ASM and paraxial Fresnel.
Vacuum wavelength um; homogeneous medium, downstream s=-z; no material or lens.
"""
import numpy as np
from pathlib import Path
import sys
_script_directory = str(Path(__file__).resolve().parent)
if _script_directory not in sys.path:
    sys.path.insert(0, _script_directory)
from _common import pyplot,finish
from wave_optics.scalar_propagation import propagate_field

# EDIT PARAMETERS
wavelength = .633
medium_index = 1.
aperture = 'circle'  # circle, square, gaussian
width = 4.
distance = 40.
window = 32.
samples = 256
show_plot = True


def compute():
    lam=wavelength/medium_index;dx=window/samples
    if lam<=0 or dx>lam/4 or distance<0 or not 0<width<window/2:
        raise ValueError('Check positive wavelength, dx <= lambda/4, distance >= 0, and aperture padding.')
    x=(np.arange(samples)-samples//2)*dx;xx,yy=np.meshgrid(x,x)
    if aperture=='circle':boundary=(xx**2+yy**2<=(width/2)**2).astype(complex)
    elif aperture=='square':boundary=((abs(xx)<=width/2)&(abs(yy)<=width/2)).astype(complex)
    elif aperture=='gaussian':boundary=np.exp(-(xx**2+yy**2)/width**2)
    else:raise ValueError('Choose circle, square or gaussian.')
    asm=propagate_field(boundary,dx,dx,lam,distance,'asm')['field']
    fresnel=propagate_field(boundary,dx,dx,lam,distance,'fresnel')['field']
    return dict(x=x,boundary=boundary,asm=asm,fresnel=fresnel,error=np.linalg.norm(asm-fresnel)/np.linalg.norm(asm))


def plot(data):
    plt=pyplot(show_plot);fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');x=data['x']
    for ax,name,title in zip(axes,['boundary','asm','fresnel'],['Input intensity','ASM output','Fresnel output']):
        im=ax.imshow(abs(data[name])**2,origin='lower',extent=[x[0],x[-1],x[0],x[-1]])
        ax.set(xlabel='x (um)',ylabel='y (um)',title=title);fig.colorbar(im,ax=ax)
    return fig


if __name__=='__main__':
    result=compute();print('Full-window relative complex-field difference:',result['error'],'; check window and samples separately.')
    figure=plot(result);finish(figure,show_plot)
