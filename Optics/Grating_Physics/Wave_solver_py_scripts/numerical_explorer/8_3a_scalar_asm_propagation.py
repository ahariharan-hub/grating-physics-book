"""2D prescribed apertures and exact scalar ASM versus paraxial Fresnel propagation.
No material scattering or imaging lens. FFT grid is periodic: test padding and dx.
All lengths um; wavelength is vacuum; exp(-i omega t), output distance s>=0.
Edit this block, run with F5. No main() and no automatic saving.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _playground import np,positive,integer,phase,image,run,BLUE,ORANGE
from wave_optics.scalar_propagation import propagate_field

# EDIT PARAMETERS
wavelength=.633
medium_index=1.
aperture='circle'          # 'slit', 'double_slit', 'circle', 'rectangle', 'gaussian'
width=4.                  # Diameter, slit width, or Gaussian amplitude radius
height=6.                 # Rectangle height (um)
separation=8.             # Double-slit center separation (um)
phase_shift=0.             # Uniform aperture phase, radians
distance=40.
window=64.                # Padded FFT domain width and height
samples=512               # Spatial sampling per transverse axis
display_half_width=14.
interactive=True
plot_backend='QtAgg'


def compute(p):
    lam=positive(p['wavelength'],'wavelength')/positive(p['medium_index'],'medium_index')
    width=positive(p['width'],'width');height=positive(p['height'],'height')
    distance=positive(p['distance'],'distance',True);window=positive(p['window'],'window')
    size=integer(p['samples'],'samples',64);dx=window/size
    view=positive(p['display_half_width'],'display_half_width')
    if view>=window/2:raise ValueError('display_half_width must be smaller than window/2')
    if max(width,height,p['separation'])>=window*.8:raise ValueError('Increase FFT window around the input aperture')
    if dx>lam/4:raise ValueError('Increase samples: need dx <= medium wavelength/4')
    x=(np.arange(size)-size//2)*dx;X,Y=np.meshgrid(x,x)
    kind=p['aperture']
    if kind=='slit':boundary=(abs(X)<=width/2).astype(complex)
    elif kind=='double_slit':boundary=((abs(X-p['separation']/2)<=width/2)|(abs(X+p['separation']/2)<=width/2)).astype(complex)
    elif kind=='circle':boundary=(X**2+Y**2<=(width/2)**2).astype(complex)
    elif kind=='rectangle':boundary=((abs(X)<=width/2)&(abs(Y)<=height/2)).astype(complex)
    elif kind=='gaussian':boundary=np.exp(-(X**2+Y**2)/width**2).astype(complex)
    else:raise ValueError('Unknown aperture; choose slit, double_slit, circle, rectangle or gaussian')
    if not np.isfinite(p['phase_shift']):raise ValueError('phase_shift must be finite')
    boundary*=np.exp(1j*p['phase_shift'])
    asm=propagate_field(boundary,dx,dx,lam,distance,'asm')['field']
    fresnel=propagate_field(boundary,dx,dx,lam,distance,'fresnel')['field']
    crop=np.flatnonzero(abs(x)<=view);select=np.ix_(crop,crop)
    intensity=abs(asm)**2;edge=max(1,size//50);border=np.ones((size,size),dtype=bool);border[edge:-edge,edge:-edge]=False
    edge_fraction=float(np.sum(intensity[border])/max(np.sum(intensity),np.finfo(float).eps))
    error=float(np.linalg.norm(asm-fresnel)/max(np.linalg.norm(asm),np.finfo(float).eps))
    return dict(x=x[crop],boundary=boundary[select],asm=asm[select],fresnel=fresnel[select],
                dx=dx,edge_fraction=edge_fraction,error=error,
                caption=f'{kind} prescribed boundary | lambda0={p["wavelength"]:g} um, n={p["medium_index"]:g}, s={distance:g} um',
                status=f'dx={dx:.4g} um; border intensity fraction={edge_fraction:.2g}; Fresnel/ASM field difference={error:.2g}. Padding and sampling need independent checks.')


def draw(data,figure,axes,colorbars):
    x=data['x'];extent=[x[0],x[-1],x[0],x[-1]]
    panels=[(abs(data['boundary']),r'Input $|U_0|$','Amplitude','viridis',{}),
            (phase(data['boundary']),r'Input $\arg U_0$','Phase (rad)','twilight',dict(vmin=-np.pi,vmax=np.pi)),
            (abs(data['asm'])**2,r'ASM output $|U|^2$','Relative intensity','inferno',{}),
            (abs(data['asm']),r'ASM output $|U|$','Amplitude','viridis',{}),
            (phase(data['asm']),r'ASM output $\arg U$','Phase (rad)','twilight',dict(vmin=-np.pi,vmax=np.pi))]
    for ax,(values,title,label,cmap,limits) in zip(axes[:5],panels):
        image(figure,ax,values,extent,colorbars,title=title,label=label,cmap=cmap,**limits)
        ax.set(xlabel='x (um)',ylabel='y (um)',aspect='equal')
    middle=int(np.argmin(abs(x)))
    axes[5].plot(x,abs(data['asm'][middle])**2,color=BLUE,label='ASM')
    axes[5].plot(x,abs(data['fresnel'][middle])**2,'--',color=ORANGE,label='Fresnel')
    axes[5].set(title='Same output plane, y=0',xlabel='x (um)',ylabel='Relative intensity')
    axes[5].legend(fontsize=8);axes[5].grid(True,alpha=.2)


parameters=dict(wavelength=wavelength,medium_index=medium_index,aperture=aperture,width=width,height=height,
                separation=separation,phase_shift=phase_shift,distance=distance,window=window,
                samples=samples,display_half_width=display_half_width)

if __name__=='__main__':
    controls=[('wavelength','lambda0 (um)',.5,1.,None),('width','Width (um)',1,12,None),
              ('distance','Distance (um)',0,100,None),('phase_shift','Input phase (rad)',-np.pi,np.pi,None)]
    experiment=run(parameters,controls,compute,draw,(2,3),'Scalar aperture propagation: ASM and Fresnel',interactive=interactive,plot_backend=plot_backend)
