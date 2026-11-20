"""Finite illuminated ideal openings: prescribed mask + FFT/ASM, not FMM scattering.
The slits are invariant along y; the finite array and envelope vary along x.
Lengths are um. Vacuum wavelength is converted to lambda0/n. Output distance
s>=0; time factor exp(-i omega t). Edit parameters, then run the whole file.
No main() function and no automatic file saving. Requires NumPy/Matplotlib.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _playground import np,positive,integer,propagate_1d,image,run,BLUE,ORANGE
from wave_optics.binary_transmission import binary_transmission_field,binary_transmission_coefficients

# EDIT PARAMETERS
wavelength=.193            # Vacuum wavelength (um)
medium_index=1.
period=.6
opening_width=.2
openings=7                # Finite number of openings, not a periodic scatterer
waist=2.5                 # Incident Gaussian amplitude exp(-x^2/waist^2)
illumination='gaussian'   # 'gaussian' or 'uniform'; array remains finite
maximum_depth=12.
display_half_width=10.
fft_window=128.           # FFT periodic domain; increase to check wraparound
samples=16384            # Increase independently to check spatial sampling
depth_samples=161
interactive=True
plot_backend='QtAgg'      # None respects IDE backend


def compute(p):
    lam=positive(p['wavelength'],'wavelength')/positive(p['medium_index'],'medium_index')
    d=positive(p['period'],'period');a=positive(p['opening_width'],'opening_width')
    count=integer(p['openings'],'openings');size=integer(p['samples'],'samples',512)
    nz=integer(p['depth_samples'],'depth_samples',2)
    maximum=positive(p['maximum_depth'],'maximum_depth')
    window=positive(p['fft_window'],'fft_window');view=positive(p['display_half_width'],'display_half_width')
    if a>d:raise ValueError('opening_width must not exceed period')
    if window<=max(count*d,2*view)*1.2:raise ValueError('Increase fft_window beyond the array and displayed crop')
    dx=window/size
    if dx>min(lam/8,a/8):raise ValueError('Increase samples: need dx <= min(medium wavelength, opening width)/8')
    x=(np.arange(size)-size//2)*dx
    shift=0. if count%2 else d/2
    mask=binary_transmission_field(x-shift,d,a/d,1.,0.).real*(abs(x)<count*d/2)
    if p['illumination']=='gaussian':incident=np.exp(-(x/positive(p['waist'],'waist'))**2)
    elif p['illumination']=='uniform':incident=np.ones(size)
    else:raise ValueError("illumination must be 'gaussian' or 'uniform'")
    boundary=mask*incident
    depths=np.linspace(0,maximum,nz);crop=abs(x)<=view
    fields,diagnostics=propagate_1d(boundary,dx,lam,depths,selection=crop)
    fx=diagnostics['fx'];propagating=abs(lam*fx)<1
    theta=np.degrees(np.arcsin(lam*fx[propagating]));sort=np.argsort(theta)
    spectral=abs(dx*diagnostics['spectrum'][propagating])**2
    spectral/=max(float(np.max(spectral)),np.finfo(float).eps)
    orders=np.arange(-int(np.floor(d/lam)),int(np.floor(d/lam))+1)
    orders=orders[(abs(orders*lam/d)<1-1e-12)&(abs(binary_transmission_coefficients(orders,a/d,1.,0.))>1e-10)]
    angles=np.arcsin(orders*lam/d)
    return dict(x=x[crop],depths=depths,fields=fields,mask=mask[crop],incident=incident[crop],
                boundary=boundary[crop],theta=theta[sort],spectral=spectral[sort],orders=orders,angles=angles,
                edge_fraction=diagnostics['edge_fraction'],dx=dx,
                caption=f'{count} prescribed openings | lambda0={p["wavelength"]:g} um, n={p["medium_index"]:g} | finite illumination; no material scattering',
                status=f'dx={dx:.4g} um; max edge intensity fraction={diagnostics["edge_fraction"]:.2g}. Check larger padding AND finer sampling.')


def draw(data,figure,axes,colorbars):
    from matplotlib.colors import LogNorm
    x,z=data['x'],data['depths'];intensity=abs(data['fields'])**2
    axes[0].plot(x,data['mask'],'--',color=ORANGE,label='Ideal amplitude transmission')
    axes[0].plot(x,data['incident'],color=BLUE,label='Incident amplitude')
    axes[0].set(title='Finite object and illumination',xlabel='x (um)',ylabel='Amplitude');axes[0].legend(fontsize=8)
    maximum=max(1.,float(np.max(intensity)))
    image(figure,axes[1],np.maximum(intensity,1e-5),[x[0],x[-1],z[0],z[-1]],colorbars,
          title=r'Propagated intensity $|U(x,s)|^2$',label='Relative intensity (log)',norm=LogNorm(1e-5,maximum))
    axes[1].set(xlabel='x (um)',ylabel='Output distance s (um)',ylim=(z[-1],0),aspect='equal')
    for m,angle in zip(data['orders'],data['angles']):
        slope=np.tan(angle);distance=.9*min(z[-1],max(abs(x))/abs(slope) if abs(slope)>1e-14 else np.inf)
        axes[1].plot([0,distance*slope],[0,distance],'w:',lw=1)
        axes[1].text(distance*slope,distance,f'{m:+d}',color='white',fontsize=7)
    axes[2].plot(data['theta'],data['spectral'],color=BLUE)
    axes[2].set(title='Angular spectrum: shape, not diffraction efficiency',xlabel='Angle from normal (deg)',ylabel='Peak-normalized spectral intensity',xlim=(-85,85))
    axes[3].plot(x,intensity[-1],color=BLUE)
    axes[3].set(title=f'Output profile at s={z[-1]:g} um',xlabel='x (um)',ylabel='Relative intensity')
    for ax in (axes[0],axes[2],axes[3]):ax.grid(True,alpha=.2)


parameters=dict(wavelength=wavelength,medium_index=medium_index,period=period,opening_width=opening_width,
    openings=openings,waist=waist,illumination=illumination,maximum_depth=maximum_depth,
    display_half_width=display_half_width,fft_window=fft_window,samples=samples,depth_samples=depth_samples)

if __name__=='__main__':
    controls=[('wavelength','lambda0 (um)',.1,.5,None),('period','Pitch (um)',.4,1.5,None),
              ('opening_width','Opening (um)',.1,.5,None),('openings','Openings',1,15,1),
              ('waist','Waist (um)',1,8,None),('maximum_depth','Depth (um)',2,24,None)]
    experiment=run(parameters,controls,compute,draw,(2,2),'Finite grating: prescribed openings + ASM',interactive=interactive,plot_backend=plot_backend)
