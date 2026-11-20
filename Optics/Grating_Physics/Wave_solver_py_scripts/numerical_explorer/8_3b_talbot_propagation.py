"""Periodic thin-mask near-field recurrence: ASM versus paraxial Fresnel.
A Talbot self-image is free propagation, not an aerial image formed by a lens.
The textbook z_T=2*d^2/lambda is paraxial; exact ASM need not revive a sharp mask.
Lengths um, lambda0 vacuum, lambda=lambda0/n; exp(-i omega t). No saving.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _playground import np,positive,integer,propagate_1d,image,run,BLUE,ORANGE,GREEN
from wave_optics.binary_transmission import binary_transmission_field

# EDIT PARAMETERS
wavelength=.5
medium_index=1.
period=4.
open_fraction=.5
samples=512                 # One period, excluding its repeated endpoint
depth_samples=161
maximum_talbot_periods=1.25
observation_fraction=1.      # Observation distance / paraxial Talbot distance
interactive=True
plot_backend='QtAgg'


def compute(p):
    lam=positive(p['wavelength'],'wavelength')/positive(p['medium_index'],'medium_index')
    d=positive(p['period'],'period');fraction=float(p['open_fraction'])
    if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('open_fraction must lie in [0,1]')
    size=integer(p['samples'],'samples',32);nz=integer(p['depth_samples'],'depth_samples',2)
    maximum=positive(p['maximum_talbot_periods'],'maximum_talbot_periods')
    observation=positive(p['observation_fraction'],'observation_fraction',True)
    if observation>maximum:raise ValueError('observation_fraction must not exceed maximum_talbot_periods')
    dx=d/size;x=(np.arange(size)-size//2)*dx
    mask=binary_transmission_field(x,d,fraction,1.,0.)
    talbot=2*d*d/lam;ratios=np.linspace(0,maximum,nz);depths=ratios*talbot
    asm,_=propagate_1d(mask,dx,lam,depths,'asm')
    fresnel,_=propagate_1d(mask,dx,lam,depths,'fresnel')
    selected_asm,_=propagate_1d(mask,dx,lam,[observation*talbot],'asm')
    selected_fresnel,_=propagate_1d(mask,dx,lam,[observation*talbot],'fresnel')
    carrier=np.exp(2j*np.pi/lam*depths)
    norm=max(float(np.linalg.norm(mask)),np.finfo(float).eps)
    error_asm=np.linalg.norm(asm/carrier[:,None]-mask,axis=1)/norm
    error_fresnel=np.linalg.norm(fresnel/carrier[:,None]-mask,axis=1)/norm
    return dict(x=x,mask=mask,asm=asm,fresnel=fresnel,ratios=ratios,talbot=talbot,period=d,
        selected_asm=selected_asm[0],selected_fresnel=selected_fresnel[0],observation=observation,
        error_asm=error_asm,error_fresnel=error_fresnel,
        caption=f'Periodic boundary | lambda0={p["wavelength"]:g} um, n={p["medium_index"]:g} | paraxial z_T={talbot:g} um',
        status='z_T is a paraxial prediction. ASM retains nonparaxial phase and evanescent decay; sharp-mask recurrence need not be exact. Check samples.')


def draw(data,figure,axes,colorbars):
    d=data['period'];ratios=data['ratios'];extent=[-1.5*d,1.5*d,0,ratios[-1]]
    peak=max(1.,float(np.max(abs(data['asm'])**2)),float(np.max(abs(data['fresnel'])**2)))
    for ax,key,title in zip(axes[:2],['asm','fresnel'],['Exact scalar ASM','Paraxial Fresnel']):
        values=np.tile(abs(data[key])**2,(1,3))
        image(figure,ax,values,extent,colorbars,title=title+r': $|U|^2$',label='Relative intensity',vmin=0,vmax=peak)
        ax.axhline(data['observation'],color='cyan',lw=1)
        ax.set(xlabel='x (um)',ylabel=r'$s/z_T$',ylim=(ratios[-1],0))
    axes[2].plot(data['x'],abs(data['selected_asm'])**2,color=BLUE,label='ASM')
    axes[2].plot(data['x'],abs(data['selected_fresnel'])**2,'--',color=ORANGE,label='Fresnel')
    axes[2].plot(data['x'],abs(data['mask'])**2,':',color='0.4',label='Mask-plane reference')
    axes[2].set(title=f'Intensity at s/z_T={data["observation"]:g}',xlabel='x (um)',ylabel='Relative intensity')
    axes[3].plot(ratios,data['error_asm'],color=BLUE,label='ASM')
    axes[3].plot(ratios,data['error_fresnel'],'--',color=ORANGE,label='Fresnel')
    axes[3].axvline(1,color='0.5',lw=1)
    axes[3].set(title='Field recurrence error: common carrier removed',xlabel=r'$s/z_T$',ylabel='Relative L2 difference from input')
    for ax in axes[2:]:ax.legend(fontsize=8);ax.grid(True,alpha=.2)


parameters=dict(wavelength=wavelength,medium_index=medium_index,period=period,open_fraction=open_fraction,
    samples=samples,depth_samples=depth_samples,maximum_talbot_periods=maximum_talbot_periods,
    observation_fraction=observation_fraction)

if __name__=='__main__':
    controls=[('wavelength','lambda0 (um)',.25,1.,None),('period','Pitch (um)',2,8,None),
              ('open_fraction','Open fraction',.1,.9,None),('observation_fraction','s / z_T',0,maximum_talbot_periods,None)]
    experiment=run(parameters,controls,compute,draw,(2,2),'Talbot recurrence: exact ASM versus Fresnel',interactive=interactive,plot_backend=plot_backend)
