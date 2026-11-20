"""Physical absorbing scalar grating versus an ideal binary amplitude mask.
Existing passive scalar/TE backend, normal incidence, air exterior on both sides.
Complex absorber index n+i*kappa with exp(-i omega t). Lengths um, lambda0 vacuum.
Material indices are illustrative, not calibrated 193 nm material data. No saving.
"""
from pathlib import Path
import sys
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _playground import np,positive,integer,phase,image,run,BLUE,ORANGE,GREEN
from wave_optics.absorbing_grating import AbsorbingGratingConfiguration,solve_absorbing_grating
from wave_optics.amplitude_grating import IdealAmplitudeGrating

# EDIT PARAMETERS
wavelength=.193
period=.6
opening_width=.2
thickness=.1
absorber_index=1.5
kappa=.8
M=24
output_depth=.2
interactive=True
plot_backend='QtAgg'


@lru_cache(maxsize=8)
def solve_cached(configuration):
    return solve_absorbing_grating(configuration)


def compute(p):
    c=AbsorbingGratingConfiguration(wavelength=p['wavelength'],period=p['period'],
        clear_width=p['opening_width'],thickness=p['thickness'],n_absorber=p['absorber_index'],
        kappa=p['kappa'],M=integer(p['M'],'M',0))
    depth=positive(p['output_depth'],'output_depth',True)
    solution=solve_cached(c);ideal=IdealAmplitudeGrating(c)
    x=np.linspace(-c.period/2,c.period/2,401)
    z=np.linspace(-c.thickness-max(depth,c.wavelength),.5*c.wavelength,241)
    reference=np.exp(2j*np.pi/c.wavelength*c.thickness)
    return dict(configuration=c,solution=solution,x=x,z=z,index=c.refractive_index(x,z),
                field=solution.reconstruct(x,z),fmm=solution.transmission/reference,
                ideal=ideal.transmission/reference,profile=solution.reconstruct(x,[-c.thickness-depth])[0],
                ideal_profile=ideal.reconstruct(x,[-c.thickness-depth])[0],depth=depth,
                caption=f'Physical n+i*kappa={c.n_absorber:g}+{c.kappa:g}i | lambda0={c.wavelength:g} um, air exterior, M={c.M}',
                status=f'R={solution.reflectance:.5f}, T={solution.transmittance:.5f}, A={solution.absorption:.5f}; R+T+A={solution.energy_balance:.9f}. Check M convergence.')


def draw(data,figure,axes,colorbars):
    c=data['configuration'];s=data['solution'];x,z=data['x'],data['z']
    extent=[x[0],x[-1],z[0],z[-1]]
    image(figure,axes[0],data['index'].imag,extent,colorbars,title=r'Absorbing geometry $\operatorname{Im}n(x,z)$',label='kappa',cmap='viridis')
    image(figure,axes[1],abs(data['field'])**2,extent,colorbars,title=r'Solved full field $|U(x,z)|^2$',label='Relative intensity')
    for ax in axes[:2]:
        ax.axhline(0,color='cyan',lw=.8);ax.axhline(-c.thickness,color='cyan',lw=.8)
        ax.set(xlabel='x (um)',ylabel='z (um)')
    axes[2].plot(s.orders,abs(data['fmm']),'o-',color=BLUE,label='Absorbing FMM')
    axes[2].plot(s.orders,abs(data['ideal']),'s--',mfc='none',color=ORANGE,label='Ideal binary mask')
    axes[2].set(title=r'Exit magnitudes $|C_m/U_{inc,ref}|$',xlabel='Order m',ylabel='Magnitude',xlim=(-7.5,7.5))
    axes[3].plot(s.orders,phase(data['fmm']),'o',color=BLUE,label='Absorbing FMM')
    axes[3].plot(s.orders,phase(data['ideal']),'s',mfc='none',color=ORANGE,label='Ideal binary mask')
    axes[3].set(title='Exit phases, same incident reference',xlabel='Order m',ylabel='Phase (rad)',xlim=(-7.5,7.5),ylim=(-3.3,3.3))
    axes[4].plot(x,abs(data['profile'])**2,color=BLUE,label='Absorbing FMM')
    axes[4].plot(x,abs(data['ideal_profile'])**2,'--',color=ORANGE,label='Ideal mask propagation')
    axes[4].set(title=f'Same output plane: depth={data["depth"]:g} um',xlabel='x (um)',ylabel='Relative intensity')
    axes[5].bar(['Reflection R','Transmission T','Absorption A'],[s.reflectance,s.transmittance,s.absorption],color=['0.5',BLUE,ORANGE])
    axes[5].set(title='Power fractions R, T, A (volume loss)',ylabel='Incident power fraction',ylim=(0,1.05))
    for ax in axes[2:5]:ax.legend(fontsize=7);ax.grid(True,alpha=.2)


parameters=dict(wavelength=wavelength,period=period,opening_width=opening_width,thickness=thickness,
                absorber_index=absorber_index,kappa=kappa,M=M,output_depth=output_depth)

if __name__=='__main__':
    controls=[('opening_width','Opening (um)',.05,.5,None),('thickness','h (um)',.01,.5,None),
              ('absorber_index','Absorber n',1,3,None),('kappa','Absorber kappa',0,3,None),
              ('M','Fourier M',0,48,1),('output_depth','Output depth (um)',0,1,None)]
    experiment=run(parameters,controls,compute,draw,(2,3),'Absorbing scalar FMM versus ideal binary mask',interactive=interactive,plot_backend=plot_backend)
