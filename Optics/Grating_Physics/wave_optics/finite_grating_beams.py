"""Finite-illumination thin-mask experiment; lengths in micrometres.

Time factor exp(-i omega t); grating z=0, output z=-s with s>=0.
The existing +z ASM multiplier is reused at positive depth s: signed
k_z=-sqrt(k^2-k_x^2), so exp(i*k_z*z)=exp(i*sqrt(...)*s).
No scattering efficiencies, reflection or mirror transformations are modeled.
"""
from dataclasses import dataclass
import numpy as np
from .binary_transmission import binary_transmission_field, binary_transmission_coefficients
from .scalar_propagation import transfer_function, spectrum_geometry


@dataclass(frozen=True)
class FiniteGratingConfiguration:
    wavelength: float = 1.
    period: float = 4.
    radius: float = 8.
    angle_deg: float = 0.
    maximum_depth: float = 80.
    samples_per_period: int = 64
    minimum_window: float = 512.

    def __post_init__(self):
        for name in ('wavelength','period','radius','maximum_depth','minimum_window'):
            value=getattr(self,name)
            if not np.isscalar(value) or not np.isfinite(value) or value<=0:
                raise ValueError(name+' must be positive and finite.')
        if not np.isfinite(self.angle_deg) or abs(self.angle_deg)>=80:
            raise ValueError('Use a finite incidence angle strictly between -80 and 80 degrees.')
        if isinstance(self.samples_per_period,bool) or int(self.samples_per_period)!=self.samples_per_period or self.samples_per_period<16:
            raise ValueError('Use an integer samples_per_period >= 16.')
        if self.period/self.samples_per_period>self.wavelength/4:
            raise ValueError('Need dx <= wavelength/4 to resolve intensity of propagating fields.')


def intensity_overlap(first,second,dx):
    """Normalized integral of |U_m||U_n|; NaN if either field has zero norm.

    Integrals are transverse intensity per unit y length, not total power.
    Fields have matching shapes [...,x]; no field phase is discarded upstream.
    """
    a,b=np.asarray(first),np.asarray(second)
    if a.shape!=b.shape or a.ndim<1 or not np.isfinite(dx) or dx<=0:
        raise ValueError('Require matching fields and positive dx.')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Fields must be finite.')
    pa=np.sum(abs(a)**2,axis=-1)*dx
    pb=np.sum(abs(b)**2,axis=-1)*dx
    denominator=np.sqrt(pa*pb)
    overlap=np.divide(np.sum(abs(a)*abs(b),axis=-1)*dx,denominator,
        out=np.full_like(denominator,np.nan,dtype=float),where=denominator>0)
    return np.clip(overlap,0,1)


class FiniteGratingExperiment:
    """A padded periodic calculation with explicit approximate order windows."""

    def __init__(self,configuration=FiniteGratingConfiguration()):
        self.configuration=c=configuration
        self.dx=c.period/c.samples_per_period
        # Fixed display range is distinct from this much wider FFT domain.
        width=max(c.minimum_window,12*c.radius+5*c.maximum_depth)
        self.size=2**int(np.ceil(np.log2(width/self.dx)))
        self.x=(np.arange(self.size)-self.size/2+.5)*self.dx
        self.fx=np.fft.fftfreq(self.size,self.dx)
        self.k_x=2*np.pi*self.fx
        self.k=2*np.pi/c.wavelength
        self.G=2*np.pi/c.period
        self.k_x_i=self.k*np.sin(np.deg2rad(c.angle_deg))
        self.incident=np.exp(-self.x**2/c.radius**2)*np.exp(1j*self.k_x_i*self.x)
        self.mask=binary_transmission_field(self.x,c.period,.5,1.,0.)
        self.boundary=self.mask*self.incident
        self.spectrum=np.fft.fft(self.boundary)
        self.signed_k_z=-spectrum_geometry(self.fx,0.,c.wavelength)['kz']
        self.propagating=abs(self.k_x)<self.k
        low=int(np.floor((-self.k-self.k_x_i)/self.G))-1
        high=int(np.ceil((self.k-self.k_x_i)/self.G))+1
        labels=np.arange(low,high+1)
        centers=self.k_x_i+labels*self.G
        # Even nonzero harmonics vanish in the ideal 50%-open mask. Their
        # finite-window leakage is not advertised as a distinct strong beam.
        coefficients=binary_transmission_coefficients(labels,.5,1.,0.)
        keep=(abs(centers)<self.k*(1-1e-10))&(abs(coefficients)>1e-10)
        self.orders=labels[keep]
        self.centers=centers[keep]

    def window(self,order):
        if order not in self.orders:raise ValueError('Choose a nonzero, strictly propagating order center.')
        center=self.k_x_i+int(order)*self.G
        # Disjoint half-open cells bounded midway to adjacent ideal centers;
        # truncate at propagation cutoff to isolate propagating contributions.
        return ((self.k_x>=center-self.G/2)&(self.k_x<center+self.G/2)&self.propagating)

    def propagate_spectrum(self,spectrum,depths):
        spectrum=np.asarray(spectrum,dtype=complex)
        depths=np.asarray(depths,dtype=float)
        if spectrum.shape!=(self.size,) or depths.ndim!=1 or not len(depths) or not np.isfinite(spectrum).all() or not np.isfinite(depths).all() or np.any(depths<0):
            raise ValueError('Require one finite spectrum on this grid and nonnegative depth samples.')
        return np.stack([np.fft.ifft(spectrum*transfer_function(self.fx,0.,self.configuration.wavelength,float(s))) for s in depths])

    def selected_fields(self,pair=(0,1),depths=None):
        if len(pair)!=2 or pair[0]==pair[1]:raise ValueError('Choose two distinct order centers.')
        depths=np.linspace(0,self.configuration.maximum_depth,81) if depths is None else np.asarray(depths,dtype=float)
        if np.any(depths>self.configuration.maximum_depth):raise ValueError('Depth exceeds the configured padded experiment.')
        windows=np.stack([self.window(m) for m in pair])
        spectra=windows*self.spectrum
        fields=np.stack([self.propagate_spectrum(s,depths) for s in spectra])
        full=self.propagate_spectrum(self.spectrum,depths)
        cross=2*np.real(fields[0].conj()*fields[1])
        overlap=intensity_overlap(fields[0],fields[1],self.dx)
        powers=np.sum(abs(fields)**2,axis=-1)*self.dx
        full_norm=np.sum(abs(full)**2,axis=-1)*self.dx
        spectral_norm=np.sum(abs(self.spectrum)**2)/self.size*self.dx
        prop_norm=np.sum(abs(self.spectrum[self.propagating])**2)/self.size*self.dx
        # Signed spectral flux, with the common physical prefactor omitted.
        # Evanescent samples have zero real normal flux.
        flux_weights=(-self.signed_k_z.real)/self.k
        spectral_flux=np.sum(flux_weights*abs(self.spectrum)**2)/self.size*self.dx
        edge=abs(self.x)>.45*(self.size*self.dx)
        edge_fraction=np.max(np.sum(abs(full[:,edge])**2,axis=-1)*self.dx/np.maximum(full_norm,1e-300))
        warnings=[]
        midpoint_tail=float(np.exp(-(self.configuration.radius*self.G/4)**2))
        if midpoint_tail>.01:warnings.append('Neighboring Gaussian spectral replicas overlap: order isolation is approximate and window-dependent.')
        for order,window in zip(pair,windows):
            center=self.k_x_i+order*self.G
            if abs(center)+self.G/2>=self.k:
                warnings.append(f'Order {order:+d}: window reaches the grazing cutoff; clipped propagating content can distort its beam.')
        if edge_fraction>1e-4:warnings.append('FFT edge energy is appreciable: enlarge the computation window and compare before interpreting separation.')
        significant=self.propagating&(abs(self.spectrum)>abs(self.spectrum).max()*1e-4)
        slopes=np.abs(self.k_x[significant]/(-self.signed_k_z[significant].real))
        estimated_reach=(float(depths.max())*float(slopes.max()) if len(slopes) else 0.)+4*self.configuration.radius
        if estimated_reach>.45*self.size*self.dx:
            warnings.append('Significant near-grazing spectrum may reach the FFT boundary: padding convergence is required even if the edge-intensity diagnostic is small.')
        return dict(x=self.x.copy(),z=-depths.copy(),depth=depths.copy(),dx=self.dx,
            fields=fields,total=fields.sum(axis=0),full=full,cross=cross,
            overlap=overlap,integrated_intensity=powers,full_integrated_intensity=full_norm,
            windows=windows,spectra=spectra,k_x=self.k_x.copy(),k_z=self.signed_k_z.copy(),
            pair=tuple(int(m) for m in pair),warnings=warnings,
            diagnostics=dict(edge_fraction=float(edge_fraction),estimated_transverse_reach=estimated_reach,midpoint_amplitude_tail=midpoint_tail,
                boundary_integrated_intensity=float(spectral_norm),propagating_integrated_intensity=float(prop_norm),
                downward_flux=float(spectral_flux),fft_width=self.size*self.dx,delta_k_x=2*np.pi/(self.size*self.dx)),
            metadata=dict(time_convention='exp(-i omega t)',output_half_space='z<=0',
                length_unit='um',invariant_axis='y',order_definition='half-open midpoint windows clipped to |kx|<k',
                model='prescribed 50%-open thin amplitude mask',configuration=self.configuration))
