"""Ideal aperture at z=0, propagated to the physical absorber output planes."""

from dataclasses import dataclass, field, replace

import numpy as np

from .absorbing_grating import AbsorbingGratingConfiguration, AbsorbingGratingSolution, solve_absorbing_grating
from .binary_transmission import binary_transmission_coefficients, binary_transmission_field
from .scalar_grating import _exterior_branch, _real_scalar, _samples
from .thin_grating import ThinPhaseGrating, compare_grating_models


FIELD_FFT_ATOL = 1e-12  # Absolute coefficient units; incident field amplitude is one.


def field_fourier_coefficients(field, x_samples, period, orders):
    """Extract field coefficients from ordered, endpoint-excluded period samples.

    For x_j=x_start+j*period/N and U=sum_m C_m exp(i*2*pi*m*x/period),
    C_m=FFT(U)[m % N]/N * exp(-i*2*pi*m*x_start/period).
    The origin correction retains absolute phase for centred or shifted grids.
    This transforms the complex field, not its amplitude or intensity. It is
    a representation check for a sampled modal solution, not a scattering solve.
    Requested orders must lie strictly below the sampling Nyquist frequency.
    """
    p = _real_scalar(period, 'period', positive=True)
    x = _samples(x_samples, 'x_samples')
    u = np.asarray(field, dtype=complex)
    m = np.asarray(orders)
    if u.ndim != 1 or u.size != x.size or x.size < 2 or not np.all(np.isfinite(u)):
        raise ValueError('field must contain one finite complex value per sample, with at least two samples.')
    if not np.allclose((x-x[0])/p, np.arange(x.size)/x.size, rtol=1e-10, atol=1e-12):
        raise ValueError('Samples must ascend uniformly over one period, excluding the repeated endpoint.')
    if m.ndim != 1 or m.size == 0 or not np.issubdtype(m.dtype, np.integer) or np.any(abs(m) >= x.size/2):
        raise ValueError('orders must be a nonempty integer array strictly below the sampling Nyquist frequency.')
    return np.fft.fft(u)[m % x.size]/x.size * np.exp(-2j*np.pi*m*x[0]/p)


def ideal_amplitude_transmission(configuration, orders=None):
    """Binary transmission coefficients at z_obj=0, including opening shift."""
    if not isinstance(configuration, AbsorbingGratingConfiguration):
        raise TypeError('configuration must be an AbsorbingGratingConfiguration.')
    c=configuration
    m=np.arange(-c.M,c.M+1) if orders is None else np.asarray(orders)
    coefficients=binary_transmission_coefficients(m,c.open_fraction,1.,0.)
    return coefficients*np.exp(-2j*np.pi*m*c.opening_shift/c.period)


def ideal_amplitude_coefficients(configuration, orders=None):
    """Ideal coefficients at z=-h: t_m exp(i beta_m h), from a mask at z=0."""
    c=configuration
    tm=ideal_amplitude_transmission(c,orders)
    m=np.arange(-c.M,c.M+1) if orders is None else np.asarray(orders)
    beta,_=_exterior_branch(1.,m*c.wavelength/c.period)
    return tm*np.exp(1j*(2*np.pi/c.wavelength)*beta*c.thickness)


def ideal_amplitude_object_field(configuration, x_samples):
    """Exact binary boundary at z_obj=0 under unit normal incidence."""
    if not isinstance(configuration, AbsorbingGratingConfiguration):
        raise TypeError('configuration must be an AbsorbingGratingConfiguration.')
    c=configuration
    return binary_transmission_field(_samples(x_samples,'x_samples')-c.opening_shift,
                                    c.period,c.open_fraction,1.,0.)


def ideal_amplitude_exit_field(configuration, x_samples):
    """Retained outgoing field at z=-h, after propagating the boundary from z=0."""
    model=IdealAmplitudeGrating(configuration)
    return model.reconstruct(x_samples,[-model.configuration.thickness])[0]


@dataclass(frozen=True)
class IdealAmplitudeGrating(ThinPhaseGrating):
    """Binary mask at z_obj=0; transmission remains referenced to physical z=-h.

    The inherited field interface is retained for common-plane comparisons.
    Propagation is valid for all z<=0, with evanescent terms decaying downward.
    No reflected field or material absorption is predicted by this prescription.
    """
    configuration: AbsorbingGratingConfiguration
    object_coefficients: np.ndarray = field(init=False)

    def __post_init__(self):
        if not isinstance(self.configuration,AbsorbingGratingConfiguration):
            raise TypeError('configuration must be an AbsorbingGratingConfiguration.')
        c=self.configuration
        m=np.arange(-c.M,c.M+1)
        tm=ideal_amplitude_transmission(c,m)
        k0=2*np.pi/c.wavelength
        beta,kinds=_exterior_branch(1.,m*c.wavelength/c.period)
        beta=k0*beta
        coefficients=tm*np.exp(1j*beta*c.thickness)
        for name,value in dict(orders=m,kx=2*np.pi*m/c.period,beta_output=beta,
            output_kinds=kinds,object_coefficients=tm,transmission=coefficients,
            transmission_efficiency=np.where(kinds=='propagating',beta.real/k0*abs(coefficients)**2,0.)).items():
            value.setflags(write=False)
            object.__setattr__(self,name,value)

    def harmonics(self,z_samples):
        """C_m(z)=t_m exp(-i beta_m z) below the chosen thin-mask plane z=0."""
        z=_samples(z_samples,'z_samples')
        if np.any(z>0):
            raise ValueError('Thin-mask output planes must satisfy z <= 0.')
        with np.errstate(over='raise',invalid='raise'):
            values=self.object_coefficients*np.exp(-1j*z[:,None]*self.beta_output)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite propagated ideal coefficients.')
        return values


class AbsorbingComparisonData:
    """A bounded thickness sweep; every case is solved once and reused.

    For every thickness h, the comparison plane is z=-h-distance. Thus the
    sweep follows the exit at fixed output distance, not a fixed laboratory
    z. Unit incident phase stays fixed at z=0 for every solve. Field images
    are cached only for the requested cases by the view. Supply an existing
    absorbing solution to reuse it at its original thickness; other cases
    are solved with that solution's configuration and the requested depth.
    """

    def __init__(self, configuration=None, thicknesses=(.03, .08, .15, .3, .5, .75, 1.), *, solution=None):
        if solution is not None:
            if not isinstance(solution, AbsorbingGratingSolution):
                raise TypeError('solution must be an existing AbsorbingGratingSolution.')
            if configuration is not None and configuration != solution.configuration:
                raise ValueError('The supplied configuration must match the existing solution.')
            configuration = solution.configuration
        self.configuration = AbsorbingGratingConfiguration() if configuration is None else configuration
        c = self.configuration
        samples = _samples(thicknesses, 'thicknesses')
        if np.any(samples <= 0) or len(samples) > 20:
            raise ValueError('Use 1 to 20 positive thickness samples.')
        self.thicknesses = np.unique(np.r_[samples, c.thickness])
        self.thicknesses.setflags(write=False)
        self.solutions = tuple(solution if solution is not None and h == c.thickness else
                               solve_absorbing_grating(replace(c, thickness=float(h))) for h in self.thicknesses)
        self.ideal_models = tuple(IdealAmplitudeGrating(s.configuration) for s in self.solutions)

    def planes(self, distance=0.):
        distance = float(distance)
        if not np.isfinite(distance) or distance < 0:
            raise ValueError('distance must be finite and nonnegative.')
        # Shared comparison record calls the ideal boundary prediction 'thin'.
        return tuple(compare_grating_models(s, -s.configuration.thickness-distance, thin_model=ideal)
                     for s, ideal in zip(self.solutions, self.ideal_models))
