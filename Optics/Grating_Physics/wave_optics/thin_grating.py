"""Analytical thin phase grating and common-plane scalar BVP comparison.

Unit normal-incidence illumination has U_i(x, 0)=1 and exp(-i omega t).
The centred ridge occupies a fraction f of each period. This local model
prescribes exp(i*k0*n(x)*h) at z=-h; it neglects reflection and lateral
coupling. Equal real exterior indices are required. No scattering solve,
FFT, phase alignment, or power renormalization is performed here.
"""

from dataclasses import dataclass, field

import numpy as np

from .scalar_grating import (
    ScalarGratingConfiguration, ScalarGratingSolution,
    _exterior_branch, _real_scalar, _samples,
)
from .binary_transmission import (
    binary_transmission_coefficients, binary_transmission_field, dielectric_phase_levels,
)


def _configuration(configuration):
    if not isinstance(configuration, ScalarGratingConfiguration):
        raise TypeError('configuration must be a ScalarGratingConfiguration.')
    if configuration.n_incident != configuration.n_output:
        raise ValueError('The thin phase-mask comparison requires equal exterior indices.')
    return configuration


def thin_mask_coefficients(configuration, orders=None):
    """Exact Fourier coefficients at z=-h, with the absolute exit-plane phase.

    C_m=t_g*delta_m0+(t_r-t_g)*f*sinc(m*f), with NumPy's normalized sinc.
    The default retains -M,...,M; optional integer indices need not be
    contiguous. The mask itself has infinitely many harmonics in general.
    """
    c = _configuration(configuration)
    m = np.arange(-c.M, c.M+1) if orders is None else np.asarray(orders)
    tr, tg = dielectric_phase_levels(c)
    return binary_transmission_coefficients(m, c.duty_cycle, tr, tg)


def thin_mask_exit_field(configuration, x_samples):
    """Exact prescribed mask field, before Fourier truncation, shape (nx,).

    Ridge edges use the same half-open interval [-f*d/2, f*d/2) as the
    material geometry. A Fourier sum instead tends to the midpoint at a jump.
    """
    c = _configuration(configuration)
    tr, tg = dielectric_phase_levels(c)
    return binary_transmission_field(x_samples, c.period, c.duty_cycle, tr, tg)


@dataclass(frozen=True)
class ThinPhaseGrating:
    """Fixed analytical mask with the scalar solver's output-field interface.

    ``transmission`` is referenced to z=-h, not z=0. ``beta_output`` is the
    nonnegative-real/nonnegative-imaginary root; signed transmitted kz=-beta.
    Array results are read-only. Propagating efficiencies use longitudinal
    flux weighting and are not forced to sum to one. There is no predicted
    reflected field: its omission is a model assumption, not a BVP result.
    """

    configuration: ScalarGratingConfiguration
    orders: np.ndarray = field(init=False)
    kx: np.ndarray = field(init=False)
    beta_output: np.ndarray = field(init=False)
    output_kinds: np.ndarray = field(init=False)
    transmission: np.ndarray = field(init=False)
    transmission_efficiency: np.ndarray = field(init=False)

    def __post_init__(self):
        c = _configuration(self.configuration)
        m = np.arange(-c.M, c.M+1)
        coefficients = thin_mask_coefficients(c)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                k0 = 2*np.pi/c.wavelength
                alpha = m*(c.wavelength/c.period)
                beta, kinds = _exterior_branch(c.n_output, alpha)
                kx, beta = k0*alpha, k0*beta
                efficiency = np.where(kinds == 'propagating',
                                      beta.real/(k0*c.n_incident)*abs(coefficients)**2, 0.)
            except (FloatingPointError, OverflowError) as error:
                raise ValueError('Thin-mask parameters exceed the supported numerical range.') from error
        for name, value in dict(orders=m, kx=kx, beta_output=beta, output_kinds=kinds,
                                transmission=coefficients, transmission_efficiency=efficiency).items():
            if name != 'output_kinds' and not np.all(np.isfinite(value)):
                raise ValueError('Nonfinite thin-mask result; check parameters and units.')
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def harmonics(self, z_samples):
        """Outgoing coefficients at z<=-h, shape (nz,2*M+1).

        A downward distance D=-h-z gives C_m(z)=C_m(-h)*exp(+i*beta_m*D).
        For beta=i*kappa this decays, rather than grows, away from the mask.
        """
        z = _samples(z_samples, 'z_samples')
        if np.any(z > -self.configuration.thickness):
            raise ValueError('Thin-mask output planes must satisfy z <= -thickness.')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                distance = -self.configuration.thickness-z
                values = self.transmission*np.exp(1j*distance[:, None]*self.beta_output)
            except (FloatingPointError, OverflowError) as error:
                raise ValueError('Propagation distance exceeds the supported numerical range.') from error
        if not np.all(np.isfinite(values)):
            raise ValueError('Nonfinite propagated thin-mask coefficients.')
        return values

    def at_reference_plane(self, z_ref):
        """Complex coefficients at one finite scalar output-plane coordinate."""
        return self.harmonics(_real_scalar(z_ref, 'z_ref'))[0]

    def reconstruct(self, x_samples, z_samples):
        """Retained Fourier sum, shape (nz,nx), with no sampling-based mask FFT."""
        x = _samples(x_samples, 'x_samples')
        coefficients = self.harmonics(z_samples)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                values = coefficients @ np.exp(1j*self.kx[:, None]*x)
            except (FloatingPointError, OverflowError) as error:
                raise ValueError('Field samples exceed the supported numerical range.') from error
        if not np.all(np.isfinite(values)):
            raise ValueError('Nonfinite reconstructed thin-mask field.')
        return values


@dataclass(frozen=True)
class GratingComparisonPlane:
    """Computed common-plane values; phases are radians, missing phases NaN.

    ``difference`` is HBVP minus thin, without aligning either complex phase.
    ``phase_difference`` is arg(HBVP*conj(thin)) only when BOTH amplitudes
    exceed the absolute threshold in units of the incident amplitude.
    Efficiencies vanish for grazing/evanescent orders. HBVP R and T totals
    include every retained propagating order, not just a displayed subset.
    """

    z_ref: float
    orders: np.ndarray
    kinds: np.ndarray
    thin: np.ndarray
    hbvp: np.ndarray
    difference: np.ndarray
    thin_phase: np.ndarray
    hbvp_phase: np.ndarray
    phase_difference: np.ndarray
    thin_efficiency: np.ndarray
    hbvp_efficiency: np.ndarray
    hbvp_reflectance: float
    hbvp_transmittance: float

    def __post_init__(self):
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    @property
    def thin_transmittance(self):
        return float(np.sum(self.thin_efficiency))

    @property
    def hbvp_power_balance(self):
        return self.hbvp_reflectance+self.hbvp_transmittance


def compare_grating_models(solution, z_ref, *, thin_model=None, phase_floor=1e-6):
    """Compare an EXISTING BVP solution with its analytical thin phase mask.

    Supply ``thin_model`` to reuse it across plane translations. Neither this
    helper nor ThinPhaseGrating calls the scattering solver. No comparison
    of phases is reported for coefficients below ``phase_floor`` (>0).
    """
    if not isinstance(solution, ScalarGratingSolution):
        raise TypeError('solution must be an existing ScalarGratingSolution.')
    thin_model = ThinPhaseGrating(solution.configuration) if thin_model is None else thin_model
    if not isinstance(thin_model, ThinPhaseGrating):
        raise TypeError('thin_model must be a ThinPhaseGrating.')
    if thin_model.configuration != solution.configuration:
        raise ValueError('Both models must describe the same geometry and retained orders.')
    z = _real_scalar(z_ref, 'z_ref')
    floor = _real_scalar(phase_floor, 'phase_floor', positive=True)
    thin = thin_model.at_reference_plane(z)  # Validates the shared output half-space.
    hbvp = solution.at_reference_plane(z)
    valid_thin, valid_hbvp = abs(thin) >= floor, abs(hbvp) >= floor
    return GratingComparisonPlane(
        z, solution.orders, solution.output_kinds, thin, hbvp, hbvp-thin,
        np.where(valid_thin, np.angle(thin), np.nan),
        np.where(valid_hbvp, np.angle(hbvp), np.nan),
        np.where(valid_thin & valid_hbvp, np.angle(hbvp*np.conj(thin)), np.nan),
        thin_model.transmission_efficiency, solution.transmission_efficiency,
        float(np.sum(solution.reflection_efficiency)),
        float(np.sum(solution.transmission_efficiency)),
    )
