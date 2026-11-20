"""Single-layer, normal-incidence scalar Helmholtz grating scattering.

Lengths use one common unit; wavelength is in vacuum. With exp(-i*omega*t),
the incident wave travels from z > 0 toward -z. The grating occupies -h < z < 0.
This scalar model matches U and dU/dz, not general vector Maxwell boundary data.
See scalar_grating.md for the derivation, reference planes and limitations.
"""

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy import linalg


def _real_scalar(value, name, *, positive=False):
    try:
        if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
            raise ValueError
        array = np.asarray(value, dtype=float)
        if array.ndim != 0 or not np.isfinite(array):
            raise ValueError
        result = float(array)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite real scalar.') from error
    if positive and result <= 0:
        raise ValueError(f'{name} must be positive.')
    return result


def _samples(value, name):
    try:
        if np.iscomplexobj(value):
            raise ValueError
        array = np.asarray(value, dtype=float)
        if array.ndim > 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite scalar or nonempty 1D array.') from error
    return np.atleast_1d(array)


@dataclass(frozen=True)
class ScalarGratingConfiguration:
    """Rectangular ridge centred on x=0 in every period (normal incidence).

    Defaults use micrometres. Retain orders -M,...,M, including evanescent
    orders. Increasing M is necessary to assess convergence; M=10 is only
    a starting value. Positive real indices and thickness are required.
    """

    wavelength: float = .633
    period: float = 1.5
    thickness: float = .3
    duty_cycle: float = .5
    n_incident: float = 1.
    n_output: float = 1.
    n_ridge: float = 1.5
    n_groove: float = 1.
    M: int = 10

    def __post_init__(self):
        for name in ('wavelength', 'period', 'thickness', 'n_incident',
                     'n_output', 'n_ridge', 'n_groove'):
            object.__setattr__(self, name, _real_scalar(getattr(self, name), name,
                                                      positive=True))
        duty = _real_scalar(self.duty_cycle, 'duty_cycle')
        if not 0 <= duty <= 1:
            raise ValueError('duty_cycle must lie in [0, 1].')
        object.__setattr__(self, 'duty_cycle', duty)
        if isinstance(self.M, (bool, np.bool_)) or not isinstance(self.M, (int, np.integer)) or self.M < 0:
            raise ValueError('M must be a nonnegative integer; orders are -M,...,M.')
        object.__setattr__(self, 'M', int(self.M))

    def permittivity_coefficients(self, p):
        """Analytic epsilon_p for integer p, with exp(+i*p*2*pi*x/period).

        This describes the layer only. Coefficients up to |p|=2*M enter the
        convolution matrix even though the field retains only |m|<=M.
        """
        p = np.asarray(p)
        if not np.issubdtype(p.dtype, np.integer):
            raise ValueError('p must contain integer Fourier indices.')
        f = self.duty_cycle
        if f in (0., 1.):
            index = self.n_groove if f == 0 else self.n_ridge
            return np.asarray((p == 0)*index**2, dtype=np.complex128)
        return np.asarray((p == 0)*self.n_groove**2
                          + (self.n_ridge**2-self.n_groove**2)*f*np.sinc(p*f),
                          dtype=np.complex128)

    def refractive_index(self, x_samples, z_samples):
        """Exact prescribed n(x,z), shape (nz,nx), not a truncated Fourier sum.

        At z=0 return the incident index; at z=-thickness return the output
        index. At each ridge edge use the half-open interval [-f*d/2,f*d/2).
        """
        x, z = _samples(x_samples, 'x_samples'), _samples(z_samples, 'z_samples')
        wrapped = np.remainder(x/self.period+.5, 1.)-.5
        ridge = (wrapped >= -self.duty_cycle/2) & (wrapped < self.duty_cycle/2)
        layer = np.where(ridge, self.n_ridge, self.n_groove)
        return np.where(z[:, None] >= 0, self.n_incident,
                        np.where(z[:, None] <= -self.thickness, self.n_output, layer))


def _exterior_branch(index, alpha):
    """Dimensionless positive-real/positive-imaginary outgoing root."""
    squared = index**2-alpha**2
    tolerance = 64*np.finfo(float).eps*np.maximum(index**2, alpha**2)
    grazing = abs(squared) <= tolerance
    squared = np.where(grazing, 0., squared)
    kinds = np.where(grazing, 'grazing',
                     np.where(squared > 0, 'propagating', 'evanescent'))
    return np.sqrt(squared.astype(np.complex128)), kinds


def _layer_basis(q, depth, length):
    """Bounded modal basis and derivatives in s=-k0*z, on 0<=s<=length.

    Travelling waves are anchored at opposite boundaries. Near q*length=0,
    use cos(q*s), sin(q*s)/(q*length) instead: the zero-q limit is 1,s/length.
    Both solutions survive at cutoff, without a growing exponential.
    """
    depth = np.asarray(depth)[:, None]
    near_zero = abs(q*length) < .1
    first = np.exp(1j*depth*q)
    second = np.exp(1j*(length-depth)*q)
    dfirst, dsecond = 1j*q*first, -1j*q*second
    small_q = q[near_zero]
    argument = depth*small_q
    first[:, near_zero] = np.cos(argument)
    second[:, near_zero] = depth/length*np.sinc(argument/np.pi)
    dfirst[:, near_zero] = -small_q*np.sin(argument)
    dsecond[:, near_zero] = np.cos(argument)/length
    return first, second, dfirst, dsecond


@dataclass(frozen=True)
class ScalarGratingSolution:
    """Solved complex amplitudes, scalar flux efficiencies and field access.

    All order arrays have length 2*M+1 in ascending order. Reflection is
    referenced to z=0; transmission to z=-thickness. beta_* are positive-root
    longitudinal magnitudes, not signed kz (transmitted kz=-beta_output).
    Arrays are read-only so the field and reported coefficients stay consistent.
    """

    configuration: ScalarGratingConfiguration
    orders: np.ndarray
    kx: np.ndarray
    beta_incident: np.ndarray
    beta_output: np.ndarray
    incident_kinds: np.ndarray
    output_kinds: np.ndarray
    reflection: np.ndarray
    transmission: np.ndarray
    reflection_efficiency: np.ndarray
    transmission_efficiency: np.ndarray
    coupling_matrix: np.ndarray
    condition_number: float
    boundary_residual: float
    _k0: float = field(repr=False)
    _q: np.ndarray = field(repr=False)
    _vectors: np.ndarray = field(repr=False)
    _amplitudes: np.ndarray = field(repr=False)
    eigenvector_condition: float = 1.
    eigen_residual: float = 0.

    def __post_init__(self):
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    @property
    def power_balance(self):
        """Sum of outgoing reflected and transmitted scalar power fractions."""
        return float(np.sum(self.reflection_efficiency)+np.sum(self.transmission_efficiency))

    def harmonics(self, z_samples, *, derivative=0):
        """Total A_m(z), or dA_m/dz for derivative=1; always shape (nz,2*M+1).

        Above the layer this includes the prescribed incident zeroth order
        as well as reflection. Below it there is only outgoing transmission.
        Interface values use the exterior solution. No evanescent filtering.
        """
        z = _samples(z_samples, 'z_samples')
        if derivative not in (0, 1):
            raise ValueError('derivative must be 0 (field) or 1 (d/dz).')
        h = self.configuration.thickness
        result = np.empty((z.size, self.orders.size), dtype=np.complex128)
        above, below = z >= 0, z <= -h
        inside = ~(above | below)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                reflected = np.exp(1j*z[above, None]*self.beta_incident)*self.reflection
                if derivative:
                    reflected *= 1j*self.beta_incident
                incident = np.exp(-1j*self.beta_incident[self.configuration.M]*z[above])
                if derivative:
                    incident *= -1j*self.beta_incident[self.configuration.M]
                reflected[:, self.configuration.M] += incident
                result[above] = reflected
                transmitted = np.exp(-1j*(z[below, None]+h)*self.beta_output)*self.transmission
                if derivative:
                    transmitted *= -1j*self.beta_output
                result[below] = transmitted
                f, g, df, dg = _layer_basis(self._q, -self._k0*z[inside], self._k0*h)
                if derivative:
                    f, g = -self._k0*df, -self._k0*dg
                result[inside] = (f*self._amplitudes[0]+g*self._amplitudes[1]) @ self._vectors.T
            except FloatingPointError as error:
                raise ValueError('Field samples exceed the supported numerical range.') from error
        if not np.all(np.isfinite(result)):
            raise RuntimeError('Nonfinite harmonic amplitudes; check configuration and sampling.')
        return result

    def at_reference_plane(self, z_ref):
        """Return C_m=A_m(z_ref), shape (2*M+1,), for one finite real z_ref.

        Use z_ref<=-thickness for an outgoing output-plane field. On the
        incident side C includes incident plus reflected waves; their total
        magnitudes need not stay fixed when that reference plane moves.
        """
        return self.harmonics(_real_scalar(z_ref, 'z_ref'))[0]

    def reconstruct(self, x_samples, z_samples):
        """Complex U(x,z)=sum_m A_m(z)*exp(i*kx_m*x), shape (nz,nx)."""
        x = _samples(x_samples, 'x_samples')
        with np.errstate(over='raise', invalid='raise'):
            try:
                result = self.harmonics(z_samples) @ np.exp(1j*self.kx[:, None]*x)
            except FloatingPointError as error:
                raise ValueError('Field samples exceed the supported numerical range.') from error
        if not np.all(np.isfinite(result)):
            raise RuntimeError('Nonfinite reconstructed field.')
        return result


def _match_boundaries(matrix, rhs):
    """Equilibrated dense solve, with conditioning and backward-error checks."""
    singular_message = ('Scalar grating boundary matching is singular or numerically unresolved. '
                        'An exact grazing threshold or a trapped mode can make the outgoing '
                        'problem nonunique; change wavelength/period slightly and check convergence.')
    row_scale = np.max(abs(matrix), axis=1)
    if np.any(row_scale == 0):
        raise np.linalg.LinAlgError(singular_message)
    scaled = matrix/row_scale[:, None]
    column_scale = np.max(abs(scaled), axis=0)
    if np.any(column_scale == 0):
        raise np.linalg.LinAlgError(singular_message)
    scaled /= column_scale
    singular_values = linalg.svdvals(scaled)
    condition = float(singular_values[0]/singular_values[-1]) if singular_values[-1] else np.inf
    if not np.isfinite(condition) or condition >= 1/np.finfo(float).eps:
        raise np.linalg.LinAlgError(singular_message)
    if condition > 1e12:
        warnings.warn(f'Ill-conditioned scalar grating boundary solve (condition={condition:.3g}); '
                      'complex coefficients may be inaccurate.', RuntimeWarning, stacklevel=3)
    amplitudes = linalg.solve(scaled, rhs/row_scale)/column_scale
    residual = float(linalg.norm(matrix @ amplitudes-rhs, np.inf)
                     / (linalg.norm(matrix, np.inf)*linalg.norm(amplitudes, np.inf)
                        + linalg.norm(rhs, np.inf)))
    if not np.all(np.isfinite(amplitudes)) or not np.isfinite(residual):
        raise RuntimeError('Nonfinite scalar grating boundary solution.')
    if residual > 1e-10:
        raise RuntimeError(f'Scalar grating boundary matching failed (relative residual={residual:.3g}).')
    return amplitudes, condition, residual


def solve_scalar_grating(configuration):
    """Solve the unit-incident-field scattering BVP for one rectangular layer.

    The real lossless convolution matrix is Hermitian. Its eigenvectors
    decouple A''+Q*A=0 inside the layer. Two boundary conditions determine
    both modal solutions; reflected/transmitted orders are outputs, not inputs.
    This dense solver costs O((2*M+1)**3) and is intended for modest truncations.
    """
    if not isinstance(configuration, ScalarGratingConfiguration):
        raise TypeError('configuration must be a ScalarGratingConfiguration.')
    return _solve_layer(configuration)


def _solve_layer(configuration, *, absorbing=False):
    """Shared scalar matching engine; exterior media must remain real/lossless.

    The absorbing configuration is validated by absorbing_grating.py. Keep
    the Hermitian lossless path unchanged. Passive layers use a general
    eigensystem, with residual/conditioning checks and decaying modal roots.
    """
    c = configuration
    orders = np.arange(-c.M, c.M+1)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            k0 = 2*np.pi/c.wavelength
            alpha = orders*(c.wavelength/c.period)
            length = k0*c.thickness
            coupling = c.permittivity_coefficients(orders[:, None]-orders[None, :])-np.diag(alpha**2)
            if not np.isfinite(length) or not np.all(np.isfinite(coupling)):
                raise ValueError('Configuration exceeds the supported numerical range.')
            eigen_condition, eigen_residual = 1., 0.
            if absorbing:
                eigenvalues, vectors = linalg.eig(coupling)
                eigen_condition = float(np.linalg.cond(vectors))
                eigen_residual = float(linalg.norm(coupling @ vectors-vectors*eigenvalues)
                                      / (linalg.norm(coupling)*linalg.norm(vectors)))
                if not np.isfinite(eigen_condition) or eigen_condition >= 1/np.finfo(float).eps:
                    raise np.linalg.LinAlgError('Absorbing layer has an unresolved eigenbasis.')
                if eigen_condition > 1e10:
                    warnings.warn('Ill-conditioned absorbing eigenbasis; check order convergence.',
                                  RuntimeWarning, stacklevel=3)
                if eigen_residual > 1e-10:
                    raise RuntimeError('Absorbing layer eigensystem residual exceeds 1e-10.')
                tolerance = 256*np.finfo(float).eps*max(1., linalg.norm(coupling, 2))
                if np.min(eigenvalues.imag) < -tolerance:
                    raise RuntimeError('Passive layer produced growing eigenmodes.')
                # Roundoff near the real axis must not select a growing root.
                eigenvalues = eigenvalues.real+1j*np.maximum(eigenvalues.imag, 0.)
            else:
                eigenvalues, vectors = linalg.eigh(coupling)
            q = np.sqrt(eigenvalues.astype(np.complex128))
            bi, incident_kinds = _exterior_branch(c.n_incident, alpha)
            bo, output_kinds = _exterior_branch(c.n_output, alpha)
            f, g, df, dg = _layer_basis(q, np.array([0., length]), length)
            # Eliminate r and t using continuity of U. The remaining Robin
            # conditions in s=-k0*z are A_s+i*bi*A=2i*bi*e0 (top),
            # A_s-i*bo*A=0 (bottom). Derivatives are divided by k0.
            top = np.hstack((vectors*df[0]+1j*bi[:, None]*vectors*f[0],
                             vectors*dg[0]+1j*bi[:, None]*vectors*g[0]))
            bottom = np.hstack((vectors*df[1]-1j*bo[:, None]*vectors*f[1],
                                vectors*dg[1]-1j*bo[:, None]*vectors*g[1]))
            matrix = np.vstack((top, bottom))
            rhs = np.zeros(2*orders.size, dtype=np.complex128)
            rhs[c.M] = 2j*bi[c.M]
            amplitudes, condition, residual = _match_boundaries(matrix, rhs)
            amplitudes = amplitudes.reshape(2, -1)
            reflection = vectors @ (f[0]*amplitudes[0]+g[0]*amplitudes[1])
            reflection[c.M] -= 1
            transmission = vectors @ (f[1]*amplitudes[0]+g[1]*amplitudes[1])
            eta_r = np.where(incident_kinds == 'propagating', bi.real/bi[c.M].real*abs(reflection)**2, 0.)
            eta_t = np.where(output_kinds == 'propagating', bo.real/bi[c.M].real*abs(transmission)**2, 0.)
            physical_coupling = k0**2*coupling
            kx, beta_i, beta_o = k0*alpha, k0*bi, k0*bo
        except (FloatingPointError, OverflowError) as error:
            raise ValueError('Configuration exceeds the supported numerical range.') from error
    for value in (reflection, transmission, eta_r, eta_t, physical_coupling, kx, beta_i, beta_o):
        if not np.all(np.isfinite(value)):
            raise RuntimeError('Nonfinite scalar grating result; rescale lengths or check configuration.')
    if not absorbing and abs(np.sum(eta_r)+np.sum(eta_t)-1) > 1e-8:
        warnings.warn('Scalar flux balance differs from unity by more than 1e-8; '
                      'inspect conditioning and threshold proximity.', RuntimeWarning, stacklevel=2)
    return ScalarGratingSolution(c, orders, kx, beta_i, beta_o, incident_kinds, output_kinds,
                                 reflection, transmission, eta_r, eta_t, physical_coupling,
                                 condition, residual, k0, q, vectors, amplitudes,
                                 eigen_condition, eigen_residual)
