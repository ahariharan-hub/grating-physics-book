"""Passive single-layer scalar FMM for a rectangular 2D periodic lattice.

Time convention exp(-i omega t); illumination travels toward -z. Layer:
-h < z < 0. Matches U and dU/dz, retaining the existing scalar assumptions.
Lengths share one unit; kx_incident and ky_incident have inverse-length units.
"""
from dataclasses import dataclass, field, replace
import warnings
import numpy as np
from scipy import linalg
from scipy.integrate import quad
from .scalar_grating import (_real_scalar, _samples, _exterior_branch,
                             _layer_basis, _match_boundaries)


def _passive_index(value, name):
    """Positive real part, nonnegative imaginary part for exp(-i omega t)."""
    try:
        if isinstance(value, (bool, np.bool_)) or np.asarray(value).ndim != 0:
            raise ValueError
        result = complex(value)
        if not np.isfinite(result) or result.real <= 0 or result.imag < 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite passive index (Re > 0, Im >= 0).') from error
    return result if result.imag else result.real


@dataclass(frozen=True)
class ScalarGrating2DConfiguration:
    """Centered rectangular inclusion; exterior indices remain real/lossless.

    For an amplitude-mask comparison use n_inclusion=1 (clear opening) and
    n_background=n+1j*kappa (absorber). Defaults preserve the dielectric case.
    """
    wavelength: float = .633
    period_x: float = 1.5
    period_y: float = 1.5
    thickness: float = .3
    fill_x: float = .5
    fill_y: float = .5
    n_incident: float = 1.
    n_output: float = 1.
    n_inclusion: complex = 1.5
    n_background: complex = 1.
    Mx: int = 2
    My: int = 2
    kx_incident: float = 0.
    ky_incident: float = 0.

    def __post_init__(self):
        for name in ('wavelength', 'period_x', 'period_y', 'thickness',
                     'n_incident', 'n_output'):
            object.__setattr__(self, name, _real_scalar(getattr(self, name), name, positive=True))
        for name in ('n_inclusion', 'n_background'):
            object.__setattr__(self, name, _passive_index(getattr(self, name), name))
        for name in ('fill_x', 'fill_y'):
            value = _real_scalar(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must lie in [0, 1].')
            object.__setattr__(self, name, value)
        for name in ('Mx', 'My'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer.')
            object.__setattr__(self, name, int(value))
        for name in ('kx_incident', 'ky_incident'):
            object.__setattr__(self, name, _real_scalar(getattr(self, name), name))
        k = 2*np.pi*self.n_incident/self.wavelength
        if np.hypot(self.kx_incident, self.ky_incident) >= k:
            raise ValueError('Incident zeroth order must be strictly propagating.')

    def permittivity_coefficients(self, p, q):
        """Analytic epsilon_pq of a centered rectangular inclusion."""
        p, q = np.broadcast_arrays(np.asarray(p), np.asarray(q))
        if not all(np.issubdtype(a.dtype, np.integer) for a in (p, q)):
            raise ValueError('Fourier indices must be integers.')
        rectangle = self.fill_x*self.fill_y*np.sinc(p*self.fill_x)*np.sinc(q*self.fill_y)
        # Exact uniform limits avoid floating-point sinc(integer) leakage.
        if self.fill_x == 1:
            rectangle = np.where(p == 0, rectangle, 0.)
        if self.fill_y == 1:
            rectangle = np.where(q == 0, rectangle, 0.)
        return np.asarray((p == 0)*(q == 0)*self.n_background**2 +
                          (self.n_inclusion**2-self.n_background**2)*rectangle, dtype=complex)

    def refractive_index(self, x_samples, y_samples, z_samples):
        """Exact index on a Cartesian grid, shape (nz, ny, nx)."""
        x, y, z = (_samples(a, n) for a, n in
                   ((x_samples, 'x'), (y_samples, 'y'), (z_samples, 'z')))
        wx = np.remainder(x/self.period_x+.5, 1.)-.5
        wy = np.remainder(y/self.period_y+.5, 1.)-.5
        ix = (wx >= -self.fill_x/2) & (wx < self.fill_x/2)
        iy = (wy >= -self.fill_y/2) & (wy < self.fill_y/2)
        layer = np.where(iy[:, None] & ix[None, :], self.n_inclusion, self.n_background)
        return np.where(z[:, None, None] >= 0, self.n_incident,
                        np.where(z[:, None, None] <= -self.thickness, self.n_output, layer))


@dataclass(frozen=True)
class ScalarGrating2DSolution:
    configuration: ScalarGrating2DConfiguration
    orders: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
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
    absorption: float = 0.
    absorption_quadrature_error: float = 0.
    eigenvector_condition: float = 1.
    eigen_residual: float = 0.

    def __post_init__(self):
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    @property
    def incident_index(self):
        return self.configuration.Mx*(2*self.configuration.My+1)+self.configuration.My

    @property
    def power_balance(self):
        """Outgoing R+T; equals 1-A for absorbing layers."""
        return float(self.reflection_efficiency.sum()+self.transmission_efficiency.sum())

    @property
    def reflectance(self):
        return float(self.reflection_efficiency.sum())

    @property
    def transmittance(self):
        return float(self.transmission_efficiency.sum())

    @property
    def energy_balance(self):
        """R+T+A, with A independently integrated through the layer."""
        return self.power_balance+self.absorption

    def downward_flux(self, z_samples):
        """Cell-averaged downward scalar flux / incident normal flux."""
        u, du = self.harmonics(z_samples), self.harmonics(z_samples, derivative=1)
        return -np.imag(np.sum(u.conj()*du, axis=1))/self.beta_incident[self.incident_index].real

    def harmonics(self, z_samples, *, derivative=0):
        """Total A_m(z), or dA_m/dz for derivative=1; always shape (nz,K).

        Above the layer this includes the prescribed incident zeroth order
        as well as reflection. Below it there is only outgoing transmission.
        Interface values use the exterior solution. No evanescent filtering.
        """
        z = _samples(z_samples, 'z_samples')
        if derivative not in (0, 1):
            raise ValueError('derivative must be 0 (field) or 1 (d/dz).')
        h = self.configuration.thickness
        result = np.empty((z.size, len(self.orders)), dtype=np.complex128)
        above, below = z >= 0, z <= -h
        inside = ~(above | below)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                reflected = np.exp(1j*z[above, None]*self.beta_incident)*self.reflection
                if derivative:
                    reflected *= 1j*self.beta_incident
                incident = np.exp(-1j*self.beta_incident[self.incident_index]*z[above])
                if derivative:
                    incident *= -1j*self.beta_incident[self.incident_index]
                reflected[:, self.incident_index] += incident
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
        """Return C_m=A_m(z_ref), shape (K,), for one finite real z_ref.

        Use z_ref<=-thickness for an outgoing output-plane field. On the
        incident side C includes incident plus reflected waves; their total
        magnitudes need not stay fixed when that reference plane moves.
        """
        return self.harmonics(_real_scalar(z_ref, 'z_ref'))[0]

    def reconstruct(self, x_samples, y_samples, z_samples):
        """Cartesian field U(z,y,x), including the Bloch phase."""
        x, y = _samples(x_samples, 'x'), _samples(y_samples, 'y')
        return np.einsum('za,ay,ax->zyx', self.harmonics(z_samples),
                         np.exp(1j*self.ky[:, None]*y),
                         np.exp(1j*self.kx[:, None]*x), optimize=True)


def solve_scalar_grating_2d(configuration):
    """Solve one unit-amplitude incident Bloch wave on a passive layer.

    Orders are flattened with m outermost and n innermost. Dense cost is
    cubic in K=(2*Mx+1)*(2*My+1); independently check both order cutoffs.
    """
    if not isinstance(configuration, ScalarGrating2DConfiguration):
        raise TypeError('configuration must be a ScalarGrating2DConfiguration.')
    c = configuration
    m, n = np.meshgrid(np.arange(-c.Mx, c.Mx+1), np.arange(-c.My, c.My+1), indexing='ij')
    orders = np.column_stack((m.ravel(), n.ravel()))
    count = len(orders)
    center = c.Mx*(2*c.My+1)+c.My
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        k0 = 2*np.pi/c.wavelength
        ax = c.kx_incident/k0 + orders[:, 0]*c.wavelength/c.period_x
        ay = c.ky_incident/k0 + orders[:, 1]*c.wavelength/c.period_y
        transverse = np.hypot(ax, ay)
        delta = orders[:, None, :]-orders[None, :, :]
        epsilon = c.permittivity_coefficients(delta[..., 0], delta[..., 1])
        coupling = epsilon-np.diag(transverse**2)
        length = k0*c.thickness
        if not np.isfinite(length) or not np.all(np.isfinite(coupling)):
            raise ValueError('Configuration exceeds the supported numerical range.')
        absorbing = bool(np.any(epsilon.imag))
        eigen_condition, eigen_residual = 1., 0.
        if absorbing:
            eigenvalues, vectors = linalg.eig(coupling)
            eigen_condition = float(np.linalg.cond(vectors))
            eigen_residual = float(linalg.norm(coupling @ vectors-vectors*eigenvalues)
                                   / (linalg.norm(coupling)*linalg.norm(vectors)))
            if not np.isfinite(eigen_condition) or eigen_condition >= 1/np.finfo(float).eps:
                raise np.linalg.LinAlgError('Absorbing 2D layer has an unresolved eigenbasis.')
            if eigen_condition > 1e10:
                warnings.warn('Ill-conditioned absorbing 2D eigenbasis; check convergence.', RuntimeWarning)
            if not np.isfinite(eigen_residual) or eigen_residual > 1e-10:
                raise RuntimeError('Absorbing 2D layer eigensystem residual exceeds 1e-10.')
            tolerance = 256*np.finfo(float).eps*max(1., linalg.norm(coupling, 2))
            if np.min(eigenvalues.imag) < -tolerance:
                raise RuntimeError('Passive 2D layer produced growing eigenmodes.')
            eigenvalues = eigenvalues.real+1j*np.maximum(eigenvalues.imag, 0.)
        else:
            eigenvalues, vectors = linalg.eigh(coupling)
        q = np.sqrt(eigenvalues.astype(complex))
        bi, ki = _exterior_branch(c.n_incident, transverse)
        bo, ko = _exterior_branch(c.n_output, transverse)
        if ki[center] != 'propagating':
            raise ValueError('Incident wave is numerically grazing; move away from cutoff.')
        f, g, df, dg = _layer_basis(q, np.array([0., length]), length)
        top = np.hstack((vectors*df[0]+1j*bi[:, None]*vectors*f[0],
                         vectors*dg[0]+1j*bi[:, None]*vectors*g[0]))
        bottom = np.hstack((vectors*df[1]-1j*bo[:, None]*vectors*f[1],
                            vectors*dg[1]-1j*bo[:, None]*vectors*g[1]))
        rhs = np.zeros(2*count, complex)
        rhs[center] = 2j*bi[center]
        amplitudes, condition, residual = _match_boundaries(np.vstack((top, bottom)), rhs)
        amplitudes = amplitudes.reshape(2, count)
        reflection = vectors @ (f[0]*amplitudes[0]+g[0]*amplitudes[1])
        reflection[center] -= 1
        transmission = vectors @ (f[1]*amplitudes[0]+g[1]*amplitudes[1])
        er = np.where(ki == 'propagating', bi.real/bi[center].real*abs(reflection)**2, 0.)
        et = np.where(ko == 'propagating', bo.real/bi[center].real*abs(transmission)**2, 0.)
        arrays = (k0*ax, k0*ay, k0*bi, k0*bo, reflection, transmission, er, et, k0**2*coupling)
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise RuntimeError('Nonfinite scalar 2D grating result.')
    if not absorbing and abs(er.sum()+et.sum()-1) > 1e-8:
        warnings.warn('Scalar flux balance differs from unity; check conditioning.', RuntimeWarning)
    solution = ScalarGrating2DSolution(c, orders, *arrays[:4], ki, ko, *arrays[4:],
                                      condition, residual, k0, q, vectors, amplitudes,
                                      eigenvector_condition=eigen_condition, eigen_residual=eigen_residual)
    if absorbing:
        absorption, error = _volume_absorption_2d(solution)
        solution = replace(solution, absorption=absorption, absorption_quadrature_error=error)
        if absorption < -1e-9 or abs(solution.energy_balance-1) > max(1e-7, 10*error):
            raise RuntimeError('Absorbing 2D flux/volume-loss check failed; check convergence and conditioning.')
    return solution


def _volume_absorption_2d(solution):
    """Cell-averaged k0^2 integral Im(epsilon)|U|^2 dz / beta_incident_00.

    The material convolution integrates the truncated field exactly in x,y;
    only depth is integrated numerically. A is not defined as 1-R-T.
    """
    c, orders = solution.configuration, solution.orders
    delta = orders[:, None, :]-orders[None, :, :]
    epsilon = c.permittivity_coefficients(delta[..., 0], delta[..., 1])
    loss = (epsilon-epsilon.conj().T)/(2j)
    scale = solution._k0**2/solution.beta_incident[solution.incident_index].real

    def integrand(depth):
        u = solution.harmonics(-depth)[0]
        return float(scale*np.vdot(u, loss @ u).real)

    skin = min(c.thickness/2, 1/(solution._k0*max(np.max(solution._q.imag), 1e-12)))
    return quad(integrand, 0., c.thickness, epsabs=2e-10, epsrel=2e-9,
                points=sorted(set([skin, c.thickness-skin])), limit=250)
