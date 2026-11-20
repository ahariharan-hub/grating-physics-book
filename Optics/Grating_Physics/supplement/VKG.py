# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:22:47 2026

@author: aharihar

VKG: Vector_Kirchhoff_Geometry

Utilities for plotting:

1. A vector-defined 1D binary grating
2. A direction-cosine sphere
3. The grating basis vectors
4. An optional quartz substrate

Coordinate convention
---------------------
g_hat : periodic direction
l_hat : line or groove direction
n_hat : optical-axis / mask-normal direction

For the default DUV orientation, n_hat points along -z. The quartz-absorber
interface is z = 0, with quartz on the +z side and absorbers on the -z side.
"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# Geometry
# ============================================================

def plot_quartz_substrate(
    ax,
    g_hat,
    l_hat,
    n_hat,
    width,
    line_length,
    thickness,
    color="lightcyan",
    alpha=0.2,
):
    """Draw a quartz slab extending from the grating plane along -n_hat.

    The slab is centered in the grating plane, with ``width`` along g_hat
    and ``line_length`` along l_hat. For n_hat = [0, 0, -1], it occupies
    0 <= z <= thickness. Basis vectors must be mutually perpendicular.

    ``thickness`` is a positive display thickness in the same units as the
    grating geometry. It need not represent the full physical substrate.
    This illustration adds no refraction or material response to the
    Kirchhoff calculation. Returns the substrate's Poly3DCollection.
    """
    basis = np.asarray([g_hat, l_hat, n_hat], dtype=float)
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("Substrate basis vectors must have three finite components.")
    norms = np.linalg.norm(basis, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Substrate basis vectors must be nonzero.")
    basis = basis / norms[:, None]
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-10, rtol=0):
        raise ValueError("Substrate basis vectors must be mutually perpendicular.")
    dimensions = np.asarray([width, line_length, thickness], dtype=float)
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
        raise ValueError("Substrate width, line_length, and thickness must be positive.")

    g_hat, l_hat, n_hat = basis
    dg = 0.5 * width * g_hat
    dl = 0.5 * line_length * l_hat
    depth = -thickness * n_hat

    # Attachment face at the mask plane and outer face on the quartz side.
    interface = np.array([-dg - dl, dg - dl, dg + dl, -dg + dl])
    vertices = np.vstack([interface, interface + depth])
    face_indices = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    substrate = Poly3DCollection(
        vertices[face_indices],
        facecolor=color,
        edgecolor="teal",
        linewidth=0.6,
        alpha=alpha,
        label="Quartz",
    )
    ax.add_collection3d(substrate)
    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    ax.text(
        *(0.85 * dg - 0.85 * dl + depth),
        "Quartz",
        color="teal",
        fontsize=9,
        ha="right",
        va="bottom",
        zorder=1000,
    )
    return substrate


def plot_grating_3d(
    ax,
    g_hat,
    period,
    duty,
    line_length,
    height,
    n_lines=5,
    color="royalblue",
    alpha=0.8,
    substrate_thickness=None,
    substrate_color="lightcyan",
    substrate_alpha=0.2,
):
    """
    Plot a 1D binary grating with lines extending along the y-axis.

    Parameters
    ----------
    ax : matplotlib 3D axis
        Axis on which the grating is plotted.
    g_hat : array-like, shape (3,)
        Unit vector defining the periodic direction.
        The current bar3d implementation expects g_hat along x.
    period : float
        Grating period.
    duty : float
        Clear fraction of one period.
    line_length : float
        Length of each absorber line along y.
    height : float
        Absorber thickness. Bars occupy -height <= z <= 0, with the
        quartz-absorber interface at z = 0.
    n_lines : int, default=5
        Number of absorber lines.
    color : str, default="royalblue"
        Absorber color.
    alpha : float, default=0.8
        Absorber transparency.
    substrate_thickness : float or None, default=None
        Optional quartz display thickness along +z. None hides the slab.
        The slab covers n_lines periods and the full line_length.
    substrate_color : str, default="lightcyan"
        Quartz color.
    substrate_alpha : float, default=0.2
        Quartz transparency. The substrate is visualization geometry only.
    """
    g_hat = np.asarray(g_hat, dtype=float)

    if substrate_thickness is not None:
        plot_quartz_substrate(
            ax, g_hat, [0.0, 1.0, 0.0], [0.0, 0.0, -1.0],
            width=n_lines * period,
            line_length=line_length,
            thickness=substrate_thickness,
            color=substrate_color,
            alpha=substrate_alpha,
        )

    absorber_width = (1.0 - duty) * period

    m_values = np.arange(n_lines) - (n_lines - 1) / 2

    for m in m_values:
        center = m * period * g_hat

        x0 = center[0] - absorber_width / 2
        y0 = -line_length / 2
        z0 = -height

        ax.bar3d(
            x0,
            y0,
            z0,
            absorber_width,
            line_length,
            height,
            color=color,
            alpha=alpha,
            shade=True,
        )


def plot_grating_3d_gen(
    ax,
    g_hat,
    l_hat,
    n_hat,
    period,
    duty,
    line_length,
    height,
    n_lines=5,
    color="royalblue",
    alpha=0.8,
    substrate_thickness=None,
    substrate_color="lightcyan",
    substrate_alpha=0.2,
):
    """
    Plot a 1D binary grating in an arbitrary 3D orientation.

    g_hat : periodic direction
    l_hat : absorber-line direction
    n_hat : absorber-height direction

    substrate_thickness : optional positive quartz display thickness
        The slab extends along -n_hat, opposite the absorbers, and covers
        n_lines periods and the full line_length. None hides the slab.
        This is visualization geometry only.
    substrate_color, substrate_alpha : quartz color and transparency
    """

    g_hat = np.asarray(g_hat, dtype=float)
    l_hat = np.asarray(l_hat, dtype=float)
    n_hat = np.asarray(n_hat, dtype=float)

    # Normalize the basis vectors
    g_hat = g_hat / np.linalg.norm(g_hat)
    l_hat = l_hat / np.linalg.norm(l_hat)
    n_hat = n_hat / np.linalg.norm(n_hat)

    if substrate_thickness is not None:
        plot_quartz_substrate(
            ax, g_hat, l_hat, n_hat,
            width=n_lines * period,
            line_length=line_length,
            thickness=substrate_thickness,
            color=substrate_color,
            alpha=substrate_alpha,
        )
    
    # Check basis orientation
    rhs = np.cross(g_hat, l_hat)
    print("cross(g_hat, l_hat) =", rhs)    
    print("n_hat =", n_hat)

    absorber_width = (1.0 - duty) * period

    # Absorber-line indices centered around zero
    m_values = np.arange(n_lines) - (n_lines - 1) / 2

    for m in m_values:

        center = m * period * g_hat

        dg = 0.5 * absorber_width * g_hat
        dl = 0.5 * line_length * l_hat
        dh = height * n_hat

        # Four bottom vertices
        v0 = center - dg - dl
        v1 = center + dg - dl
        v2 = center + dg + dl
        v3 = center - dg + dl

        # Four top vertices
        v4 = v0 + dh
        v5 = v1 + dh
        v6 = v2 + dh
        v7 = v3 + dh

        faces = [
            [v0, v1, v2, v3],  # bottom
            [v4, v5, v6, v7],  # top
            [v0, v1, v5, v4],  # side
            [v1, v2, v6, v5],  # side
            [v2, v3, v7, v6],  # side
            [v3, v0, v4, v7],  # side
        ]

        absorber = Poly3DCollection(
            faces,
            facecolor=color,
            edgecolor="navy",
            linewidth=0.5,
            alpha=alpha,
        )

        ax.add_collection3d(absorber)

def plot_direction_cosine_sphere(
    ax,
    radius,
    n_theta=40,
    n_phi=80,
    color="gray",
    linewidth=0.4,
    alpha=0.5,
):
    """
    Plot a direction-cosine sphere centered at the origin.

    Parameters
    ----------
    ax : matplotlib 3D axis
        Axis on which the sphere is plotted.
    radius : float
        Display radius. The physical direction cosines remain normalized.
    n_theta : int, default=40
        Number of polar-angle samples.
    n_phi : int, default=80
        Number of azimuth-angle samples.
    color : str, default="gray"
        Wireframe color.
    linewidth : float, default=0.4
        Wireframe line width.
    alpha : float, default=0.5
        Wireframe transparency.
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)

    theta, phi = np.meshgrid(theta, phi)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    ax.plot_wireframe(
        x,
        y,
        z,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )


def plot_basis_vectors(
    ax,
    g_hat,
    l_hat,
    n_hat,
    sphere_radius,
    g_length=None,
    l_length=None,
    n_length=None,
):
    """
    Plot the grating basis vectors.

    Parameters
    ----------
    ax : matplotlib 3D axis
        Axis on which the vectors are plotted.
    g_hat : array-like, shape (3,)
        Periodic-direction unit vector.
    l_hat : array-like, shape (3,)
        Line-direction unit vector.
    n_hat : array-like, shape (3,)
        Optical-axis or mask-normal unit vector.
    sphere_radius : float
        Direction-cosine sphere display radius.
    g_length, l_length, n_length : float or None
        Display lengths. Defaults are based on sphere_radius.
    """
    g_hat = np.asarray(g_hat, dtype=float)
    l_hat = np.asarray(l_hat, dtype=float)
    n_hat = np.asarray(n_hat, dtype=float)

    if g_length is None:
        g_length = sphere_radius

    if l_length is None:
        l_length = sphere_radius

    if n_length is None:
        n_length = 2.0 * sphere_radius

    ax.quiver(
        0, 0, 0,
        *(g_length * g_hat),
        color="red",
        linewidth=3,
        arrow_length_ratio=0.2,
    )

    ax.quiver(
        0, 0, 0,
        *(l_length * l_hat),
        color="green",
        linewidth=3,
        arrow_length_ratio=0.2,
    )

    # Draw the normal through the complete direction-cosine sphere.
    n_start = -sphere_radius * n_hat

    ax.quiver(
        *n_start,
        *(n_length * n_hat),
        color="black",
        linewidth=3,
        arrow_length_ratio=0.1,
    )

    ax.text(
        *(1.1 * g_length * g_hat),
        "g_hat",
        color="red",
        fontsize=14,
    )

    ax.text(
        *(1.1 * l_length * l_hat),
        "l_hat",
        color="green",
        fontsize=14,
    )

    n_endpoint = n_start + n_length * n_hat

    ax.text(
        *(1.1 * n_endpoint),
        "n_hat",
        color="black",
        fontsize=14,
    )
    
   
    

# ============================================================
# Illumination
# ============================================================
        
def plot_incident_plane_wave(
    ax,
    theta_deg,
    phi_deg,
    sphere_radius,
    n_hat=np.array([0.0, 0.0, -1.0]),
    wavelength=193.0,
    ray_color="magenta",
    s_color="olive",
    p_color="brown",
    basis_length=500.0,
    show_basis=True,
    show_label=True,
    reference_hat=None,
):
    """Plot illumination using angles measured in the fixed laboratory frame.

    theta_deg is measured from -z and phi_deg from +x toward +y.
    At normal incidence, reference_hat selects the transverse s basis;
    it defaults to laboratory +y. This selects a basis, not the physical
    source polarization. See compute_polarization_basis for the fallback.
    """

    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    k_hat = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        -np.cos(theta),
    ])

    k_hat /= np.linalg.norm(k_hat)

    k0 = 2 * np.pi / wavelength
    k_inc = k0 * k_hat

    # source point on sphere
    tail = -sphere_radius * k_hat

    ax.quiver(
        tail[0],
        tail[1],
        tail[2],
        sphere_radius * k_hat[0],
        sphere_radius * k_hat[1],
        sphere_radius * k_hat[2],
        color=ray_color,
        linewidth=3,
        arrow_length_ratio=0.1,
    )

    ax.scatter(*tail, color=ray_color, s=80)

    if show_label:
        ax.text(*tail, r"$k_{inc}$", color=ray_color)

    # -------------------------
    # local TE/TM basis
    # -------------------------

    s_hat, p_hat = compute_polarization_basis(k_hat, n_hat, reference_hat)

    if show_basis:

        L = basis_length

        ax.quiver(
            tail[0],
            tail[1],
            tail[2],
            L * s_hat[0],
            L * s_hat[1],
            L * s_hat[2],
            color=s_color,
            linewidth=3,
        )

        ax.quiver(
            tail[0],
            tail[1],
            tail[2],
            L * p_hat[0],
            L * p_hat[1],
            L * p_hat[2],
            color=p_color,
            linewidth=3,
        )

        ax.text(
            *(tail + 1.1 * L * s_hat),
            "TE / s",
            color=s_color,
        )

        ax.text(
            *(tail + 1.1 * L * p_hat),
            "TM / p",
            color=p_color,
        )

    return k_hat, s_hat, p_hat, k_inc    


# ============================================================
# Diffraction
# ============================================================

def compute_diffraction_orders(
    k_hat_inc,
    wavelength,
    period,
    g_hat,
    n_hat,
    orders=range(-5, 6),
    propagation_sign=+1,
    include_evanescent=True,
    tol=1e-12,
):
    """
    Compute diffraction-order wavevectors for a 1D grating.

    The tangential grating equation is

        k_m,t = k_inc,t + m G

    where

        G = (2*pi / period) * g_hat

    The normal component is obtained from

        |k_m| = k0 = 2*pi / wavelength.

    Parameters
    ----------
    k_hat_inc : array_like, shape (3,)
        Incident propagation unit vector. It must point in the physical
        propagation direction toward the grating.

    wavelength : float
        Wavelength. Must use the same length unit as ``period``.

    period : float
        Grating period.

    g_hat : array_like, shape (3,)
        Unit vector in the periodic direction of the grating.

    n_hat : array_like, shape (3,)
        Unit surface-normal vector.

        For your current geometry:

            n_hat = [0, 0, -1]

        points toward the transmitted propagation side.

    orders : iterable of int, optional
        Diffraction orders to calculate.

    propagation_sign : {+1, -1}, optional
        Selects the normal propagation direction.

            +1 : normal component along +n_hat
            -1 : normal component along -n_hat

        Use +1 for your current transmitted-order convention.

    include_evanescent : bool, optional
        If True, include non-propagating orders in the output.
        Evanescent orders have ``k_normal`` represented as an imaginary
        value but do not have a real propagation direction ``k_hat``.

    tol : float, optional
        Numerical tolerance for propagation classification.

    Returns
    -------
    results : list of dict
        One dictionary per diffraction order with keys including:

            order
            propagating
            k0
            G
            k_t
            k_t_magnitude
            k_normal
            k_vec
            k_hat
            direction_cosines
    """

    # --------------------------------------------------------
    # Validate and normalize inputs
    # --------------------------------------------------------

    k_hat_inc = np.asarray(k_hat_inc, dtype=float)
    g_hat = np.asarray(g_hat, dtype=float)
    n_hat = np.asarray(n_hat, dtype=float)

    if k_hat_inc.shape != (3,):
        raise ValueError("k_hat_inc must be a three-component vector.")

    if g_hat.shape != (3,):
        raise ValueError("g_hat must be a three-component vector.")

    if n_hat.shape != (3,):
        raise ValueError("n_hat must be a three-component vector.")

    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")

    if period <= 0:
        raise ValueError("period must be positive.")

    if propagation_sign not in (+1, -1):
        raise ValueError("propagation_sign must be +1 or -1.")

    k_hat_inc_norm = np.linalg.norm(k_hat_inc)
    g_hat_norm = np.linalg.norm(g_hat)
    n_hat_norm = np.linalg.norm(n_hat)

    if k_hat_inc_norm < tol:
        raise ValueError("k_hat_inc must be nonzero.")

    if g_hat_norm < tol:
        raise ValueError("g_hat must be nonzero.")

    if n_hat_norm < tol:
        raise ValueError("n_hat must be nonzero.")

    k_hat_inc = k_hat_inc / k_hat_inc_norm
    g_hat = g_hat / g_hat_norm
    n_hat = n_hat / n_hat_norm

    # --------------------------------------------------------
    # Ensure grating direction lies in grating plane
    # --------------------------------------------------------

    g_normal_component = np.dot(g_hat, n_hat)

    if abs(g_normal_component) > 1e-10:
        raise ValueError(
            "g_hat must lie in the grating plane, so dot(g_hat, n_hat) "
            "must be approximately zero."
        )

    # --------------------------------------------------------
    # Wavevector and reciprocal grating vector
    # --------------------------------------------------------

    k0 = 2.0 * np.pi / wavelength
    G_magnitude = 2.0 * np.pi / period
    G = G_magnitude * g_hat

    k_inc = k0 * k_hat_inc

    # Tangential incident wavevector
    k_inc_normal_scalar = np.dot(k_inc, n_hat)

    k_inc_t = (
        k_inc
        - k_inc_normal_scalar * n_hat
    )

    results = []

    # --------------------------------------------------------
    # Calculate each diffraction order
    # --------------------------------------------------------

    for m in orders:

        m = int(m)

        # Tangential grating equation
        k_t = k_inc_t + m * G

        k_t_squared = np.dot(k_t, k_t)
        k_t_magnitude = np.sqrt(max(k_t_squared, 0.0))

        normal_squared = k0**2 - k_t_squared

        # Allow tiny negative values caused by floating-point error
        if normal_squared >= -tol * k0**2:

            normal_squared = max(normal_squared, 0.0)
            k_normal_magnitude = np.sqrt(normal_squared)

            k_normal = (
                propagation_sign
                * k_normal_magnitude
            )

            k_vec = k_t + k_normal * n_hat
            k_hat = k_vec / k0

            result = {
                "order": m,
                "propagating": True,
                "k0": k0,
                "G": G.copy(),
                "k_inc": k_inc.copy(),
                "k_inc_t": k_inc_t.copy(),
                "k_t": k_t.copy(),
                "k_t_magnitude": k_t_magnitude,
                "k_normal": k_normal,
                "k_vec": k_vec,
                "k_hat": k_hat,
                "direction_cosines": k_hat.copy(),
            }

        else:

            if not include_evanescent:
                continue

            decay_constant = np.sqrt(-normal_squared)

            # Complex normal component. The propagation sign is kept
            # explicit, although the physically decaying branch depends
            # on the selected spatial half-space and exp(+ik.r) convention.
            k_normal = (
                1j
                * propagation_sign
                * decay_constant
            )

            k_vec = (
                k_t.astype(complex)
                + k_normal * n_hat
            )

            result = {
                "order": m,
                "propagating": False,
                "k0": k0,
                "G": G.copy(),
                "k_inc": k_inc.copy(),
                "k_inc_t": k_inc_t.copy(),
                "k_t": k_t.copy(),
                "k_t_magnitude": k_t_magnitude,
                "k_normal": k_normal,
                "decay_constant": decay_constant,
                "k_vec": k_vec,
                "k_hat": None,
                "direction_cosines": None,
            }

        results.append(result)

    return results

def print_diffraction_orders(results):

    print("\nDiffraction Orders")
    print("-" * 80)

    for r in results:

        m = r["order"]

        if r["propagating"]:

            u, v, w = r["k_hat"]

            print(
                f"m={m:+2d}  "
                f"u={u:+.6f}  "
                f"v={v:+.6f}  "
                f"w={w:+.6f}"
            )

        else:

            print(
                f"m={m:+2d}  evanescent"
            )
            
def plot_diffraction_orders(
    ax,
    results,
    sphere_radius,
    color="darkorange",
    linewidth=2.0,
    arrow_length_ratio=0.08,
    marker_color="red",
    marker_size=50,
    show_arrows=True,
    show_points=True,
    show_labels=True,
    label_scale=1.05,
):
    """
    Plot propagating diffraction orders on a direction-cosine sphere.

    Each propagating diffraction order starts at the origin and ends at

        r_m = sphere_radius * k_hat_m

    Evanescent orders are not plotted because they do not have a real
    far-field propagation direction.

    Parameters
    ----------
    ax : matplotlib 3D axis
        Axis on which the diffraction orders are plotted.

    results : list of dict
        Output returned by ``compute_diffraction_orders``.

    sphere_radius : float
        Display radius of the direction-cosine sphere.

    color : str, optional
        Arrow color.

    linewidth : float, optional
        Width of diffraction-order arrows.

    arrow_length_ratio : float, optional
        Relative size of the arrowhead.

    marker_color : str, optional
        Color of order points on the sphere.

    marker_size : float, optional
        Size of order points.

    show_arrows : bool, optional
        Plot order vectors from the origin.

    show_points : bool, optional
        Plot order locations on the sphere.

    show_labels : bool, optional
        Label each propagating order with its integer order number.

    label_scale : float, optional
        Radial scale factor used to position labels slightly beyond
        the direction-cosine sphere.

    Returns
    -------
    artists : list
        Matplotlib artists created by the function.
    """

    if sphere_radius <= 0:
        raise ValueError("sphere_radius must be positive.")

    artists = []

    for result in results:

        if not result["propagating"]:
            continue

        m = result["order"]

        k_hat_m = np.asarray(
            result["k_hat"],
            dtype=float,
        )

        if k_hat_m.shape != (3,):
            raise ValueError(
                f"Diffraction order m={m} does not contain "
                "a valid three-component k_hat."
            )

        norm = np.linalg.norm(k_hat_m)

        if norm == 0:
            raise ValueError(
                f"Diffraction order m={m} has a zero k_hat vector."
            )

        # Ensure the plotted direction is exactly normalized.
        k_hat_m = k_hat_m / norm

        # Endpoint on the direction-cosine sphere.
        endpoint = sphere_radius * k_hat_m

        if show_arrows:

            arrow = ax.quiver(
                0.0,
                0.0,
                0.0,
                endpoint[0],
                endpoint[1],
                endpoint[2],
                color=color,
                linewidth=linewidth,
                arrow_length_ratio=arrow_length_ratio,
            )

            artists.append(arrow)

        if show_points:

            point = ax.scatter(
                endpoint[0],
                endpoint[1],
                endpoint[2],
                color=marker_color,
                s=marker_size,
                depthshade=False,
            )

            artists.append(point)

        if show_labels:

            label_position = label_scale * endpoint

            label = ax.text(
                label_position[0],
                label_position[1],
                label_position[2],
                f"m={m:+d}",
                color=marker_color,
                fontsize=10,
            )

            artists.append(label)

    return artists            


# ============================================================
# Local Order Bases
# ============================================================          

def _unit_vector(vector, name, tol):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-component vector.")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < tol:
        raise ValueError(f"{name} must have a finite, nonzero norm.")
    return vector / norm


def compute_polarization_basis(k_hat, n_hat, reference_hat=None, tol=1e-12):
    """Return transverse unit vectors s and p, with s x p = -k_hat.

    Away from normal incidence, s = normalize(k_hat x n_hat) and
    p = normalize(s x k_hat). Input vectors are normalized internally.

    At normal incidence the incidence plane is undefined. Project
    reference_hat (laboratory +y by default) perpendicular to k_hat.
    If that reference is parallel to k_hat, use the Cartesian axis least
    aligned with k_hat instead. tol is a positive numerical tolerance.
    """
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")
    k_hat = _unit_vector(k_hat, "k_hat", tol)
    n_hat = _unit_vector(n_hat, "n_hat", tol)
    if reference_hat is None:
        reference_hat = [0.0, 1.0, 0.0]
    reference_hat = _unit_vector(reference_hat, "reference_hat", tol)

    s_hat = np.cross(k_hat, n_hat)
    s_norm = np.linalg.norm(s_hat)
    if s_norm < tol:
        # Equivalent to reference - dot(reference, k) * k, without
        # subtracting nearly equal longitudinal components.
        s_hat = np.cross(k_hat, np.cross(reference_hat, k_hat))
        s_norm = np.linalg.norm(s_hat)
        if s_norm < tol:
            candidate = np.eye(3)[np.argmin(np.abs(k_hat))]
            s_hat = np.cross(k_hat, np.cross(candidate, k_hat))
            s_norm = np.linalg.norm(s_hat)

    s_hat = s_hat / s_norm
    p_hat = np.cross(s_hat, k_hat)
    p_hat = p_hat / np.linalg.norm(p_hat)
    return s_hat, p_hat


def compute_order_polarization_bases(
    results,
    n_hat,
    reference_hat=None,
    tol=1e-12,
):
    """
    Compute the local TE/TM basis for each propagating diffraction order.

    Basis convention
    ----------------
    s_hat_m = normalize(cross(k_hat_m, n_hat))
    p_hat_m = normalize(cross(s_hat_m, k_hat_m))

    Therefore:

        s_hat_m x p_hat_m = -k_hat_m

    This is the same convention used for the incident polarization basis.

    Parameters
    ----------
    results : list of dict
        Output from ``compute_diffraction_orders``.

    n_hat : array_like, shape (3,)
        Unit surface-normal vector.

    reference_hat : array_like, shape (3,), optional
        Fallback direction used when an order is normal to the grating
        surface and cross(k_hat_m, n_hat) is undefined.

        Defaults to laboratory +y. The reference is projected perpendicular
        to the order direction; a parallel reference is replaced by the
        least-aligned Cartesian axis. See compute_polarization_basis.

    tol : float, optional
        Numerical tolerance.

    Returns
    -------
    results_with_basis : list of dict
        Copy of the diffraction-order results. Propagating orders contain:

            s_hat
            p_hat

        Evanescent orders contain:

            s_hat = None
            p_hat = None
    """

    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")
    n_hat = _unit_vector(n_hat, "n_hat", tol)
    if reference_hat is None:
        reference_hat = [0.0, 1.0, 0.0]
    reference_hat = _unit_vector(reference_hat, "reference_hat", tol)

    results_with_basis = []

    for result in results:

        # Copy so the original results are not modified.
        result_new = result.copy()

        if not result["propagating"]:

            result_new["s_hat"] = None
            result_new["p_hat"] = None

            results_with_basis.append(result_new)
            continue

        k_hat_m = _unit_vector(
            result["k_hat"], f"Order m={result['order']} k_hat", tol
        )
        s_hat_m, p_hat_m = compute_polarization_basis(
            k_hat_m, n_hat, reference_hat, tol
        )

        result_new["k_hat"] = k_hat_m
        result_new["direction_cosines"] = k_hat_m.copy()
        result_new["s_hat"] = s_hat_m
        result_new["p_hat"] = p_hat_m

        results_with_basis.append(result_new)

    return results_with_basis  


def print_order_polarization_bases(results):

    print("\nDiffraction-Order Polarization Bases")
    print("-" * 100)

    for result in results:

        m = result["order"]

        if not result["propagating"]:
            print(f"m={m:+2d}  evanescent")
            continue

        k = np.array2string(
            result["k_hat"],
            precision=6,
            floatmode="fixed"
        )

        s = np.array2string(
            result["s_hat"],
            precision=6,
            floatmode="fixed"
        )

        p = np.array2string(
            result["p_hat"],
            precision=6,
            floatmode="fixed"
        )

        print(f"\nm={m:+2d}")
        print(f"  k_hat = {k}")
        print(f"  s_hat = {s}")
        print(f"  p_hat = {p}")

def plot_order_polarization_bases(
    ax,
    results,
    sphere_radius,
    basis_length=250.0,
    s_color="olive",
    p_color="brown",
    k_color="black",
    linewidth=2.0,
    arrow_length_ratio=0.15,
    show_s=True,
    show_p=True,
    show_k=True,
    show_order_labels=True,
    show_basis_labels=False,
):
    """
    Plot local (k,s,p) bases for propagating diffraction orders.

    Parameters
    ----------
    ax : matplotlib 3D axis

    results : list of dict
        Output from compute_order_polarization_bases().

    sphere_radius : float
        Radius used for the direction-cosine sphere.

    basis_length : float
        Display length of k/s/p vectors.

    show_s, show_p, show_k : bool
        Toggle individual basis vectors.

    show_order_labels : bool
        Label diffraction order numbers.

    show_basis_labels : bool
        Label k_m, s_m, p_m.
    """

    artists = []

    for result in results:

        if not result["propagating"]:
            continue

        m = result["order"]

        k_hat_m = np.asarray(result["k_hat"], dtype=float)
        s_hat_m = np.asarray(result["s_hat"], dtype=float)
        p_hat_m = np.asarray(result["p_hat"], dtype=float)

        # location of order on sphere
        base_point = sphere_radius * k_hat_m

        # --------------------------------------------------
        # k_m
        # --------------------------------------------------

        if show_k:

            k_arrow = ax.quiver(
                base_point[0],
                base_point[1],
                base_point[2],
                basis_length * k_hat_m[0],
                basis_length * k_hat_m[1],
                basis_length * k_hat_m[2],
                color=k_color,
                linewidth=linewidth,
                arrow_length_ratio=arrow_length_ratio,
            )

            artists.append(k_arrow)

            if show_basis_labels:

                pos = base_point + 1.1 * basis_length * k_hat_m

                artists.append(
                    ax.text(
                        pos[0],
                        pos[1],
                        pos[2],
                        f"$k_{{{m}}}$",
                        color=k_color,
                    )
                )

        # --------------------------------------------------
        # s_m
        # --------------------------------------------------

        if show_s:

            s_arrow = ax.quiver(
                base_point[0],
                base_point[1],
                base_point[2],
                basis_length * s_hat_m[0],
                basis_length * s_hat_m[1],
                basis_length * s_hat_m[2],
                color=s_color,
                linewidth=linewidth,
                arrow_length_ratio=arrow_length_ratio,
            )

            artists.append(s_arrow)

            if show_basis_labels:

                pos = base_point + 1.1 * basis_length * s_hat_m

                artists.append(
                    ax.text(
                        pos[0],
                        pos[1],
                        pos[2],
                        f"$s_{{{m}}}$",
                        color=s_color,
                    )
                )

        # --------------------------------------------------
        # p_m
        # --------------------------------------------------

        if show_p:

            p_arrow = ax.quiver(
                base_point[0],
                base_point[1],
                base_point[2],
                basis_length * p_hat_m[0],
                basis_length * p_hat_m[1],
                basis_length * p_hat_m[2],
                color=p_color,
                linewidth=linewidth,
                arrow_length_ratio=arrow_length_ratio,
            )

            artists.append(p_arrow)

            if show_basis_labels:

                pos = base_point + 1.1 * basis_length * p_hat_m

                artists.append(
                    ax.text(
                        pos[0],
                        pos[1],
                        pos[2],
                        f"$p_{{{m}}}$",
                        color=p_color,
                    )
                )

        # --------------------------------------------------
        # order label
        # --------------------------------------------------

        if show_order_labels:

            ax.text(
                base_point[0],
                base_point[1],
                base_point[2],
                f"{m:+d}",
                fontsize=9,
                color="black",
            )

    return artists


def check_order_bases(results):
    """Print orthogonality residuals and the s x p = -k handedness residual."""

    print("\nBasis Checks")
    print("-" * 80)

    for r in results:

        if not r["propagating"]:
            continue

        m = r["order"]

        k = r["k_hat"]
        s = r["s_hat"]
        p = r["p_hat"]

        print(
            f"m={m:+2d}  "
            f"k·s={np.dot(k,s):+.2e}  "
            f"k·p={np.dot(k,p):+.2e}  "
            f"s·p={np.dot(s,p):+.2e}"
        )

        handedness = np.linalg.norm(
            np.cross(s, p) + k
        )

        print(
            f"      |s×p+k| = {handedness:.2e}"
        )

def transform_field_to_order_bases(
    results,
    s_hat_inc,
    p_hat_inc,
    Es_inc,
    Ep_inc,
):
    """
    Project incident polarization onto each diffraction-order basis.

    Incident field:

        E_inc =
            Es_inc * s_hat_inc
          + Ep_inc * p_hat_inc

    For every propagating order m:

        Es_m = dot(E_inc, s_hat_m)
        Ep_m = dot(E_inc, p_hat_m)

    Results contain Es_proj, Ep_proj, and E_sp_proj for the local
    components. E_proj is their transverse reconstruction in laboratory
    coordinates, and E_vec stores the same projected vector. E_inc retains
    the original incident field separately for every order.

    Evanescent orders retain E_inc but have None for all projected fields.
    These are geometric projections; mask transmission coefficients and
    diffraction amplitude factors have not been applied.
    """

    E_inc = (
        Es_inc * np.asarray(s_hat_inc)
        +
        Ep_inc * np.asarray(p_hat_inc)
    )

    results_out = []

    for result in results:

        result_new = result.copy()
        result_new["E_inc"] = E_inc.copy()

        if not result["propagating"]:

            result_new["Es_proj"] = None
            result_new["Ep_proj"] = None
            result_new["E_sp_proj"] = None
            result_new["E_proj"] = None
            result_new["E_vec"] = None

            results_out.append(result_new)
            continue

        s_hat_m = result["s_hat"]
        p_hat_m = result["p_hat"]

        Es_m = np.dot(E_inc, s_hat_m)
        Ep_m = np.dot(E_inc, p_hat_m)

        result_new["Es_proj"] = Es_m
        result_new["Ep_proj"] = Ep_m
        result_new["E_sp_proj"] = np.array([Es_m, Ep_m])
        E_proj = Es_m * s_hat_m + Ep_m * p_hat_m
        result_new["E_proj"] = E_proj
        result_new["E_vec"] = E_proj.copy()

        results_out.append(result_new)

    return results_out

def print_order_field_projections(results):

    print("\nField Projections")
    print("-" * 80)

    for r in results:

        m = r["order"]

        if not r["propagating"]:
            continue

        print(
            f"m={m:+2d}  "
            f"Es_proj={r['Es_proj']:+.6f}  "
            f"Ep_proj={r['Ep_proj']:+.6f}"
        )
        
def compute_order_rotation_matrices(
    results,
    s_hat_inc,
    p_hat_inc,
):
    """
    Compute TE/TM projection matrices, stored under the existing R_sp key.

    The matrix projects incident TE/TM components onto the local transverse
    plane of diffraction order m. The function name is retained for
    compatibility; these matrices are generally nonorthogonal.

        [Es_m]
        [Ep_m]

          = R_m

        [Es_inc]
        [Ep_inc]

    For real orthonormal bases with s x p = -k on both sides,
    det(R_m) = dot(k_hat_m, k_hat_inc), and the singular values are
    1 and abs(dot(k_hat_m, k_hat_inc)). These are geometric relationships,
    not diffraction efficiencies or a Kirchhoff amplitude factor.
    """

    results_out = []

    for result in results:

        result_new = result.copy()

        if not result["propagating"]:

            result_new["R_sp"] = None

            results_out.append(result_new)
            continue

        s_m = result["s_hat"]
        p_m = result["p_hat"]

        R = np.array(
            [
                [
                    np.dot(s_m, s_hat_inc),
                    np.dot(s_m, p_hat_inc),
                ],
                [
                    np.dot(p_m, s_hat_inc),
                    np.dot(p_m, p_hat_inc),
                ],
            ]
        )

        result_new["R_sp"] = R

        results_out.append(result_new)

    return results_out        

def print_order_rotation_matrices(results):
    """Print TE/TM projection matrices; retain the historical function name."""

    print("\nTE/TM Projection Matrices")
    print("-" * 80)

    for r in results:

        if not r["propagating"]:
            continue

        m = r["order"]

        print(f"\nm={m:+d}")

        print(
            np.array2string(
                r["R_sp"],
                precision=6,
                floatmode="fixed",
            )
        )        
        
def check_order_rotation_matrices(results):
    """Print and return residuals for transverse projection identities.

    Each propagating result must contain R_sp, k_hat, and k_inc from the
    diffraction pipeline. Both TE/TM bases use s x p = -k. The expected
    determinant is the signed cosine between the propagation directions;
    the expected singular values are 1 and the absolute value of that
    cosine. R.T @ R need not equal I when the transverse planes differ.

    Returns one diagnostics dictionary per propagating order, including
    determinant_error and singular_value_error (zero for valid geometry).
    """

    print("\nTE/TM Projection Matrix Checks")
    print("-" * 80)
    diagnostics = []

    for r in results:

        if not r["propagating"]:
            continue

        m = r["order"]
        R = r["R_sp"]

        k_inc = _unit_vector(r["k_inc"], "k_inc", 1e-12)
        k_m = _unit_vector(r["k_hat"], "k_hat", 1e-12)
        direction_cosine = np.clip(np.dot(k_m, k_inc), -1.0, 1.0)
        detR = np.linalg.det(R)
        determinant_error = abs(detR - direction_cosine)
        singular_values = np.linalg.svd(R, compute_uv=False)
        expected_singular_values = np.array([1.0, abs(direction_cosine)])
        singular_value_error = np.linalg.norm(
            singular_values - expected_singular_values
        )

        diagnostics.append({
            "order": m,
            "direction_cosine": direction_cosine,
            "determinant": detR,
            "determinant_error": determinant_error,
            "singular_values": singular_values,
            "singular_value_error": singular_value_error,
        })

        print(
            f"m={m:+2d}  "
            f"det(R)={detR:+.6f}  "
            f"cos(angle)={direction_cosine:+.6f}  "
            f"det_error={determinant_error:.3e}  "
            f"sv_error={singular_value_error:.3e}"
        )

    return diagnostics


def print_order_obliquity(results):
    """Print the signed cosine between incident and order directions.

    The historical function name is retained. det(R_sp) equals this cosine
    for consistent real TE/TM bases; it describes projection geometry only.
    """

    print("\nOrder Direction Geometry")
    print("-" * 80)

    for r in results:

        if not r["propagating"]:
            continue

        m = r["order"]

        detR = np.linalg.det(r["R_sp"])
        k_inc = _unit_vector(r["k_inc"], "k_inc", 1e-12)
        k_m = _unit_vector(r["k_hat"], "k_hat", 1e-12)
        direction_cosine = np.clip(np.dot(k_m, k_inc), -1.0, 1.0)

        print(
            f"m={m:+2d}  "
            f"cos(angle)={direction_cosine:+.6f}  "
            f"det(R)={detR:+.6f}"
        )
