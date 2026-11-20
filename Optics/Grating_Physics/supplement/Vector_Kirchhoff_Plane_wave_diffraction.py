# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:24:14 2026

@author: aharihar
"""

import numpy as np
import matplotlib.pyplot as plt

import VKG


g_hat = np.array([1.0, 0.0, 0.0])
l_hat = np.array([0.0, 1.0, 0.0])
n_hat = np.array([0.0, 0.0, -1.0])

sphere_radius = 2000.0

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

# ============================================================
# Plot vector grating geometry
# ============================================================

VKG.plot_grating_3d(
    ax=ax,
    g_hat=g_hat,
    period=800.0,
    duty=0.5,
    line_length=2000.0,
    height=1.0,
    n_lines=5,
    substrate_thickness=300.0,  # Display thickness only.
)

VKG.plot_direction_cosine_sphere(
    ax=ax,
    radius=sphere_radius,
)

VKG.plot_basis_vectors(
    ax=ax,
    g_hat=g_hat,
    l_hat=l_hat,
    n_hat=n_hat,
    sphere_radius=sphere_radius,
)

# ============================================================
# Incident plane wave
# ============================================================

k_hat, s_hat, p_hat, k_inc = VKG.plot_incident_plane_wave(
    ax=ax,
    theta_deg=45.0,
    phi_deg=-45.0,
    sphere_radius=sphere_radius,
    n_hat=n_hat,
    wavelength=193.0,
)


# ============================================================
# Compute diffraction orders
# ============================================================

orders = VKG.compute_diffraction_orders(
    k_hat_inc=k_hat,
    wavelength=193.0,
    period=800.0,
    g_hat=g_hat,
    n_hat=n_hat,
    orders=range(-5, 6),
    propagation_sign=+1,
    include_evanescent=True,
)


# ============================================================
# Print diffraction orders
# ============================================================

VKG.print_diffraction_orders(orders)

# ============================================================
# Plot propagating diffraction orders
# ============================================================

VKG.plot_diffraction_orders(
    ax=ax,
    results=orders,
    sphere_radius=sphere_radius,
    color="darkorange",
    linewidth=2.0,
    arrow_length_ratio=0.06,
    marker_color="red",
    marker_size=50,
    show_arrows=True,
    show_points=True,
    show_labels=True,
)


# ============================================================
# Compute local polarization basis for every order
# ============================================================

orders = VKG.compute_order_polarization_bases(
    results=orders,
    n_hat=n_hat,
    reference_hat=l_hat,
)

orders = VKG.compute_order_rotation_matrices(
    orders,
    s_hat_inc=s_hat,
    p_hat_inc=p_hat,
)

VKG.check_order_rotation_matrices(orders)

VKG.print_order_rotation_matrices(orders)

# ============================================================
# Local TE/TM bases check
# ============================================================
VKG.check_order_bases(orders)

# ============================================================
# Print polarization bases orders
# ============================================================

VKG.print_order_polarization_bases(orders)

# ============================================================
# Plot local TE/TM bases
# ============================================================

VKG.plot_order_polarization_bases(
    ax=ax,
    results=orders,    
    sphere_radius=sphere_radius,
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
    show_basis_labels=True,
)


# ============================================================
# Field to Local TE/TM bases 
# ============================================================

orders = VKG.transform_field_to_order_bases(
    results=orders,
    s_hat_inc=s_hat,
    p_hat_inc=p_hat,
    Es_inc=1.0,
    Ep_inc=0.0,
)

# orders = VKG.transform_field_to_order_bases(
#     results=orders,
#     s_hat_inc=s_hat,
#     p_hat_inc=p_hat,
#     Es_inc=0.0,
#     Ep_inc=1.0,
# )

# orders = VKG.transform_field_to_order_bases(
#     results=orders,
#     s_hat_inc=s_hat,
#     p_hat_inc=p_hat,
#     Es_inc=1/np.sqrt(2),
#     Ep_inc=1/np.sqrt(2),
# )

# ============================================================
# Print projected field components
# ============================================================

VKG.print_order_field_projections(orders)

# ============================================================
# Axis settings
# ============================================================

ax.set_xlim(-sphere_radius, sphere_radius)
ax.set_ylim(-sphere_radius, sphere_radius)
ax.set_zlim(-sphere_radius, sphere_radius)
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title("Incident Plane Wave and Diffraction Orders")

plt.tight_layout()
plt.show()
