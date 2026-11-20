# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:41:52 2026

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

ax.set_xlim(-sphere_radius, sphere_radius)
ax.set_ylim(-sphere_radius, sphere_radius)
ax.set_zlim(-sphere_radius, sphere_radius)
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Incident Plane Waves and Polarization Bases")

#%% 
# ============================================================
# Incident plane wave
# ============================================================

wavelength = 193.0  # nm

theta_deg = 45.0    # polar angle from optical axis (-z)
phi_deg   = 0.0    # azimuth angle

theta = np.radians(theta_deg)
phi   = np.radians(phi_deg)

# propagation direction (unit vector)
k_hat = np.array([
    np.sin(theta)*np.cos(phi),
    np.sin(theta)*np.sin(phi),
   -np.cos(theta)                 # minus sign because optical axis = -z
])

# actual wavevector
k0 = 2*np.pi / wavelength
k_inc = k0 * k_hat

print("k_hat =", k_hat)
print("|k_hat| =", np.linalg.norm(k_hat))
print("k_inc =", k_inc)

# ============================================================
# Plot incident ray
# ============================================================

# point on sphere where ray begins
tail = -sphere_radius * k_hat

# arrow points toward origin
direction = sphere_radius * k_hat

ax.quiver(
    tail[0], tail[1], tail[2],
    direction[0], direction[1], direction[2],
    color="magenta",
    linewidth=3,
    arrow_length_ratio=0.1,
)

ax.scatter(*tail, color="red", s=80)

ax.text(
    tail[0],
    tail[1],
    tail[2],
    r"$k_{inc}$",
    color="magenta",
)

# ============================================================
# TE(s)/TM(p) basis
# ============================================================

s_hat, p_hat = VKG.compute_polarization_basis(k_hat, n_hat)
print(s_hat)
print(p_hat)


tail = -sphere_radius * k_hat

L = 500

ax.quiver(
    tail[0], tail[1], tail[2],
    L*s_hat[0], L*s_hat[1], L*s_hat[2],
    color="olive",
    linewidth=3,
)

ax.quiver(
    tail[0], tail[1], tail[2],
    L*p_hat[0], L*p_hat[1], L*p_hat[2],
    color="brown",
    linewidth=3,
)

ax.text(*(tail + 1.1*L*s_hat), "TE / s", color="olive")
ax.text(*(tail + 1.1*L*p_hat), "TM / p", color="brown")


#%% Modular version of above

k_hat, s_hat, p_hat, k_inc = VKG.plot_incident_plane_wave(
    ax=ax,
    theta_deg=45.0,
    phi_deg=45.0,
    sphere_radius=sphere_radius,
    n_hat=n_hat,
)

plt.tight_layout()
plt.show()











