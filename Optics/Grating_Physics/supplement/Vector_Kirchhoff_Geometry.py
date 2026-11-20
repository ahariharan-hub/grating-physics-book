# -*- coding: utf-8 -*-

"""
Created on Mon Sep  7 09:38:11 2026

@author: aharihar

=============================================================
Vector Lithography Grating Geometry Viewer
=============================================================

Purpose
-------
Define and visualize a 1D binary lithography grating using a
vector-based coordinate system.

Coordinate System
-----------------
g_hat : grating (periodic) direction
l_hat : line / groove direction
n_hat : mask normal

such that

    g_hat × l_hat = -n_hat

The quartz-absorber interface is z = 0. Quartz lies on the +z side
(shown with a reduced display thickness), while absorber lines occupy
-height <= z <= 0.

Grating Parameters
------------------
period          : grating period
duty_cycle      : clear/open fraction
absorber_height : absorber thickness

Current Scope
-------------
✓ Vector coordinate system
✓ 3D grating geometry
✓ Direction-cosine sphere
✓ Coordinate vectors (ĝ, ℓ̂, n̂)

Future Extensions
-----------------
□ Incident plane wave k̂
□ Polarization vectors
□ Vector Kirchhoff mask model
□ Diffraction order generation
□ TE/TM basis construction
□ Scattering coefficient calculation


=============================================================
"""


import numpy as np
import matplotlib.pyplot as plt

import VKG

# ==========================================================
# Grating coordinate system
# ==========================================================

g_hat = np.array([1.0, 0.0, 0.0])   # periodic direction
l_hat = np.array([0.0, 1.0, 0.0])   # line direction
n_hat = np.array([0.0, 0.0, -1.0])   # surface normal

# ==========================================================
# Grating geometry
# ==========================================================

period = 800.0          # nm
duty   = 0.5

absorber_width = (1 - duty) * period   # chrome width
line_length    = 2000.0                # arbitrary
height         = 1.0                  # nm

# ==========================================================
# Plot
# ==========================================================

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

VKG.plot_quartz_substrate(
    ax=ax,
    g_hat=g_hat,
    l_hat=l_hat,
    n_hat=n_hat,
    width=5 * period,
    line_length=line_length,
    thickness=300.0,  # Display thickness only; not the full physical substrate.
)

# draw 5 absorber lines
for m in range(-2, 3):

    center = m * period * g_hat

    x0 = center[0] - absorber_width/2
    y0 = -line_length/2
    z0 = -height

    ax.bar3d(
        x0, y0, z0,
        absorber_width,
        line_length,
        height,
        color='royalblue',
        shade=True,
        alpha=0.8
    )

# ==========================================================
# Direction-cosine sphere centered at (0, 0, 0)
# ==========================================================

sphere_radius = 2000.0   # display scale in plot units

theta = np.linspace(0, np.pi, 40)
phi = np.linspace(0, 2*np.pi, 80)

theta, phi = np.meshgrid(theta, phi)

# Unit direction cosines scaled for visualization
sx = sphere_radius * np.sin(theta) * np.cos(phi)
sy = sphere_radius * np.sin(theta) * np.sin(phi)
sz = sphere_radius * np.cos(theta)

ax.plot_wireframe(
    sx, sy, sz,
    color="gray",
    linewidth=0.4,
    alpha=0.5
)

# ==========================================================
# Coordinate vectors
# ==========================================================

Lg = sphere_radius
Ll = sphere_radius
Ln = 2*sphere_radius

ax.quiver(0,0,0, *(Lg*g_hat),
          color='red',
          linewidth=3,
          arrow_length_ratio=0.2)

ax.quiver(0,0,0, *(Ll*l_hat),
          color='green',
          linewidth=3,
          arrow_length_ratio=0.2)

ax.quiver(0,0,sphere_radius, *(Ln*n_hat),
          color='black',
          linewidth=3,
          arrow_length_ratio=0.1)

ax.text(*(1.1*Lg*g_hat), "ĝ", color='red', fontsize=14)
ax.text(*(1.1*Ll*l_hat), "ℓ̂", color='green', fontsize=14)
ax.text(*(1.1*Ln*n_hat), "n̂", color='black', fontsize=14)

# ==========================================================
# Formatting
# ==========================================================

ax.set_xlim(-2000, 2000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(-600, 600)
ax.set_box_aspect((4, 2, 1))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title("Vector-Defined 1D Binary Grating")

plt.tight_layout()
plt.show()

#%% Modular form of above

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

ax.set_xlim(-2000, 2000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(-600, 600)
ax.set_box_aspect((4, 2, 1))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Vector-Defined 1D Binary Grating")

plt.tight_layout()
plt.show()

#%% 

import numpy as np
import matplotlib.pyplot as plt

import VKG


g_hat = np.array([1.0, 0.0, 0.0])
l_hat = np.array([0.0, 1.0, 0.0])
n_hat = np.array([0.0, 0.0, -1.0])

sphere_radius = 2000.0

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

VKG.plot_grating_3d_gen(
    ax=ax,
    g_hat=g_hat,
    l_hat=l_hat,
    n_hat=n_hat,
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

ax.set_xlim(-2000, 2000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(-600, 600)
ax.set_box_aspect((4, 2, 1))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Vector-Defined 1D Binary Grating")

plt.tight_layout()
plt.show()
