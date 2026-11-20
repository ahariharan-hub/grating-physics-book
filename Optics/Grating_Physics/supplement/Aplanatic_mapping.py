# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 07:56:29 2026

@author: ash-w
"""

import numpy as np
import plotly.graph_objects as go


# ============================================================
# PARAMETERS
# ============================================================

Rp = 1.0                       # Physical pupil radius
theta_max_deg = 60.0           # Maximum angular extent
theta_deg = 35.0               # Selected ray angle

theta_max = np.deg2rad(theta_max_deg)
theta = np.deg2rad(theta_deg)

# Radius of reference sphere chosen so that
# Rp = R_ref * sin(theta_max)
R_ref = Rp / np.sin(theta_max)

# Sine-condition normalized pupil coordinate
v = np.sin(theta) / np.sin(theta_max)

# Corresponding physical pupil radial coordinate
rp = Rp * v

print(f"theta_max = {theta_max_deg:.2f} deg")
print(f"theta     = {theta_deg:.2f} deg")
print(f"v         = {v:.4f}")
print(f"rp        = {rp:.4f}")
print(f"Rp        = {Rp:.4f}")


# ============================================================
# 1. REFERENCE SPHERE
# ============================================================

theta_grid = np.linspace(0, theta_max, 60)
phi_grid = np.linspace(0, 2*np.pi, 100)

THETA, PHI = np.meshgrid(theta_grid, phi_grid)

X_sphere = R_ref * np.sin(THETA) * np.cos(PHI)
Y_sphere = R_ref * np.sin(THETA) * np.sin(PHI)
Z_sphere = R_ref * np.cos(THETA)


# ============================================================
# 2. PHYSICAL PUPIL PLANE
# ============================================================

rho = np.linspace(0, Rp, 40)
phi = np.linspace(0, 2*np.pi, 100)

RHO, PHI_p = np.meshgrid(rho, phi)

X_pupil = RHO * np.cos(PHI_p)
Y_pupil = RHO * np.sin(PHI_p)

# Put pupil at z = 0
Z_pupil = np.zeros_like(X_pupil)


# ============================================================
# 3. SELECTED POINT ON REFERENCE SPHERE
# ============================================================

# Choose azimuth phi = 0 for clarity
phi_selected = 0.0

x_ref = R_ref * np.sin(theta) * np.cos(phi_selected)
y_ref = R_ref * np.sin(theta) * np.sin(phi_selected)
z_ref = R_ref * np.cos(theta)


# ============================================================
# 4. CORRESPONDING POINT IN PHYSICAL PUPIL
# ============================================================

x_p = rp
y_p = 0.0
z_p = 0.0


# ============================================================
# 5. FIGURE
# ============================================================

fig = go.Figure()


def mapping_state(theta_value):
    """Return the plotted coordinates for one selected ray angle."""
    radius = Rp * np.sin(theta_value) / np.sin(theta_max)
    sphere_x = R_ref * np.sin(theta_value)
    sphere_z = R_ref * np.cos(theta_value)
    return radius, sphere_x, sphere_z


# ------------------------------------------------------------
# Reference sphere
# ------------------------------------------------------------

fig.add_trace(
    go.Surface(
        x=X_sphere,
        y=Y_sphere,
        z=Z_sphere,
        opacity=0.35,
        showscale=False,
        name="Reference sphere"
    )
)


# ------------------------------------------------------------
# Physical pupil
# ------------------------------------------------------------

fig.add_trace(
    go.Surface(
        x=X_pupil,
        y=Y_pupil,
        z=Z_pupil,
        opacity=0.35,
        showscale=False,
        name="Physical pupil"
    )
)


# ------------------------------------------------------------
# Pupil rim
# ------------------------------------------------------------

phi_rim = np.linspace(0, 2*np.pi, 300)

fig.add_trace(
    go.Scatter3d(
        x=Rp*np.cos(phi_rim),
        y=Rp*np.sin(phi_rim),
        z=np.zeros_like(phi_rim),
        mode="lines",
        line=dict(width=5),
        name="Pupil rim"
    )
)


# ------------------------------------------------------------
# Optical axis
# ------------------------------------------------------------

z_axis = np.linspace(0, R_ref * 1.15, 100)

fig.add_trace(
    go.Scatter3d(
        x=np.zeros_like(z_axis),
        y=np.zeros_like(z_axis),
        z=z_axis,
        mode="lines",
        line=dict(width=5, dash="dash"),
        name="Optical axis"
    )
)


# ------------------------------------------------------------
# Direction vector on reference sphere
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter3d(
        x=[0, x_ref],
        y=[0, y_ref],
        z=[0, z_ref],
        mode="lines+markers",
        line=dict(width=7),
        marker=dict(size=5),
        name="Direction vector"
    )
)


# ------------------------------------------------------------
# Selected point on reference sphere
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter3d(
        x=[x_ref],
        y=[y_ref],
        z=[z_ref],
        mode="markers+text",
        marker=dict(size=8),
        text=[f"θ = {theta_deg:.1f}°"],
        textposition="top center",
        hovertemplate="Reference sphere<br>θ = %{customdata:.1f}°<extra></extra>",
        customdata=[theta_deg],
        name="Reference-sphere point"
    )
)


# ------------------------------------------------------------
# Selected pupil point
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter3d(
        x=[x_p],
        y=[y_p],
        z=[z_p],
        mode="markers+text",
        marker=dict(size=8),
        text=[f"rₚ = {rp:.3f}"],
        textposition="bottom center",
        hovertemplate="Physical pupil<br>rₚ/Rₚ = %{customdata:.3f}<extra></extra>",
        customdata=[v],
        name="Pupil point"
    )
)


# ------------------------------------------------------------
# Physical radial coordinate rp
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter3d(
        x=[0, x_p],
        y=[0, y_p],
        z=[0, 0],
        mode="lines",
        line=dict(width=8),
        name="Sine-condition pupil coordinate"
    )
)


# ------------------------------------------------------------
# Maximum pupil radius Rp
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter3d(
        x=[0, Rp],
        y=[0, 0],
        z=[0, 0],
        mode="lines+text",
        line=dict(width=4, dash="dot"),
        text=["", "Rₚ"],
        textposition="top center",
        name="Rₚ"
    )
)


# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(

    title=(
        "Physical Pupil → Reference Sphere Mapping"
        f"<br><sup>"
        f"v = rₚ/Rₚ = sin(θ)/sin(θmax) = {v:.3f}"
        f"<br>Orange: ray direction · green: mapped pupil coordinate (not a projection)"
        f"</sup>"
    ),

    scene=dict(

        xaxis=dict(
            title="x",
            range=[-1.3*Rp, 1.3*Rp]
        ),

        yaxis=dict(
            title="y",
            range=[-1.3*Rp, 1.3*Rp]
        ),

        zaxis=dict(
            title="z",
            range=[0, 1.25*R_ref]
        ),

        aspectmode="data",

        camera=dict(
            eye=dict(
                x=1.5,
                y=1.5,
                z=1.1
            )
        )
    ),

    width=1000,
    height=800,

    margin=dict(
        l=0,
        r=0,
        b=0,
        t=80
    )
)


slider_angles_deg = np.arange(0.0, theta_max_deg + 0.1, 5.0)
frames = []

for angle_deg in slider_angles_deg:
    angle = np.deg2rad(angle_deg)
    radius, sphere_x, sphere_z = mapping_state(angle)
    normalized_radius = radius / Rp

    frames.append(
        go.Frame(
            name=f"theta-{angle_deg:.0f}",
            data=[
                go.Scatter3d(
                    x=[0, sphere_x],
                    y=[0, 0],
                    z=[0, sphere_z]
                ),
                go.Scatter3d(
                    x=[sphere_x],
                    y=[0],
                    z=[sphere_z],
                    text=[f"θ = {angle_deg:.1f}°"],
                    customdata=[angle_deg]
                ),
                go.Scatter3d(
                    x=[radius],
                    y=[0],
                    z=[0],
                    text=[f"rₚ = {radius:.3f}"],
                    customdata=[normalized_radius]
                ),
                go.Scatter3d(
                    x=[0, radius],
                    y=[0, 0],
                    z=[0, 0]
                )
            ],
            traces=[4, 5, 6, 7],
            layout=go.Layout(
                title=(
                    "Physical Pupil → Reference Sphere Mapping"
                    f"<br><sup>"
                    f"v = rₚ/Rₚ = sin(θ)/sin(θmax) = {normalized_radius:.3f}"
                    f"<br>Orange: ray direction · green: mapped pupil coordinate (not a projection)"
                    f"</sup>"
                )
            )
        )
    )

fig.frames = frames
fig.update_layout(
    sliders=[
        dict(
            active=int(round(theta_deg / 5.0)),
            currentvalue=dict(prefix="Selected angle: "),
            pad=dict(t=35),
            steps=[
                dict(
                    label=f"{angle_deg:.0f}°",
                    method="animate",
                    args=[
                        [f"theta-{angle_deg:.0f}"],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0)
                        )
                    ]
                )
                for angle_deg in slider_angles_deg
            ]
        )
    ]
)


fig.show(renderer="browser")