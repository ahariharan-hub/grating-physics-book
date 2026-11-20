"""Linked closed-surface/ aperture lesson. Coordinates are (x, y, s), s=-z.

All numerical surface derivatives use the full 3D gradient dotted with the
specified normal. The planar helper in propagation_interactive is not used.
"""
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import ipywidgets as w
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib import rc_context

from .propagation_pictures import BLUE, ORANGE, GREEN, INK, GREY, arrow
from .propagation_interactive import PropagationPanel

K = 2*np.pi
SOURCE = np.array([0., -.4, -3.])
CENTER = np.array([0., 0., 1.5])
STAGES = ('A: Helmholtz', 'B: Green response', 'C: Green identity',
          'D: Small sphere', 'E: Closing surface', 'F: Aperture guess')


def _point(p, name):
    p = np.asarray(p, dtype=float)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError(f'{name} must contain three finite real coordinates.')
    return p


def response(q, origin, k=K):
    """Outgoing g and its full gradient with respect to q, including 1/(4pi)."""
    q = np.asarray(q, dtype=float)
    origin = _point(origin, 'origin')
    if q.shape[-1:] != (3,) or not np.all(np.isfinite(q)):
        raise ValueError('q must have final dimension 3 and finite coordinates.')
    if not np.isfinite(k) or k <= 0:
        raise ValueError('k must be positive and finite.')
    d = q-origin
    r = np.linalg.norm(d, axis=-1)
    if np.any(r == 0):
        raise ValueError('The point-source response is singular at its origin.')
    g = np.exp(1j*k*r)/(4*np.pi*r)
    grad = (g*(1j*k-1/r)/r)[..., None]*d
    return g, grad


@lru_cache(maxsize=8)
def _directions(order):
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 8:
        raise ValueError('order must be an integer of at least 8.')
    mu, weights = np.polynomial.legendre.leggauss(order)
    phi = np.arange(2*order)*np.pi/order
    ring = np.sqrt(1-mu**2)[:, None]
    dirs = np.stack(np.broadcast_arrays(ring*np.cos(phi), ring*np.sin(phi), mu[:, None]), axis=-1)
    solid = np.broadcast_to(weights[:, None]*np.pi/order, dirs.shape[:-1]).copy()
    dirs.setflags(write=False); solid.setflags(write=False)
    return dirs, solid


def sphere_terms(p, radius=2.1, center=CENTER, source=SOURCE, order=48, inner=False):
    """Integrate U dnG and -G dnU on a sphere; U=4pi*g_source.

    Outer normal points inward; an exclusion sphere's normal points radially
    outward into the punctured region. Its centre must be p.
    """
    p = _point(p, 'p'); center = _point(center, 'center'); source = _point(source, 'source')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be positive and finite.')
    if inner:
        if not np.array_equal(center, p) or radius >= np.linalg.norm(source-p):
            raise ValueError('An exclusion sphere must be centred at p and exclude the source.')
    elif np.linalg.norm(p-center) >= radius or np.linalg.norm(source-center) <= radius:
        raise ValueError('p must lie inside and the physical source outside the sphere.')
    directions, solid = _directions(order)
    q = center+radius*directions
    normal = directions if inner else -directions
    g, dg = response(q, p)
    u, du = response(q, source); u = 4*np.pi*u; du = 4*np.pi*du
    a = u*np.sum(dg*normal, axis=-1)*radius**2*solid
    b = -g*np.sum(du*normal, axis=-1)*radius**2*solid
    reference = 4*np.pi*response(p, source)[0]
    # Bins are latitude rings, each integrated over the complete azimuth.
    return dict(value=a.sum(), slope=b.sum(), total=a.sum()+b.sum(),
                rings=(a+b).sum(axis=1), reference=reference)


def closure_terms(p, radius=6., order=96):
    """Actual unobstructed reference field on the hemispherical closure C.

    The hemisphere is s>=0; the disk completing its boundary has normal +s.
    No physical screen is present in this numerical reference check.
    """
    p = _point(p, 'p')
    if not np.isfinite(radius) or radius <= np.linalg.norm(p) or p[2] <= 0:
        raise ValueError('The observation point must be inside the positive-s hemisphere.')
    _directions(order)  # validates the order
    mu, weight = np.polynomial.legendre.leggauss(order)
    mu = (mu+1)/2; weight = weight/2
    phi = np.arange(2*order)*np.pi/order
    ring = np.sqrt(1-mu**2)[:, None]
    dirs = np.stack(np.broadcast_arrays(ring*np.cos(phi),ring*np.sin(phi),mu[:, None]),axis=-1)
    q = radius*dirs; normal = -dirs
    u, du = response(q,SOURCE); u=4*np.pi*u;du=4*np.pi*du
    g, dg = response(q,p)
    da = radius**2*weight[:,None]*np.pi/order
    a = np.sum(u*np.sum(dg*normal,axis=-1)*da)
    b = np.sum(-g*np.sum(du*normal,axis=-1)*da)
    return dict(value=a,slope=b,total=a+b)


def _geometry(ax, p, radius, eta, exclusion=False, cells=False, probe=False):
    """Common source/point geometry, projected onto the (s,y) plane."""
    cy, cs = CENTER[1:]; t=eta
    q=CENTER+radius*np.array([0.,np.sin(t),np.cos(t)])
    ax.add_patch(Circle((cs,cy),radius,fc='#f1f6fa',ec=INK,lw=1.8))
    if cells:
        for d in np.arange(-2.4,2.5,.5):
            span=np.sqrt(max(0,radius**2-d**2))
            if span:
                ax.plot([cs+d,cs+d],[-span,span],color=BLUE,alpha=.2,lw=.8)
                ax.plot([cs-span,cs+span],[d,d],color=BLUE,alpha=.2,lw=.8)
        # Two neighbouring cells, same face, opposite normals.
        ax.add_patch(Rectangle((cs-.5,-.5),.5,.5,fill=False,ec=BLUE,lw=1.2))
        ax.add_patch(Rectangle((cs,-.5),.5,.5,fill=False,ec=BLUE,lw=1.2))
        arrow(ax,(cs-.2,-.23),(cs+.16,-.23),color=BLUE)
        arrow(ax,(cs+.2,-.33),(cs-.16,-.33),color=ORANGE)
    ax.plot(SOURCE[2],SOURCE[1],'*',ms=12,color=BLUE)
    ax.annotate('Physical source $P_0$',(SOURCE[2],SOURCE[1]),xytext=(2,-22),textcoords='offset points',fontsize=9)
    ax.plot(p[2],p[1],'o',color=ORANGE)
    p_label='Observation $P$\n+ Green-probe centre' if probe else 'Observation $P$'
    ax.annotate(p_label,(p[2],p[1]),xytext=(7,-20),textcoords='offset points',fontsize=9)
    ax.plot(q[2],q[1],'o',color=GREEN)
    ax.annotate('$Q$ on $S$',(q[2],q[1]),xytext=(7,7),textcoords='offset points',fontsize=10)
    arrow(ax,(q[2],q[1]),(q[2]-.35*np.cos(t),q[1]-.35*np.sin(t)),color=GREEN)
    ax.plot([q[2],p[2]],[q[1],p[1]],':',color=ORANGE,lw=1.5)
    ax.text(cs,radius+.16,'Mathematical boundary S',ha='center',fontsize=10)
    if exclusion:
        ax.add_patch(Circle((p[2],p[1]),.25,fc='white',ec=ORANGE,ls='--'))
        ax.text(p[2]+.3,p[1]+.3,"S' (enlarged)",fontsize=9,color=ORANGE)
    ax.set(xlim=(-3.8,5),ylim=(-3.25,3.25),aspect='equal',xlabel=r'$s=-z$ / wavelength',ylabel=r'$y$ / wavelength')
    ax.spines[['top','right']].set_visible(False)
    return q


def boundary_sample(q, p, normal):
    """Two complex Kirchhoff densities for the known incident/reference field."""
    g, dg = response(q, p)
    ui, dui = response(q, SOURCE)
    ui, hi = 4*np.pi*ui, 4*np.pi*np.dot(dui,normal)
    return dict(g=g, derivative=np.dot(dg,normal), incident_value=ui,
                incident_slope=hi, value_term=ui*np.dot(dg,normal), slope_term=-g*hi)


@lru_cache(maxsize=16)
def _inner_curve(y):
    p=np.array([0.,y,1.3]);eps=np.geomspace(.04,.6,48)
    terms=[sphere_terms(p,e,center=p,inner=True,order=24) for e in eps]
    a=np.array([t['value']/t['reference'] for t in terms])
    b=np.array([t['slope']/t['reference'] for t in terms])
    return eps,a,b


@lru_cache(maxsize=16)
def _closure_curve(y):
    p=np.array([0.,y,1.3]);rr=np.linspace(4,14,28)
    ref=4*np.pi*response(p,SOURCE)[0]
    return rr,np.array([abs(closure_terms(p,r,order=80)['total']/ref) for r in rr])


def stage_figure(stage=0, y=.3, radius=2.1, angle=.9, epsilon=.2, closure=6.,
                 wavelength=1., accounting=0, morph=1., width=1.7,
                 boundary_eta=.6, reveal=False, distance=1.3):
    """One manipulated concept. Every numeric annotation comes from ``data``.

    The interpolated contour and cells are geometry/accounting schematics;
    sphere/closure integrals and incident/Green densities are computed in 3D.
    """
    if stage not in range(6):raise ValueError('stage must be 0 through 5.')
    if not (-.6<=y<=.6 and 1.5<=radius<=2.8 and 0<=angle<=2*np.pi
            and .04<=epsilon<=.6 and 4<=closure<=14 and .7<=wavelength<=1.5
            and accounting in range(4) and 0<=morph<=1 and .6<=width<=4
            and -3<=boundary_eta<=3 and .6<=distance<=3.):
        raise ValueError('Controls are outside the supported lesson range.')
    p=np.array([0.,y,distance if stage in (1,5) else 1.3]);data={}
    fig=Figure(figsize=(11.5,4.9),layout='constrained',facecolor='white')
    left,right=fig.subplots(1,2)
    if stage<4:q=_geometry(left,p,radius,angle,exclusion=stage==2,cells=False,probe=stage==1)
    if stage==0:
        k=2*np.pi/wavelength;r=np.linspace(1.8,7,601);u=np.exp(1j*k*r)/r
        ur=u*(1j*k-1/r);urr=u*(-k*k-2j*k/r+2/r**2)
        lap=urr+2*ur/r;selected=np.linalg.norm(p-SOURCE)
        up=np.exp(1j*k*selected)/selected
        lap_p=up*(-k*k-2j*k/selected+2/selected**2)+2*up*(1j*k-1/selected)/selected
        # Equal-phase fronts make the wavelength control visible in the geometry.
        for front in np.arange(wavelength,8.1,wavelength):
            left.add_patch(Circle((SOURCE[2],SOURCE[1]),front,fill=False,ec=BLUE,alpha=.15,lw=1))
        right.plot(r,u.real,color=BLUE,lw=2.2,label='Real field')
        right.plot(r,u.imag,'--',color=ORANGE,lw=1.4,alpha=.8,label='Imaginary field')
        right.plot(r,1/r,':',color=GREY,lw=.9)
        right.plot(r,-1/r,':',color=GREY,lw=.9)
        right.axvline(selected,color=GREEN,ls=':',lw=1)
        right.scatter([selected,selected],[up.real,up.imag],color=GREEN,s=32,zorder=3)
        right.set(xlim=(1.8,7),ylim=(-.6,.6),xlabel=r'Distance from source $r_0/\lambda_{ref}$',ylabel='Field amplitude',
                  title=f'Shorter wavelength: more oscillations\n'+r'$\lambda/\lambda_{ref}$'+f' = {wavelength:.2f}')
        right.legend(fontsize=9,loc='upper right',ncol=2)
        left.set_title(r'Physical illumination $U$ from $P_0$')
        left.set_xlabel(r'$s=-z$ / reference wavelength');left.set_ylabel(r'$y$ / reference wavelength')
        # Keep curvature as a compact second look, not the main visual.
        balance=left.inset_axes([.52,.04,.43,.24])
        rr=np.linspace(max(1.8,selected-.5),min(7,selected+.5),91)
        uu=np.exp(1j*k*rr)/rr
        balance.plot(rr,uu.real,color=BLUE,lw=1.1)
        balance.plot(rr,-uu.real,'--',color=ORANGE,lw=1.1)
        balance.axvline(selected,color=GREEN,ls=':',lw=.8)
        balance.set_title(r'Local balance: $U$ and $\nabla^2U/k^2$',fontsize=8)
        balance.tick_params(labelsize=7);balance.set_ylim(-.3,.3)
        data.update(field=up,laplacian=lap_p,k=k,selected_radius=selected,
                    laplacian_residual=float(np.max(abs(lap+k*k*u))),
                    profile_radius=r,profile_field=u,profile_laplacian=lap)
    elif stage==1:
        g,_=response(q,p);r=np.linalg.norm(q-p)
        # Same visual vocabulary as the earlier point-response explorer: phase
        # fronts plus one complex arrow. These fronts belong to the Green probe.
        for front in np.arange(.5,5,.5):
            left.add_patch(Circle((p[2],p[1]),front,fill=False,ec=GREEN,alpha=.16,lw=.8))
        left.text(.02,.98,'Auxiliary response G, sampled at Q\nP remains the observation point',transform=left.transAxes,va='top',fontsize=8,color=GREEN)
        left.annotate(r'$R_{PQ}$',((q[2]+p[2])/2,(q[1]+p[1])/2),xytext=(5,8),textcoords='offset points',color=ORANGE,fontsize=11)
        arrow(right,(0,0),(g.real,g.imag),color=GREEN,lw=2.5)
        limit=.20
        right.add_patch(Circle((0,0),abs(g),fill=False,ec=GREEN,alpha=.3))
        right.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',xlabel='Real response',ylabel='Imaginary response',
                  title=f'Separation = {r:.2f}; response length = {abs(g):.4f}\nPhase = {np.angle(g):+.2f} rad')
        right.axhline(0,color=GREY,lw=.7);right.axvline(0,color=GREY,lw=.7)
        right.text(.02,.02,'Fixed amplitude scale',transform=right.transAxes,fontsize=9,color=GREY)
        left.set_title(r'Mathematical Green probe $G$ centred at $P$')
        data.update(response=g,separation=r,phasor_limit=limit,q=q,p=p.copy(),distance=distance)
    elif stage==2:
        qface=np.array([0.,-.25,1.5]);sample=boundary_sample(qface,p,np.array([0.,0.,-1.]))
        face=(sample['value_term']+sample['slope_term'])*.25
        opposite=-face
        # Each small cube has an inward normal at the same shared face.
        if accounting<3:
            for j,x in enumerate([1.,1.5]):
                active=(j==0 and accounting==0) or accounting in (1,2)
                left.add_patch(Rectangle((x,-.5),.5,.5,fc=[BLUE,ORANGE][j],alpha=.25 if active else .06,ec=INK))
            left.plot(1.5,-.25,'.',color=INK)
            left.annotate('$Q_f$',(1.5,-.25),xytext=(5,-18),textcoords='offset points',fontsize=9)
        if accounting<3:
            for d in np.arange(-1.5,1.6,.5):
                span=np.sqrt(max(0,radius**2-d*d))
                left.plot([1.5+d,1.5+d],[-span,span],color=GREY,alpha=.15)
        if accounting==0:arrow(left,(1.5,-.25),(1.05,-.25),color=BLUE,lw=2)
        elif accounting in (1,2):
            arrow(left,(1.5,-.18),(1.05,-.18),color=BLUE,lw=2)
            arrow(left,(1.5,-.32),(1.95,-.32),color=ORANGE,lw=2)
            if accounting==2:left.plot(1.5,-.25,'x',color=INK,ms=14,mew=2)
        if accounting==3:
            left.add_patch(Circle((CENTER[2],CENTER[1]),radius,fill=False,ec=GREEN,lw=3))
            left.add_patch(Circle((p[2],p[1]),.25,fill=False,ec=ORANGE,lw=3))
        right.axis('off')
        titles=['1. Account for the left cell','2. Add its neighbour','3. Cancel the shared face','4. Keep only S and S\u2032']
        right.text(.02,.96,titles[accounting],va='top',fontsize=14,color=INK)
        if accounting<3:
            for j,x in enumerate([.12,.48]):
                right.add_patch(Rectangle((x,.57),.36,.24,fc=[BLUE,ORANGE][j],alpha=.23 if (j==0 or accounting>0) else .03,ec=INK))
            arrow(right,(.48,.74),(.28,.74),color=BLUE,lw=2)
            if accounting>0:arrow(right,(.48,.64),(.68,.64),color=ORANGE,lw=2)
            if accounting==2:right.text(.48,.66,'X',ha='center',fontsize=22)
            right.text(.48,.49,'Same sampled face at Qf; opposite inward normals',ha='center',fontsize=9)
        else:
            right.add_patch(Circle((.48,.69),.14,fill=False,ec=GREEN,lw=2))
            right.add_patch(Circle((.49,.68),.025,fill=False,ec=ORANGE,lw=2))
            right.text(.65,.73,'S',color=GREEN,fontsize=12)
            right.text(.54,.64,"S'",color=ORANGE,fontsize=11)
            right.text(.48,.49,'Only exterior and exclusion boundaries remain',ha='center',fontsize=9)
        inset=right.inset_axes([.12,.06,.78,.32]);lim=max(.01,abs(face)*1.4)
        if accounting<3:
            arrow(inset,(0,0),(face.real,face.imag),color=BLUE,lw=2)
            if accounting>0:arrow(inset,(face.real,face.imag),(0,0),color=ORANGE,lw=2)
        inset.plot(0,0,'o',color=GREEN,ms=4)
        inset.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='Real face term',ylabel='Imaginary',aspect='equal')
        inset.tick_params(labelsize=8);inset.axhline(0,color=GREY,lw=.5);inset.axvline(0,color=GREY,lw=.5)
        left.set_title('Sampled cell-face terms; not interference')
        data.update(face_term=face,opposite_term=opposite,accounting=accounting,
                    remaining_boundaries=accounting==3,accounted=face if accounting==0 else 0j,
                    displayed_terms=(face,) if accounting==0 else ((face,opposite) if accounting==1 else (0j,)))
    elif stage==3:
        eps,a,b=_inner_curve(y)
        right.plot(eps,a.real,color=BLUE,label='Real first term / U(P)')
        right.plot(eps,b.real,color=ORANGE,label='Real second term / U(P)')
        right.plot(eps,(a+b).real,'--',color=GREEN,label='Real total / U(P) = -1')
        right.plot(eps,a.imag,':',color=BLUE,label='Imaginary first term / U(P)')
        right.plot(eps,b.imag,':',color=ORANGE,label='Imaginary second term / U(P)')
        t=sphere_terms(p,epsilon,center=p,inner=True)
        right.axvline(epsilon,color=GREY,ls=':')
        right.scatter([epsilon,epsilon],[(t['value']/t['reference']).real,(t['slope']/t['reference']).real],color=[BLUE,ORANGE],zorder=3)
        right.set(xlim=(.04,.6),ylim=(-1.3,.9),xlabel=r'$\epsilon$ / wavelength',ylabel='Term / U(P)',title='Selected contributions change; total stays -U(P)')
        right.legend(fontsize=7.5,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
        left.add_patch(Circle((p[2],p[1]),epsilon,fill=False,ec=ORANGE,lw=2))
        arrow(left,(p[2]+epsilon,p[1]),(p[2]+epsilon+.35,p[1]),color=ORANGE)
        # A true-scale local view makes even the smallest sphere visible.
        zoom=left.inset_axes([.03,.64,.28,.30]);zoom.add_patch(Circle((0,0),epsilon,fill=False,ec=ORANGE,lw=2))
        zoom.plot(0,0,'o',color=ORANGE,ms=3);arrow(zoom,(epsilon,0),(epsilon+.15,0),color=ORANGE)
        zoom.set(xlim=(-.8,.8),ylim=(-.8,.8),aspect='equal');zoom.set_title("S' fixed-scale zoom",fontsize=8);zoom.tick_params(labelsize=7)
        left.set_title("Actual S' radius and normal into punctured volume")
        data.update(inner=t,inner_max_error=float(np.max(abs(a+b+1))),area=4*np.pi*epsilon**2,
                    normal_weight=np.exp(1j*K*epsilon)*(1j*K*epsilon-1)/(4*np.pi*epsilon**2),epsilon=epsilon)
    elif stage==4:
        theta=np.linspace(-np.pi,np.pi,401)
        original=np.column_stack([CENTER[2]+radius*np.cos(theta),radius*np.sin(theta)])
        target=np.column_stack([closure*np.maximum(np.cos(theta),0),closure*np.sin(theta)])
        contour=(1-morph)*original+morph*target
        # The near geometry is the main picture. A fixed-scale overview retains
        # the expanding closure without making the reader's aperture microscopic.
        left.plot(contour[:,0],contour[:,1],color=GREEN,lw=2.3)
        left.set(xlim=(-4,5),ylim=(-3.5,3.5),aspect='equal',xlabel=r'$s=-z$ / wavelength',ylabel=r'$y$ / wavelength')
        left.plot(SOURCE[2],SOURCE[1],'*',color=BLUE,ms=11)
        left.annotate('Physical source P0',SOURCE[[2,1]],xytext=(-3,-23),textcoords='offset points',fontsize=9)
        left.plot(p[2],p[1],'o',color=ORANGE,ms=6)
        left.annotate('P stays here',p[[2,1]],xytext=(8,6),textcoords='offset points',fontsize=10)
        left.set_title(f'The same boundary becomes A + B + C ({morph:.0%})')
        overview=right.inset_axes([.64,.52,.33,.40])
        overview.plot(contour[:,0],contour[:,1],color=GREEN,lw=1.2)
        overview.plot(p[2],p[1],'o',color=ORANGE,ms=3)
        overview.set(xlim=(-4,15),ylim=(-15,15),aspect='equal')
        overview.tick_params(labelsize=6);overview.set_title('Fixed-scale overview',fontsize=7.5)
        if morph==1:
            left.plot([0,0],[-.85,.85],color=GREEN,lw=4)
            left.text(-.65,.6,'A',fontsize=12,color=GREEN)
            left.text(-.65,2.6,'B',fontsize=12,color=INK)
            arrow(left,(0,.6),(.55,.6),color=GREEN)
            left.text(3.9,2.8,'C continues outward',ha='right',fontsize=9,color=GREEN)
            rr,vals=_closure_curve(y);t=closure_terms(p,closure)
            ref=4*np.pi*response(p,SOURCE)[0]
            right.plot(rr,vals,color=GREEN,lw=2);right.plot(closure,abs(t['total']/ref),'o',color=ORANGE)
            right.axvline(closure,color=GREY,ls=':')
            right.set(xlim=(4,14),ylim=(0,.25),xlabel=r'Closing radius $R_C$ / wavelength',ylabel='Combined closure contribution / |U(P)|',title='Expand C; keep the detector fixed')
            data['closure']=t
        else:
            right.axis('off')
            right.text(.04,.88,'First reshape the\nmathematical surface.',fontsize=13,color=INK)
            right.text(.04,.69,'The field has left clues\non the boundary.',fontsize=12,linespacing=1.5)
            right.text(.04,.39,'At Transformation = 1:\nA = opening; B = rest of plane;\nC = the distant closing surface.',fontsize=11,linespacing=1.6)
            right.text(.04,.08,'Intermediate shapes are schematic.\nNo detector motion or far-field approximation.',fontsize=9,color=GREY)
        data.update(contour=contour,morph=morph,p=p.copy(),closure_radius=closure)
    else:
        q=np.array([0.,boundary_eta,0.]);sample=boundary_sample(q,p,np.array([0.,0.,1.]))
        region='A' if abs(boundary_eta)<width/2 else 'B'
        for lo,hi in [(-3.5,-width/2),(width/2,3.5)]:left.plot([0,0],[lo,hi],color=INK,lw=6,solid_capstyle='butt')
        left.plot([0,0],[-width/2,width/2],color=GREEN,lw=2)
        left.plot(SOURCE[2],SOURCE[1],'*',color=BLUE,ms=11);left.text(-3.5,-.95,'Physical P0',fontsize=9)
        left.plot(p[2],p[1],'o',color=ORANGE);left.text(p[2]+.2,p[1],r'$P$',fontsize=11)
        left.plot(q[2],q[1],'o',color=GREEN,ms=7);left.annotate(f'Q in {region}',(0,boundary_eta),xytext=(10,8),textcoords='offset points',fontsize=10)
        arrow(left,(0,boundary_eta),(.55,boundary_eta),color=GREEN)
        left.plot([0,p[2]],[boundary_eta,p[1]],':',color=ORANGE)
        left.text(-.65,0,'A',color=GREEN,fontsize=12);left.text(-.65,2.8,'B',fontsize=12)
        left.set(xlim=(-4,4),ylim=(-3.5,3.5),aspect='equal',xlabel=r'$s=-z$ / wavelength',ylabel=r'$\eta,y$ / wavelength',title=f'Opening = {width:.2f}; selected Q lies on {region}')
        # Restore the earlier boundary-weight interaction, with only the
        # free-space Kirchhoff response (no RS image construction).
        eta=np.linspace(-3.5,3.5,401)
        qq=np.column_stack([np.zeros_like(eta),eta,np.zeros_like(eta)])
        gg,dgg=response(qq,p)
        gmag=abs(gg);dmag=abs(dgg[:,2])/K
        right.axvspan(-width/2,width/2,color=GREEN,alpha=.07)
        right.plot(eta,gmag,color=BLUE,lw=2,label=r'$|G|$')
        right.plot(eta,dmag,'--',color=ORANGE,lw=2,label=r'$|\partial_nG|/k$')
        right.axvline(boundary_eta,color=GREEN,ls=':',lw=1)
        right.scatter([boundary_eta,boundary_eta],[abs(sample['g']),abs(sample['derivative'])/K],color=[BLUE,ORANGE],s=26,zorder=3)
        right.set(xlim=(-3.5,3.5),ylim=(0,.15),xlabel=r'Boundary coordinate $\eta$ / wavelength',ylabel='Response weight magnitude',title='Two weights carry boundary data toward P')
        right.legend(loc='upper left',fontsize=9)
        # The arithmetic is a small, explicitly secondary view. It appears only
        # after the boundary prescription is revealed.
        if reveal:
            u,h=(sample['incident_value'],sample['incident_slope']) if region=='A' else (0j,0j)
            a=u*sample['derivative'];b=-sample['g']*h
            inset=left.inset_axes([.53,.03,.43,.28])
            inset.axhline(0,color=GREY,lw=.5);inset.axvline(0,color=GREY,lw=.5)
            arrow(inset,(0,0),(a.real,a.imag),color=BLUE,lw=1.4)
            arrow(inset,(a.real,a.imag),((a+b).real,(a+b).imag),color=ORANGE,lw=1.4)
            inset.plot((a+b).real,(a+b).imag,'o',color=GREEN,ms=3)
            inset.set(xlim=(-.65,.65),ylim=(-.65,.65),aspect='equal')
            inset.set_title('Selected patch: two complex terms',fontsize=7.5);inset.tick_params(labelsize=6)
            verdict='A retains both incident data' if region=='A' else 'B sets BOTH data to zero'
            right.text(.98,.97,verdict,transform=right.transAxes,ha='right',va='top',fontsize=9,color=GREEN if region=='A' else INK)
        else:
            u=h=a=b=None
            right.text(.98,.97,'Weights known; transmitted U and h unknown',transform=right.transAxes,ha='right',va='top',fontsize=8,color=GREY)
        data.update(sample,region=region,q=q,width=width,prescribed_value=u,prescribed_slope=h,value_term=a,slope_term=b,reveal=reveal,
                    p=p.copy(),distance=distance,weight_eta=eta,green_weight=gmag,normal_weight_profile=dmag)
    for ax in (left,right):ax.tick_params(labelsize=9)
    fig.suptitle(STAGES[stage],fontsize=16,color=INK)
    return fig,data


def reconstruction_figure(y=.3,radius=2.1):
    p=np.array([0.,y,1.3]); t=sphere_terms(p,radius)
    fig=Figure(figsize=(11.5,4.5),layout='constrained',facecolor='white');a,b=fig.subplots(1,2)
    _geometry(a,p,radius,.9);a.set_title('Change the mathematical surface; keep P0 outside')
    # Complex ring contributions show genuine 3D quadrature, not invented arrows.
    z=np.r_[0,np.cumsum(t['rings'])]
    b.plot(z.real,z.imag,'-',color=BLUE,lw=1.5,label='Cumulative latitude-ring contributions')
    b.plot(t['total'].real,t['total'].imag,'o',mfc='none',mec=GREEN,ms=11,mew=2,label='Boundary reconstruction')
    b.plot(t['reference'].real,t['reference'].imag,'x',color=ORANGE,ms=8,mew=2,label='Independent analytic U(P)')
    b.set(xlabel='Real field',ylabel='Imaginary field',aspect='equal',title='Same complex field; different surface contributions')
    b.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.18));b.axhline(0,color=GREY,lw=.6);b.axvline(0,color=GREY,lw=.6)
    return fig,t


def _png(fig):
    stream=BytesIO()
    try:fig.savefig(stream,format='png',dpi=115,bbox_inches='tight',pad_inches=.06)
    finally:fig.clear()
    return stream.getvalue()


def _complex_html(z):
    """A short number, not TeX: readable in VS Code without a math renderer."""
    sign='+' if z.imag>=0 else '\u2212'
    return f'{z.real:.4f} {sign} {abs(z.imag):.4f}i'


def _equation(stage, data):
    """Plain-HTML readouts; fixed symbolic equations live in notebook Markdown.

    HTMLMath dollar delimiters failed in the intended VS Code frontend. An
    explicit \\( ... \\) probe could not be inspected after Computer Use was
    stopped, so these readouts deliberately require no widget math renderer.
    Matplotlib's math-text strings elsewhere are unchanged.
    """
    def line(label,value):
        return f'<div style="margin:2px 0;overflow-wrap:anywhere"><b>{label}</b> {value}</div>'
    if stage==0:
        residual=data['laplacian']+data['k']**2*data['field']
        return (line('Wavelength:',f'{2*np.pi/data["k"]:.2f} reference units; k = {data["k"]:.3f}')+
                line('Field at P:',_complex_html(data['field']))+
                '<details><summary>Selected-point Helmholtz balance</summary>'+
                line('Full 3D curvature \u2207\u00b2U:',_complex_html(data['laplacian']))+
                line('Balance |\u2207\u00b2U + k\u00b2U|:',f'{abs(residual):.2e}')+'</details>')
    if stage==1:
        return (line('Separation R:',f'{data["separation"]:.3f} reference units')+
                line('Response:',f'|G| = {abs(data["response"]):.4f}; phase = {np.angle(data["response"]):+.3f} rad'))
    if stage==2:
        f=data['face_term']
        terms=[line('Left-face term:',_complex_html(f)),
               line('Left face:',_complex_html(f))+line('Opposite normal, right face:',_complex_html(-f)),
               line('Shared-face total:','0; the opposite contributions cancel.'),
               line('What remains:',"Exterior boundary S and exclusion boundary S\u2032.")]
        return terms[data['accounting']]
    if stage==3:
        t=data['inner'];ref=t['reference']
        return (line('Radius \u03b5 / area:',f'{data["epsilon"]:.3f} / {data["area"]:.4f}')+
                line('Normal derivative \u2202nG:',_complex_html(data['normal_weight']))+
                line('First integral / U(P):',_complex_html(t['value']/ref))+
                line('Second integral / U(P):',_complex_html(t['slope']/ref))+
                line('Complete inner integral / U(P):',_complex_html(t['total']/ref)))
    if stage==4:
        if data['morph']==1:
            return (line('Boundary:',f'A + B + C; closing radius R_C = {data["closure_radius"]:.1f}')+
                    line('Detector:', 's = 1.3 throughout; only the mathematical closure moves.')+
                    '<details><summary>Combined closing contribution</summary>'+
                    line('Integral on C:',_complex_html(data['closure']['total']))+'</details>')
        return line('Surface transformation:',f'{data["morph"]:.0%}; P and P\u2080 stay fixed.')
    incident=(line('Incident value U\u1d62:',_complex_html(data['incident_value']))+
              line('Incident normal slope h\u1d62:',_complex_html(data['incident_slope'])))
    if not data['reveal']:
        return line('Selected point:',f'Q is in {data["region"]}; transmitted field and slope are unknown.')
    action='retain both incident values' if data['region']=='A' else 'set BOTH values to zero'
    return (line('Kirchhoff assumes:',f'Q in {data["region"]}: {action}.')+
            line('Prescribed U / h:',_complex_html(data['prescribed_value'])+' / '+_complex_html(data['prescribed_slope']))+
            '<details><summary>Incident data and the two boundary terms</summary>'+incident+
            line('U \u00d7 \u2202nG:',_complex_html(data['value_term']))+
            line('\u2212G \u00d7 h:',_complex_html(data['slope_term']))+'</details>')


CONCEPT_NOTES = (
    'Blue fronts show physical illumination U from P0. Shorten Wavelength: the fronts move closer together and more oscillations fit in the same space.',
    'Green fronts show a mathematical probe centred at P. The physical source remains P0. Move Distance or Q angle: separation changes the response length and phase. The amplitude scale stays fixed.',
    'Advance Accounting step. The face is sampled at the same Q with opposite inward normals. The complex arrows add to zero. Grid cells and midpoint face areas illustrate bookkeeping; they are not a 2D diffraction calculation or interfering light waves.',
    'Change epsilon. The actual exclusion sphere and its fixed-scale zoom change size. Area, exact radial derivative, and both complex integrals update. Their complete sum already equals -U(P) at finite radius; only the separate limits evolve.',
    'Move Transformation to see the same boundary become the opening A, screen region B, and distant closure C. Then expand R_C: P stays fixed. Intermediate shapes are schematic; the closure curve is calculated at the hemispherical endpoint.',
    'Move Distance to reshape the two boundary weights. Change Opening or Boundary eta to select A or B, then Reveal assumptions: retain the incident field and slope in A; set both to zero in B. This prescription is an approximation.',
)


def _slider(value,low,high,step,description):
    return w.FloatSlider(value=value,min=low,max=high,step=step,description=description,
                         continuous_update=False,style={'description_width':'initial'},
                         layout=w.Layout(width='290px',max_width='100%'))


def _owned_panel(controls, update_content):
    # Markdown owns symbolic equations. Plain HTML is robust to widget renderer
    # differences, including the confirmed literal-TeX VS Code failure.
    equation=w.HTML(layout=w.Layout(width='100%',max_width='100%'))
    note=w.HTML();image=w.Image(format='png',layout=w.Layout(width='100%',max_width='100%'))
    row=w.Box(list(controls.values()),layout=w.Layout(display='flex',flex_flow='row wrap',width='100%'))
    panel=PropagationPanel([row,note,image,equation],layout=w.Layout(width='100%',max_width='100%'));panel.controls=controls
    panel.equation=equation;panel.image=image;panel.note=note;panel.data={};panel._profile_cache=None
    def update(*_):update_content(panel,{key:c.value for key,c in controls.items()})
    panel._update_callback=update;panel._refresh=update
    try:
        for c in controls.values():c.observe(update,names='value')
        update()
    except Exception:
        panel.close();raise
    return panel


def concept_explorer(stage=0):
    """A fixed concept with only its own meaningful controls; no distant stage menu."""
    if stage not in range(6):raise ValueError('stage must be 0 through 5.')
    choices=[
        lambda: dict(y=_slider(.3,-.6,.6,.1,'P transverse'),wavelength=_slider(1,.7,1.5,.05,'Wavelength')),
        lambda: dict(distance=_slider(1.3,.6,3.,.1,'Distance'),y=_slider(.3,-.6,.6,.1,'P transverse'),angle=_slider(.9,0,6.2,.1,'Q angle')),
        lambda: dict(accounting=w.SelectionSlider(options=[('1. Left cell',0),('2. Add neighbour',1),('3. Cancel face',2),('4. Boundaries remain',3)],value=0,description='Accounting step',continuous_update=False,style={'description_width':'initial'},layout=w.Layout(width='480px',max_width='100%'))),
        lambda: dict(epsilon=_slider(.2,.04,.6,.02,'epsilon')),
        lambda: dict(morph=_slider(0,0,1,.1,'Transformation'),closure=_slider(6,4,14,1,'R_C')),
        lambda: dict(distance=_slider(1.3,.6,3.,.1,'Distance'),width=_slider(1.7,.6,4,.1,'Opening'),boundary_eta=_slider(.6,-3,3,.1,'Boundary eta'),
             reveal=w.Checkbox(value=False,description='Reveal assumptions',indent=False)),
    ]
    selected=choices[stage]()
    def render(panel,values):
        if stage==4:
            panel.controls['closure'].layout.display='' if values['morph']==1 else 'none'
        fig,data=stage_figure(stage,**values)
        panel.image.value=_png(fig);panel.data=data
        panel.equation.value='<div style="font-size:95%;padding:4px 0">'+_equation(stage,data)+'</div>'
        panel.note.value='<p>'+CONCEPT_NOTES[stage]+'</p>'
    return _owned_panel(selected,render)


def reconstruction_explorer():
    """Independent sphere check; its controls never drive conceptual panels."""
    controls=dict(y=_slider(.3,-.6,.6,.1,'P transverse'),radius=_slider(2.1,1.5,2.8,.1,'Surface radius'))
    def render(panel,values):
        fig,t=reconstruction_figure(**values);panel.image.value=_png(fig);panel.data=t
        error=abs((t['total']-t['reference'])/t['reference'])
        panel.equation.value=('<div><b>Analytic field:</b> '+_complex_html(t['reference'])+'</div>'+
                              '<div><b>Reconstructed field:</b> '+_complex_html(t['total'])+'</div>')
        panel.note.value=(f'<p><b>No screen: closed-sphere numerical check.</b> Relative complex error = {error:.2e}. '
                          '48 by 96 surface samples. Change the sphere radius: the ring contributions change, '
                          'while the reference endpoint stays fixed. Moving P changes both consistently.</p>')
    return _owned_panel(controls,render)


def linked_explorer():
    """Compatibility entry point; new notebooks place fixed concepts beside explanations."""
    return concept_explorer(0)


def write_static(directory):
    """Readable default states for notebook viewers without a kernel."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);paths=[]
    with rc_context({'svg.fonttype':'path','svg.hashsalt':'green_kirchhoff_8_1','font.family':'DejaVu Sans'}):
        figures=[(f'8_1_linked_{chr(65+i)}',stage_figure(i,accounting=3 if i==2 else 0,reveal=i==5)[0]) for i in range(6)]
        figures.append(('8_1_closed_reconstruction',reconstruction_figure()[0]))
        for name,fig in figures:
            path=directory/(name+'.svg')
            try:
                from io import StringIO
                stream=StringIO();fig.savefig(stream,format='svg',bbox_inches='tight',pad_inches=.06,metadata={'Date':None,'Title':name,'Description':'Original explanatory illustration; three-dimensional surfaces shown in cross-section.'})
                contents='\n'.join(line.rstrip() for line in stream.getvalue().splitlines())+'\n'
                if not path.exists() or path.read_text(encoding='utf8')!=contents:path.write_text(contents,encoding='utf8')
            finally:fig.clear()
            paths.append(path)
    return paths
