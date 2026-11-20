"""Original historical, geometrical and mathematical diagrams for notebook 8.1."""
from pathlib import Path
from io import StringIO
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib import rc_context

BLUE='#2866a3'
ORANGE='#b85c17'
GREEN='#23734d'
INK='#243444'
GREY='#89939e'


def arrow(ax,start,end,color=GREEN,style='->',lw=1.4):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=10,
                                linewidth=lw,color=color))


def sketch_axis(fig,rect):
    ax=fig.add_axes(rect)
    ax.set(xlim=(0,1),ylim=(0,1),aspect='equal')
    ax.axis('off')
    return ax


def aperture(ax,x=.34):
    for low,high in [(.12,.42),(.58,.88)]:
        ax.plot([x,x],[low,high],color=GREY,lw=5,solid_capstyle='butt')
    ax.text(x,.03,'aperture',ha='center',fontsize=9,color=INK)


def huygens(ax):
    ax.plot([.25,.25],[.12,.88],color=BLUE,lw=2)
    for y in [.28,.50,.72]:
        ax.add_patch(Arc((.25,y),.44,.44,theta1=-90,theta2=90,color=BLUE,lw=1.1))
    ax.plot([.47,.47],[.12,.88],color=ORANGE,lw=2)
    arrow(ax,(.62,.50),(.92,.50))
    ax.text(.12,.98,'wavefront',fontsize=9,color=BLUE)
    ax.text(.44,.03,'new envelope',fontsize=9,color=ORANGE)


def fresnel(ax):
    aperture(ax)
    ax.plot(.04,.5,'o',color=ORANGE,ms=4)
    ax.text(.04,.64,'source',fontsize=9,ha='center')
    ax.plot([.88,.88],[.08,.92],color=GREY,lw=2)
    for yy in [.42,.58]:
        ax.plot([.04,.88],[.5,.5+(yy-.5)*.84/.30],ls='--',color=GREY,lw=.9)
        ax.plot([.34,.86],[yy,.66],color=GREY,lw=1,ls=':')
    for yy in [.43,.50,.57,.64,.71]:
        ax.plot([.89,.95],[yy,yy],color=ORANGE,lw=2)
    arrow(ax,(.48,.30),(.72,.30))
    ax.text(.88,.97,'screen',ha='center',fontsize=9)


def fraunhofer(ax):
    for lo,hi in [(.12,.4),(.46,.54),(.60,.88)]:
        ax.plot([.25,.25],[lo,hi],color=GREY,lw=4)
    for y in [.3,.5,.7]:arrow(ax,(.02,y),(.18,y))
    for yy in [.22,.50,.78]:arrow(ax,(.27,.5),(.74,yy))
    # Telescope tube aligned to the upper angular direction.
    corners=np.array([[.72,.70],[.93,.82],[.89,.89],[.68,.77]])
    ax.add_patch(Polygon(corners,closed=True,fill=False,edgecolor=INK,lw=1.4))
    ax.text(.77,.98,'telescope',fontsize=9,ha='center')
    ax.text(.25,.03,'slits / grating',fontsize=9,ha='center')
    ax.text(.62,.12,'directions',fontsize=9,ha='center',color=GREEN)


def reference_geometry(ax):
    """Common S-to-P geometry for the probe and secondary-response views."""
    aperture(ax,x=.23)
    ax.plot([.23,.23],[.42,.58],color=BLUE,lw=1.8)
    ax.text(.15,.94,'S',fontsize=9,color=BLUE)
    ax.plot(.82,.55,'o',color=ORANGE,ms=4)
    ax.text(.87,.57,'P',fontsize=9,color=ORANGE)


def boundary_probe(ax):
    reference_geometry(ax)
    for radius in [.20,.36,.52]:
        ax.add_patch(Arc((.82,.55),2*radius,2*radius,theta1=145,theta2=215,
                         color=ORANGE,lw=.9,ls='--'))
    ax.plot([.23,.82],[.50,.55],color=GREY,lw=.8,ls=':')
    # A probe construction, not a physical incoming-wave boundary condition.
    ax.text(.52,.14,'probe at P',ha='center',fontsize=8,color=ORANGE)


def secondary_contributions(ax):
    reference_geometry(ax)
    for y in [.44,.50,.56]:
        ax.plot(.23,y,'o',color=BLUE,ms=2.5)
        ax.add_patch(Arc((.23,y),.40,.40,theta1=-65,theta2=65,color=BLUE,lw=.8))
        ax.plot([.23,.82],[y,.55],color=GREY,lw=.8,ls=':')
    arrow(ax,(.43,.30),(.70,.30))
    ax.text(.56,.14,'weighted waves',ha='center',fontsize=8,color=BLUE)


def debye(ax):
    center=np.array([.82,.50]);radius=.64
    theta=np.linspace(np.deg2rad(135),np.deg2rad(225),100)
    cap=center+radius*np.column_stack([np.cos(theta),np.sin(theta)])
    ax.plot(*cap.T,color=BLUE,lw=2)
    for angle in [140,160,180,200,220]:
        point=center+radius*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
        ax.plot([point[0],center[0]],[point[1],center[1]],ls=':',color=GREY,lw=.8)
        arrow(ax,point+.15*(center-point),point+.58*(center-point))
    ax.add_patch(Circle(center,.055,color=ORANGE,alpha=.35))
    ax.plot(*center,'o',color=ORANGE,ms=3)
    ax.text(.07,.98,'reference cap',fontsize=9,color=BLUE)
    ax.text(.77,.30,'focus',fontsize=9,color=ORANGE)


def conducting_source(ax):
    ax.plot([.20,.20],[.16,.84],color=GREY,lw=3)
    for y in np.linspace(.19,.79,7):ax.plot([.1,.2],[y-.06,y],color=GREY,lw=.8)
    ax.plot(.20,.50,'o',color=ORANGE,ms=4)
    for radius in [.18,.36,.54]:
        ax.add_patch(Arc((.20,.5),2*radius,2*radius,theta1=-45,theta2=45,color=BLUE,lw=1))
    arrow(ax,(.43,.5),(.88,.5))
    ax.text(.50,.03,'localized source',fontsize=8,ha='center')
    ax.text(.50,.95,'conducting plane',fontsize=8,ha='center')


def plane_spectrum(ax):
    # Same reference plane S and observation direction as the boundary sketches.
    ax.plot([.23,.23],[.15,.85],color=GREY,lw=2)
    ax.plot([.82,.82],[.15,.85],color=GREY,lw=1)
    ax.text(.23,.94,'S',ha='center',fontsize=9,color=BLUE)
    ax.text(.82,.94,'observation',ha='center',fontsize=7.5,color=INK)
    for angle,color in [(25,BLUE),(-25,ORANGE)]:
        theta=np.deg2rad(angle)
        normal=np.array([np.cos(theta),np.sin(theta)])
        tangent=np.array([-np.sin(theta),np.cos(theta)])
        for x0 in [.35,.53,.71]:
            center=np.array([x0,.5])
            low,high=-.48,.48
            for dim,(minimum,maximum) in enumerate([(.23,.82),(.20,.80)]):
                limits=sorted([(minimum-center[dim])/tangent[dim],
                               (maximum-center[dim])/tangent[dim]])
                low,high=max(low,limits[0]),min(high,limits[1])
            if low<high:
                segment=center+np.array([[low],[high]])*tangent
                ax.plot(*segment.T,color=color,lw=.9)
        arrow(ax,(.32,.5),np.array([.32,.5])+.40*normal,lw=1.1)
    ax.plot(.82,.55,'o',color=ORANGE,ms=3)
    ax.text(.87,.55,'P',fontsize=8,color=ORANGE)
    ax.text(.53,.035,'plane components',ha='center',fontsize=7.5)


def timeline_figure():
    """Six compressed chronological groups with original physical sketches."""
    fig=Figure(figsize=(10.5,14.4),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.045,.975,'Historical questions and physical pictures',fontsize=18,weight='bold',color=INK,va='top')
    ax.text(.045,.947,'Compressed spacing; groups are placed by their earliest date.',fontsize=10.5,color=INK)
    headers=[(.055,'Date and author'),(.275,'Physical geometry'),(.56,'Question'),(.79,'Mathematical idea')]
    for x,title in headers:ax.text(x,.912,title,fontsize=11.5,weight='bold',color=INK)
    ys=[.83,.68,.53,.38,.23,.08]
    ax.plot([.028,.028],[ys[-1]-.018,ys[0]+.025],color=GREY,lw=1.5)
    rows=[
      ('1678 / 1690','Huygens [2]', 'How does a\nwavefront advance?', 'Construct the next\nfront as an envelope\nof secondary wavelets.\nNot yet a phase-sensitive\ndiffraction integral.'),
      ('1818 memoir','Fresnel [3]', 'Why are there fringes\nnear a geometrical\nshadow?', 'Add complex contributions\nwith relative phase.\nSquare the total to\npredict intensity.'),
      ('Early 1820s\n1822 publication','Fraunhofer [4]', 'Which directions\ncarry diffracted light?', 'Measure angular\npatterns; connect to\nthe modern far-field\ndescription.'),
      ('1882-1883 Kirchhoff\n1897 Rayleigh\nLater Sommerfeld','Boundary integrals [5]', 'What data on S\ndetermine the field\nat P?', 'Green boundary identity;\nfield or normal derivative.\nSeparate an exact theorem\nfrom approximate\naperture data.'),
      ('1909','Debye [6]', 'What field forms\nnear a focus?', 'Combine a finite cone\nof converging directions.\nA geometrical cap is\nnot necessarily a\nphysical curved pupil.'),
      ('1919 Weyl [7]\n1950 Booker-Clemmow [8]','Directional components', 'How can directional\ncomponents simplify\nthe problem?', 'Weyl: spherical / plane\nresponse identity.\nConvolution becomes\nspectral multiplication;\npropagate each component.')]
    for i,(y,(date,author,question,contribution)) in enumerate(zip(ys,rows)):
        ax.axhspan(y-.063,y+.062,color='#f4f7fa' if i%2==0 else 'white',zorder=-1)
        ax.plot(.028,y,'o',color=BLUE,ms=6)
        ax.text(.055,y+.041,date,fontsize=10.5,color=INK,va='top',linespacing=1.45)
        ax.text(.055,y-.010 if i not in [3,5] else y-.036,author,fontsize=10.3,weight='bold',color=BLUE,va='top')
        ax.text(.56,y+.026,question,fontsize=11.2,color=INK,va='top',linespacing=1.5)
        ax.text(.79,y+.026,contribution,fontsize=10.5,color=INK,va='top',linespacing=1.5)
        arrow(ax,(.239,y),(.264,y),lw=1)
        arrow(ax,(.520,y),(.548,y),lw=1)
        arrow(ax,(.756,y),(.779,y),lw=1)
        if i==3:
            boundary_probe(sketch_axis(fig,[.274,y-.057,.115,.114]))
            secondary_contributions(sketch_axis(fig,[.399,y-.057,.115,.114]))
        elif i<5:
            sketch=[huygens,fresnel,fraunhofer,None,debye][i]
            sketch(sketch_axis(fig,[.274,y-.057,.24,.114]))
        else:
            conducting_source(sketch_axis(fig,[.274,y-.057,.115,.114]))
            plane_spectrum(sketch_axis(fig,[.399,y-.057,.115,.114]))
    return fig


def box(ax,xy,width,height,text,*,color=BLUE,fill='#f1f6fa',fontsize=11):
    patch=FancyBboxPatch(xy,width,height,boxstyle='round,pad=0.012',lw=1.2,
                         edgecolor=color,facecolor=fill)
    ax.add_patch(patch)
    ax.text(xy[0]+width/2,xy[1]+height/2,text,ha='center',va='center',
            fontsize=fontsize,color=INK,linespacing=1.4)


def relationships_figure():
    """A symmetric exact-representation pair and centred approximation chain."""
    fig=Figure(figsize=(10.5,8.8),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.5,.975,'Representations and approximations',ha='center',
            fontsize=18,weight='bold',color=INK,va='top')
    ax.text(.5,.936,'One boundary problem; two equivalent full descriptions',
            ha='center',fontsize=11,color=GREY)
    box(ax,(.20,.815),.60,.085,
        'Scalar Helmholtz equation\nPrescribed plane field; homogeneous half-space\nOutgoing or decaying continuation',fontsize=11)
    # Mirror-symmetric branches into equal-sized full representations.
    arrow(ax,(.34,.800),(.235,.765))
    arrow(ax,(.66,.800),(.765,.765))
    box(ax,(.055,.635),.36,.11,
        'Full Rayleigh-Sommerfeld I\nReal-space convolution',fontsize=11.5)
    box(ax,(.585,.635),.36,.11,
        'Full angular spectrum\nFourier-space multiplication',fontsize=11.5)
    arrow(ax,(.432,.690),(.568,.690),style='<->',lw=1.8)
    ax.text(.5,.762,'Equivalent descriptions',ha='center',fontsize=10.5,
            weight='bold',color=GREEN)
    ax.text(.5,.573,'Weyl identity\nSpherical-kernel / plane-wave connection',
            ha='center',va='center',fontsize=10.5,color=INK,linespacing=1.5)
    # Both representations feed the same approximation on the centre line.
    ax.plot([.235,.235,.5],[.617,.54,.515],color=GREEN,lw=1.3)
    ax.plot([.765,.765,.5],[.617,.54,.515],color=GREEN,lw=1.3)
    arrow(ax,(.5,.515),(.5,.445),lw=1.6)
    ax.text(.5,.476,'Paraxial approximation',ha='center',fontsize=11,color=GREEN,
            bbox=dict(facecolor='white',edgecolor='none',pad=3))
    box(ax,(.25,.335),.50,.09,
        'Fresnel propagation\nQuadratic phase; suitable small-angle geometry',fontsize=11.5)
    arrow(ax,(.5,.318),(.5,.245),lw=1.6)
    ax.text(.5,.280,'Additional far-field approximation',ha='center',fontsize=11,color=GREEN,
            bbox=dict(facecolor='white',edgecolor='none',pad=3))
    box(ax,(.25,.140),.50,.085,
        'Usual paraxial Fraunhofer form\nFourier-transform angular pattern',fontsize=11.5)
    # A separate, full-width outlook preserves balance without implying a step.
    box(ax,(.10,.018),.80,.073,
        'Debye focused-wave outlook\nNear focus; finite angular cone; can retain large angles\nSeparate geometry, not a step in the paraxial chain',
        color=ORANGE,fill='#fff5eb',fontsize=10.5)
    return fig


def greens_response_figure():
    """An illustrative free-space unit source and two complex point responses."""
    fig=Figure(figsize=(10.5,3.8),facecolor='white')
    canvas=fig.add_axes([0,0,1,1]);canvas.set(xlim=(0,1),ylim=(0,1));canvas.axis('off')
    canvas.text(.04,.96,"Green's function: the response to one tiny source",fontsize=16,
                weight='bold',color=INK,va='top')
    ax=fig.add_axes([.025,.12,.43,.73])
    ax.set(xlim=(-.56,.62),ylim=(-.53,.53),aspect='equal');ax.axis('off')
    for radius in [.12,.24,.36,.48]:
        ax.add_patch(Circle((0,0),radius,fill=False,color=BLUE,lw=1.1))
    ax.plot(0,0,'o',color=ORANGE,ms=6)
    ax.annotate('Q (unit source)',xy=(0,0),xytext=(-.08,-.15),
                fontsize=10,ha='right',color=ORANGE,
                arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.9))
    # Equal-spaced circles are equal-phase fronts. This illustrative pair is
    # at two wavelengths and three-and-a-half wavelengths from the source.
    # The opposite phasors and magnitude ratio therefore agree with e^(ikR)/R.
    for point,label,xytext in [((.24,0),'P1',(.29,.24)),((.42,0),'P2',(.54,-.22))]:
        ax.plot(*point,'o',color=INK,ms=4)
        ax.annotate(label,xy=point,xytext=xytext,fontsize=11,color=INK,
                    ha='center',arrowprops=dict(arrowstyle='-',color=GREY,lw=.9))
    arrow(ax,(-.15,.28),(-.22,.41))
    arrow(ax,(-.17,-.26),(-.26,-.40))
    canvas.text(.24,.09,'Slice through spherical wavefronts',ha='center',fontsize=10,color=BLUE)
    canvas.text(.72,.79,'Each point receives an amplitude and a phase',ha='center',fontsize=11,color=INK)
    for rect,title,value in [([.53,.28,.17,.40],'Nearer P1',1.),
                             ([.76,.28,.17,.40],'Farther P2',-.24/.42)]:
        pa=fig.add_axes(rect);pa.set(xlim=(-1.25,1.25),ylim=(-1.15,1.15),aspect='equal');pa.axis('off')
        pa.axhline(0,color=GREY,lw=.8);pa.axvline(0,color=GREY,lw=.8)
        pa.add_patch(Circle((0,0),1.,fill=False,color=GREY,lw=.7,ls=':'))
        arrow(pa,(0,0),(value,0),color=ORANGE,lw=2.3)
        pa.set_title(title,fontsize=11,color=INK,pad=3)
        pa.text(1.1,-.22,'Re',fontsize=8,color=GREY)
        pa.text(.08,1.0,'Im',fontsize=8,color=GREY)
    canvas.text(.72,.21,'Stronger response          Weaker, different phase',ha='center',fontsize=10,color=INK)
    canvas.text(.72,.08,'Orange phasors: length = magnitude; angle = phase',ha='center',fontsize=10,color=ORANGE)
    return fig


def boundary_greens_figure():
    """A common reference plane and the Green constructions for its two data choices."""
    fig=Figure(figsize=(10.5,6.1),facecolor='white')
    page=fig.add_axes([0,0,1,1]);page.set(xlim=(0,1),ylim=(0,1));page.axis('off')
    page.text(.035,.97,"Kirchhoff and Rayleigh-Sommerfeld: choose the elementary response",
              fontsize=15,weight='bold',color=INK,va='top')
    page.text(.5,.895,r'Two boundary terms:  $U\,\partial_n G$  and  $G\,\partial_n U$',
              ha='center',fontsize=13,color=INK)
    descriptions=[('Kirchhoff',None,r'$G=g_P$',
                    'Both boundary terms remain',r'Theorem uses $U$ and $\partial_n U$'),
                  ('Rayleigh-Sommerfeld I',-1,r'$G_-=g_P-g_{P\prime}$',
                    r'On S: $G_-=0$',r'Prescribe $U$ on S'),
                  ('Rayleigh-Sommerfeld II',1,r'$G_+=g_P+g_{P\prime}$',
                    r'On S: $\partial_n G_+=0$',r'Prescribe $\partial_n U$ on S')]
    for i,(title,image_sign,formula,cancellation,known) in enumerate(descriptions):
        left=.035+i*.325
        page.add_patch(FancyBboxPatch((left,.17),.295,.64,boxstyle='round,pad=.008',
                       facecolor='#f4f7fa',edgecolor='#d6dfe7',lw=1))
        page.text(left+.1475,.775,title,ha='center',fontsize=11.8,weight='bold',color=INK)
        page.text(left+.1475,.715,formula,ha='center',fontsize=13,color=INK)
        ax=fig.add_axes([left+.015,.37,.265,.30]);ax.set(xlim=(-1,1),ylim=(-.9,.9),aspect='equal');ax.axis('off')
        # S remains a complete mathematical plane; the physical screen has a gap.
        ax.plot([0,0],[-.8,.8],color=GREY,lw=1,ls=(0,(3,3)),zorder=1)
        for lo,hi in [(-.8,-.48),(.08,.8)]:
            ax.plot([0,0],[lo,hi],color=INK,lw=6,solid_capstyle='butt',zorder=2)
        ax.text(-.10,-.49,'opening',fontsize=8,ha='right',color=INK)
        ax.text(0,.84,'S',ha='center',fontsize=10,color=INK)
        q=(0,-.22)
        ax.plot(*q,'o',color=INK,ms=3)
        ax.text(.07,-.36,'Q',fontsize=9,ha='left',color=INK)
        ax.plot([0,.64],[-.22,.27],ls=':',color=GREY,lw=1)
        ax.add_patch(Circle((.64,.27),.12,fill=False,edgecolor=ORANGE,lw=1.4))
        ax.text(.64,.27,'+',ha='center',va='center',color=ORANGE,fontsize=13,weight='bold')
        ax.text(.64,.49,'P',ha='center',fontsize=10,color=ORANGE)
        if image_sign is not None:
            ax.plot([-.64,0],[.27,-.22],ls=':',color=GREY,lw=1)
            color=BLUE if image_sign<0 else ORANGE
            ax.add_patch(Circle((-.64,.27),.12,fill=False,edgecolor=color,lw=1.4))
            ax.text(-.64,.27,'-' if image_sign<0 else '+',ha='center',va='center',
                    fontsize=13,color=color,weight='bold')
            ax.text(-.64,.49,"P'",ha='center',fontsize=10,color=color)
        arrow(ax,(.08,-.60),(.49,-.60),lw=1)
        ax.text(.57,-.61,'n',fontsize=9,color=GREEN)
        page.text(left+.1475,.32,cancellation,ha='center',fontsize=11,color=GREEN)
        page.text(left+.1475,.235,known,ha='center',fontsize=11.2,weight='bold',color=INK)
    page.text(.5,.115,'Opposite probes cancel G on S; equal probes cancel its normal derivative.',
              ha='center',fontsize=11,color=INK)
    page.text(.5,.055,"Kirchhoff's theorem is valid; the usual approximate aperture prescription causes the inconsistency.",
              ha='center',fontsize=10.5,color=ORANGE)
    return fig


def boundary_data_figure():
    """Original aperture/vector geometry: a cross-section of a 3D problem."""
    fig=Figure(figsize=(10.5,5.2),facecolor='white')
    ax=fig.add_axes([.025,.16,.95,.75])
    ax.set(xlim=(-3.6,6.5),ylim=(-2.3,2.6),aspect='equal');ax.axis('off')
    fig.text(.5,.96,'From an aperture point Q to an observation point P',
             ha='center',fontsize=17,weight='bold',color=INK,va='top')
    # The same external point source as the live examples. These curved fronts
    # are a cross-section of spherical illumination, not a new plane-wave case.
    source=np.array([-3.,-.4])  # (s,y); O below is only the coordinate origin.
    angles=np.linspace(-1.15,1.15,121)
    for radius in (1.,1.8,2.6):
        front=source+radius*np.column_stack([np.cos(angles),np.sin(angles)])
        ax.plot(*front.T,'--',color=BLUE,lw=1.4,alpha=.65)
    ax.plot(*source,'*',color=BLUE,ms=10)
    ax.text(-3.5,-.95,r'Physical $P_0$',fontsize=10,color=BLUE)
    arrow(ax,(-1.95,-.0325),(-.85,.3525))
    ax.text(-2.,-2.15,'Incident illumination',ha='center',fontsize=10,color=BLUE)
    ax.text(-1.5,.48,r'$\mathbf{k}_i$',fontsize=12,color=GREEN)
    ax.axvspan(0,6.4,color='#edf6f0',zorder=-3)
    ax.plot([0,0],[-2,2.35],':',color=GREY,lw=1)
    for lo,hi in [(-2,-1),(1,2.35)]:
        ax.plot([0,0],[lo,hi],color=INK,lw=7,solid_capstyle='butt')
    ax.text(.13,2.15,'Screen B',fontsize=11,color=INK)
    arrow(ax,(-.5,-1),(-.5,1),style='<->',color=INK,lw=1)
    ax.text(-.64,0,'2b',ha='right',va='center',fontsize=12,color=INK)
    ax.text(.12,-1.4,'Boundary A+B; opening A',fontsize=10,color=INK)
    origin=(0,0); q=(0,.65); point=(4.8,1.65)
    ax.plot(*origin,'o',color=INK,ms=4)
    ax.text(.1,-.3,'O',fontsize=11,color=INK)
    ax.plot(*q,'o',color=GREEN,ms=6)
    ax.text(-.12,.65,'Q',ha='right',va='center',fontsize=12,color=GREEN)
    ax.plot(*point,'o',color=ORANGE,ms=6)
    ax.plot([point[0],point[0]],[-1.1,1.95],color=GREY,lw=1.2,zorder=-1)
    ax.text(point[0]+.12,-1.05,'Observation\nscreen',fontsize=10,color=INK)
    ax.text(4.97,1.67,'P',fontsize=12,color=ORANGE)
    arrow(ax,origin,q,color=BLUE,lw=2)
    ax.text(.15,.27,r"$\mathbf{r}'$",fontsize=13,color=BLUE)
    arrow(ax,origin,point,color=BLUE,lw=1.6)
    ax.text(2.4,.56,r'$\mathbf{r}$',fontsize=14,color=BLUE)
    ax.annotate('',point,q,arrowprops=dict(arrowstyle='->',color=ORANGE,lw=1.8,linestyle='--'))
    ax.text(2.45,1.42,r"$\mathbf{R}=\mathbf{r}-\mathbf{r}'$",ha='center',fontsize=13,color=ORANGE)
    arrow(ax,(.12,.65),(1.05,.65),lw=1.2)
    ax.text(.78,.43,'n',fontsize=11,color=GREEN)
    theta=np.degrees(np.arctan2(point[1]-q[1],point[0]))
    ax.add_patch(Arc(q,1.35,1.35,theta1=0,theta2=theta,color=GREEN,lw=1))
    ax.text(.83,.82,r'$\theta$',fontsize=12,color=GREEN)
    # Orientation key is separate from the position-vector origin.
    arrow(ax,(3.9,-1.55),(5.25,-1.55),color=INK,lw=1)
    arrow(ax,(3.9,-1.55),(3.9,-.65),color=INK,lw=1)
    ax.text(5.35,-1.55,r'$s=-z$',va='center',fontsize=11,color=INK)
    ax.text(4.7,-2.0,r'Positive propagation distance: $s=-z>0$',
            ha='center',fontsize=9.5,color=INK)
    ax.text(3.9,-.48,'y',ha='center',fontsize=11,color=INK)
    ax.text(4,2.2,'Unknown outgoing field',ha='center',fontsize=11,color=INK)
    fig.text(.5,.10,r"$Q=(\xi,\eta,0)$,  $P=(x,y,s)$;  $R=|\mathbf{r}-\mathbf{r}'|$ is the travel distance.",
             ha='center',fontsize=12,color=INK)
    fig.text(.5,.035,r'Cross-section: $x=\xi=0$. Full propagation still integrates both transverse coordinates.',
             ha='center',fontsize=10.5,color=INK)
    return fig


def poisson_arago_figure():
    """Original disk geometry and a computed, explicitly Fresnel profile."""
    from .propagation_interactive import poisson_arago_field
    fig=Figure(figsize=(10.5,3.7),layout='constrained',facecolor='white')
    a,b=fig.subplots(1,2)
    a.set(xlim=(-1.8,5),ylim=(-1.8,1.8));a.axis('off')
    a.fill_between([0,4.6],-1,1,color='#e9edf1')
    a.plot([0,0],[-1,1],color=INK,lw=8,solid_capstyle='butt')
    a.text(-.12,0,'Opaque\ndisk',ha='right',va='center',fontsize=11,color=INK)
    for y in (-1.45,1.45):arrow(a,(-1.5,y),(-.15,y))
    a.text(-.85,1.72,'Plane wave',ha='center',fontsize=11,color=GREEN)
    for y in (-1,1):
        a.plot([0,4],[y,0],'--',color=ORANGE,lw=1.5)
        a.plot(0,y,'o',color=ORANGE,ms=4)
    a.plot(4,0,'o',color=ORANGE,ms=7)
    a.text(4,-.4,'Axial P',ha='center',fontsize=11,color=ORANGE)
    a.text(2.5,-1.35,'Geometrical shadow',ha='center',fontsize=10,color=GREY)
    a.text(2.5,1.35,'Equal paths from the circular rim\nEqual incident phase: contributions reinforce',
           ha='center',fontsize=10,color=INK)
    a.set_title('Poisson-Arago spot: the disk is the obstacle',fontsize=12,color=INK)
    x=np.linspace(-10,10,501)
    intensity=abs(poisson_arago_field(x))**2
    b.axvspan(-5,5,color='#e9edf1',label='Geometrical shadow')
    b.plot(x,intensity,color=BLUE,lw=1.7,label='Fresnel / Babinet prediction')
    b.plot(0,1,'o',color=ORANGE,ms=5)
    b.annotate('Bright axial spot',xy=(0,1),xytext=(.6,1.2),fontsize=10,
               arrowprops=dict(arrowstyle='->',color=ORANGE),color=INK)
    b.set(xlabel=r'Transverse position / $\lambda$',ylabel=r'$I/I_{\rm incident}$',
          title=r'Computed profile: $a=5\lambda$, $s=25\lambda$',ylim=(0,max(1.5,float(max(intensity))*1.12)))
    b.legend(fontsize=8,loc='upper right')
    return fig


def boundary_consistency_figure():
    """A one-dimensional Helmholtz example, not an aperture solution."""
    fig=Figure(figsize=(11,5.5),facecolor='white')
    fig.text(.5,.97,'Zero field does not require zero normal slope',ha='center',
             va='top',fontsize=17,weight='bold',color=INK)
    fig.text(.5,.89,r"Local example: $U''+k^2U=0$, with $h=\partial_sU$ and $k=2\pi/\lambda$",
             ha='center',fontsize=12,color=INK)
    left=fig.add_axes([.075,.30,.39,.46])
    right=fig.add_axes([.575,.30,.39,.46])
    t=np.linspace(0,1.05,421)
    for ax in (left,right):
        ax.axhline(0,color=GREY,lw=.8)
        ax.axvline(0,color=GREY,lw=1)
        ax.set(xlim=(-.04,1.05),ylim=(-1.22,1.22),
               xlabel=r'Normal distance $s/\lambda$',ylabel=r'Field $U/C$')
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9)
    left.plot(t,np.sin(2*np.pi*t),color=BLUE,lw=2,label=r'$U=C\sin(ks)$')
    local=np.linspace(0,.15,20)
    left.plot(local,2*np.pi*local,'--',color=ORANGE,lw=1.5,label='Tangent at the plane')
    left.plot(0,0,'o',color=ORANGE,ms=6)
    left.annotate(r'$U(0)=0$'+'\n'+r'$h(0)=kC\ne0$',xy=(0,0),xytext=(.43,.64),
                  fontsize=11,color=INK,arrowprops=dict(arrowstyle='->',color=ORANGE))
    left.set_title('Zero value: a nonzero solution is possible',fontsize=11.5,color=BLUE,pad=12)
    left.legend(loc='lower left',fontsize=9,frameon=False)
    right.plot(t,np.zeros_like(t),color=GREEN,lw=2.5,label='Only compatible solution: U = 0')
    attempt=.4*(1-np.cos(2*np.pi*t))
    right.plot(t,attempt,'--',color=ORANGE,lw=1.8,marker='x',markevery=45,ms=5,
               label='Nonzero trial (x): fails the equation')
    right.plot(0,0,'o',color=GREEN,ms=6)
    right.text(.56,-.48,r'$U(0)=0$ and $h(0)=0$',ha='center',fontsize=11,color=INK)
    right.set_title('Both zero: no nonzero Helmholtz solution',fontsize=11.5,color=GREEN,pad=12)
    right.legend(loc='lower left',fontsize=8.5,frameon=False)
    fig.text(.27,.17,r'$U=A\cos(ks)+B\sin(ks)$: $U(0)=0$ sets $A=0$ only.',
             ha='center',fontsize=10.5,color=INK)
    fig.text(.77,.17,r'Also $h(0)=0$ sets $B=0$, so $U=0$ everywhere.',
             ha='center',fontsize=10.5,color=INK)
    fig.text(.77,.105,r"Trial curve: $U_t=0.4C(1-\cos ks)$; $U_t''+k^2U_t=0.4k^2C\ne0$.",
             ha='center',fontsize=9,color=ORANGE)
    fig.text(.5,.05,'A one-dimensional illustration of boundary compatibility, not the outgoing aperture solution.',
             ha='center',fontsize=10,color=INK)
    fig.text(.5,.012,'The trial is not a Kirchhoff field: that reconstruction satisfies Helmholtz but cannot recover both assumed boundary data.',
             ha='center',fontsize=9,color=INK)
    return fig



def helmholtz_rule_figure():
    """A plane-wave slice: real field and scaled curvature cancel locally."""
    fig=Figure(figsize=(10,3.5),layout='constrained',facecolor='white')
    ax=fig.subplots()
    t=np.linspace(0,2,501);u=np.cos(2*np.pi*t)
    ax.plot(t,u,color=BLUE,lw=2,label=r'Real field: $\mathrm{Re}\,U=\cos(ks)$')
    ax.plot(t,-u,'--',color=ORANGE,lw=2,label=r'Scaled curvature: $\mathrm{Re}(U^{\prime\prime}/k^2)$')
    ax.axhline(0,color=GREY,lw=.8)
    ax.plot([.375,.375],[-np.sqrt(.5),np.sqrt(.5)],':',color=GREEN,lw=2)
    ax.scatter([.375,.375],[-np.sqrt(.5),np.sqrt(.5)],color=GREEN,zorder=3)
    ax.set(xlabel=r'Distance $s/\lambda$',ylabel='Normalized real part',
           title=r'Helmholtz: field + scaled curvature = 0 at every position',ylim=(-1.45,1.45))
    ax.legend(loc='lower center',ncol=2,fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle(r'A familiar example: $U=e^{iks}$, so $U^{\prime\prime}+k^2U=0$',fontsize=13)
    return fig


def green_identity_figure():
    """Closed-domain identity geometry; distinguished from the aperture plane."""
    fig=Figure(figsize=(10,4),facecolor='white')
    left=fig.add_axes([.04,.16,.36,.68]);right=fig.add_axes([.60,.16,.36,.68])
    for ax in (left,right):
        ax.set(xlim=(-1.35,1.35),ylim=(-1.25,1.25),aspect='equal');ax.axis('off')
        ax.add_patch(Circle((0,0),1,fill=False,color=INK,lw=2))
        ax.plot(.25,.15,'o',color=ORANGE);ax.text(.34,.15,'P',color=ORANGE,fontsize=12)
    for x in np.linspace(-.85,.85,11):
        for y in np.linspace(-.85,.85,11):
            if x*x+y*y<.9: left.plot(x,y,'.',color=BLUE,alpha=.4,ms=4)
    left.text(0,-1.18,'Compare U and G throughout the volume',ha='center',fontsize=10)
    a=np.linspace(0,2*np.pi,19)[:-1]
    for t in a:
        q=np.array([np.cos(t),np.sin(t)])
        right.plot(*q,'o',color=BLUE,ms=4)
    q=np.array([np.cos(.75),np.sin(.75)])
    right.plot([q[0],.25],[q[1],.15],':',color=ORANGE)
    arrow(right,q,.73*q,color=GREEN)
    right.text(q[0]+.1,q[1]+.1,'Q',fontsize=12,color=BLUE)
    right.text(0,-1.18,'Equivalent boundary terms: values and slopes',ha='center',fontsize=10)
    fig.text(.5,.94,"Green's identity: move the bookkeeping to the boundary",ha='center',fontsize=15,color=INK)
    fig.text(.5,.53,'integration\nby parts',ha='center',fontsize=11,color=GREEN)
    fig.text(.5,.40,r'$\longrightarrow$',ha='center',fontsize=30,color=GREEN)
    fig.text(.5,.045,'Closed-boundary sketch; the aperture problem uses a plane plus a distant closing surface.',ha='center',fontsize=10,color=INK)
    return fig


def kirchhoff_sum_figure():
    """Illustrative complex weights, not a numerical aperture prediction."""
    fig=Figure(figsize=(10,4.2),layout='constrained',facecolor='white')
    axes=fig.subplots(1,2)
    value=np.array([.8+.3j,.4+.65j,-.2+.4j])
    slope=np.array([-.25+.1j,.15-.3j,.35-.1j])
    for ax,terms,title in zip(axes,[np.array([value.sum(),slope.sum()]),value+slope],
        ['Two contributions from the whole surface','The same total, grouped by patch']):
        start=0j
        for j,v in enumerate(terms):
            color=[BLUE,ORANGE][j] if len(terms)==2 else BLUE
            arrow(ax,(start.real,start.imag),((start+v).real,(start+v).imag),color=color,lw=2)
            start+=v
        arrow(ax,(0,0),(start.real,start.imag),color=GREEN,lw=2)
        ax.set(xlim=(-.15,1.8),ylim=(-.15,1.5),aspect='equal',xlabel='Real part',ylabel='Imaginary part',title=title)
        ax.axhline(0,color=GREY,lw=.6);ax.axvline(0,color=GREY,lw=.6)
        ax.spines[['top','right']].set_visible(False)
    axes[0].text(.04,.95,'Blue: sum of U (normal derivative G) dA\nOrange: sum of -G h dA\nGreen: final U(P)',transform=axes[0].transAxes,va='top',fontsize=9)
    axes[1].text(.04,.95,'Each blue arrow includes BOTH terms\nGreen: identical final U(P)',transform=axes[1].transAxes,va='top',fontsize=9)
    fig.suptitle('Kirchhoff: add complex contributions before calculating intensity\nIllustrative arrows; not a calculated aperture field',fontsize=13,color=INK)
    return fig


def write_illustrations(directory):
    """Generate the three current shared SVGs; retain other figure constructors."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    with rc_context({'svg.fonttype':'path','svg.hashsalt':'propagation_pictures_8_1','font.family':'DejaVu Sans'}):
        for name,make,title in [('8_1_propagation_relationships',relationships_figure,'Modern propagation relationships'),
                                ('8_1_boundary_data',boundary_data_figure,'Domain and prescribed boundary data'),
                                ('8_1_boundary_consistency',boundary_consistency_figure,'Field value, normal slope and Helmholtz compatibility')]:
            path=directory/(name+'.svg')
            stream=StringIO();figure=make()
            try:
                figure.savefig(stream,format='svg',metadata={'Title':title,'Date':None,
                    'Description':'Original explanatory geometry; blue wavefronts or labeled field data, green propagation arrows.'})
            finally:
                figure.clear()
            normalize=lambda text:'\n'.join(line.rstrip() for line in text.splitlines())+'\n'
            contents=normalize(stream.getvalue())
            # Preserve untouched assets; write changed SVGs without path-line
            # trailing spaces emitted by Matplotlib's serializer.
            if not path.exists() or normalize(path.read_text(encoding='utf-8'))!=contents:
                path.write_text(contents,encoding='utf-8')
            paths.append(path)
    return paths
