"""Lesson guides linked to the explorer's existing cached fields."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Arc
from .plotting import figure_png
from .grating_intro import _close_widget_tree

BLUE,ORANGE,GREEN='#2866a3','#b85c17','#17806b'


def _arrow(axis,start,end,color):
    axis.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color=color,lw=1.6))


def gaussian_order_coordinates_figure(v):
    model=v.model;c=model.configuration;pair=v.pair.value
    depth=v.detector.value;extent=max(20.,depth)
    fig=Figure(figsize=(9.4,4.2),dpi=120,layout='constrained');FigureCanvasAgg(fig)
    left,right=fig.subplots(1,2)
    lo,hi=v.view_bounds['x']
    centers=[extent*model.basis(m)[1][0]/(-model.basis(m)[1][2]) for m in pair]
    lo,hi=min(lo,min(centers)-2*c.waist),max(hi,max(centers)+2*c.waist)
    show_x=(v.x>=lo)&(v.x<=hi);show_s=v.depths<=extent
    left.axhline(0,color='#526170',lw=1.2)
    left.text(lo+.02*(hi-lo),.16*extent,'Grating',fontsize=8)
    left.axhline(-depth,color=GREEN,lw=1)
    left.text(hi,-depth+.04*extent,f'Detector: s={depth:g} µm',ha='right',color=GREEN,fontsize=8)
    angles=[]
    for i,(order,color) in enumerate(zip(pair,(BLUE,ORANGE))):
        transverse,direction=model.basis(order)
        values=v.data['intensities'][i][show_s][:,show_x]
        left.contour(v.x[show_x],-v.depths[show_s],values,levels=[.135,.607],colors=[color],linewidths=.65,alpha=.45)
        end=extent*direction/(-direction[2])
        _arrow(left,(0,0),end[[0,2]],color)
        waist=np.array([-c.waist,c.waist])[:,None]*transverse
        left.plot(waist[:,0],waist[:,2],':',color=color,lw=2)
        crossing=depth*direction[0]/(-direction[2])
        left.plot(crossing,-depth,'o',color=color,ms=4)
        left.annotate(f'Order {order:+d}',xy=end[[0,2]],xytext=(3,-10),textcoords='offset points',color=color,fontsize=8)
        angle=np.rad2deg(np.arcsin(direction[0]));angles.append(f'{order:+d}: {angle:+.2f}°')
        if abs(angle)>1e-9:
            radius=.23*extent
            left.add_patch(Arc((0,0),2*radius,2*radius,theta1=min(-90,-90+angle),theta2=max(-90,-90+angle),color=color,lw=1))
    _arrow(left,(0,.29*extent),(0,.03*extent),GREEN)
    left.text(.03*(hi-lo),.25*extent,'Incident',color=GREEN,fontsize=8)
    left.set(title='Selected directions: '+', '.join(angles),xlabel='x (µm)',ylabel='z (µm)',xlim=(lo,hi),ylim=(-1.16*extent,.36*extent),aspect='equal')
    left.text(.02,.02,'Dotted: waist planes · Thin contours: intensity',transform=left.transAxes,fontsize=7)
    transverse,direction=model.basis(pair[1])
    ell=max(2*c.waist,.45*extent/(-direction[2]));q=1.3*c.waist
    projection=ell*direction;point=projection+q*transverse
    _arrow(right,(0,0),point[[0,2]],'#526170')
    _arrow(right,(0,0),projection[[0,2]],ORANGE)
    _arrow(right,projection[[0,2]],point[[0,2]],'#8d47a3')
    _arrow(right,(0,0),(.85*q*transverse)[[0,2]],'#8d47a3')
    right.plot([point[0],(q*transverse)[0]],[point[2],(q*transverse)[2]],':',color='#a5adb5',lw=1)
    right.plot(*point[[0,2]],'o',color='#243444',ms=4)
    right.annotate(r'$P:\ \mathbf{r}=\ell\hat{\mathbf{l}}+q\hat{\mathbf{q}}$',xy=point[[0,2]],xytext=(0,-20),textcoords='offset points',ha='center',fontsize=9)
    right.annotate('ℓ',xy=(.6*projection)[[0,2]],xytext=(6,0),textcoords='offset points',color=ORANGE,fontsize=11)
    right.annotate('q',xy=(projection+.5*q*transverse)[[0,2]],xytext=(0,-12),textcoords='offset points',color='#8d47a3',fontsize=11)
    right.annotate(r'$\hat{\mathbf{q}}$',xy=(.85*q*transverse)[[0,2]],xytext=(-10,6),textcoords='offset points',color='#8d47a3',fontsize=10)
    size=.2*c.waist
    corner=np.array([projection-size*direction,projection-size*direction+size*transverse,projection+size*transverse])
    right.plot(corner[:,0],corner[:,2],color='#526170',lw=1)
    cloud=np.array([np.zeros(3),projection,point,q*transverse]);xmin,xmax=cloud[:,0].min(),cloud[:,0].max();zmin=cloud[:,2].min()
    span=max(xmax-xmin,-zmin)+1.8*c.waist;center=(xmin+xmax)/2
    right.axhline(0,color='#b6bdc4',lw=.7)
    right.set(title=f'Order {pair[1]:+d}: local projection',xlabel='x (µm)',ylabel='z (µm)',xlim=(center-span/2,center+span/2),ylim=(zmin-.9*c.waist,.9*c.waist),aspect='equal')
    right.text(.02,.02,r'$\hat{\mathbf{l}}\parallel\mathbf{k}_n,\quad\hat{\mathbf{q}}\perp\hat{\mathbf{l}}$; $y=0$.',transform=right.transAxes,fontsize=8)
    for ax in (left,right):
        ax.tick_params(labelsize=8);ax.title.set_fontsize(10);ax.spines[['top','right']].set_visible(False)
    return fig


def gaussian_overlap_sequence_figure(v):
    model=v.model;c=model.configuration;pair=v.pair.value
    depths=(0.,v.detector.value,80.)
    fig=Figure(figsize=(10,3.1),dpi=120,layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(1,3,sharex=True,sharey=True);bounds=[]
    for depth in depths:
        for order in pair:
            direction=model.basis(order)[1];cosine=-direction[2]
            center=depth*direction[0]/cosine;width=c.waist*np.sqrt(1+(depth/cosine/c.rayleigh)**2)/cosine
            bounds.extend([center-2*width,center+2*width])
    xlimits=(max(v.x[0],min(bounds)),min(v.x[-1],max(bounds)))
    for index,(ax,depth) in enumerate(zip(axes,depths)):
        row=int(round(depth));first,second=v.data['intensities'][:,row]
        ax.plot(v.x,first,color=BLUE,lw=1.5,label=f'I(order {pair[0]:+d})')
        ax.plot(v.x,second,color=ORANGE,lw=1.5,label=f'I(order {pair[1]:+d})')
        ax.fill_between(v.x,np.sqrt(first*second),color=GREEN,alpha=.24,label='Overlap integrand')
        centers=[];widths=[]
        for order in pair:
            direction=model.basis(order)[1];cosine=-direction[2]
            centers.append(depth*direction[0]/cosine)
            widths.append(c.waist*np.sqrt(1+(depth/cosine/c.rayleigh)**2))
        if depth>0:
            ax.annotate('',xy=(centers[1],1.01),xytext=(centers[0],1.01),arrowprops=dict(arrowstyle='<->',lw=.8,color='#526170'))
        ax.text(.04,.95,f'Separation: {abs(centers[1]-centers[0]):.1f} µm',transform=ax.transAxes,fontsize=8)
        ax.text(.04,.85,f'Local radii: {widths[0]:.1f}, {widths[1]:.1f} µm',transform=ax.transAxes,fontsize=8)
        title='Selected' if index==1 else 'Reference'
        ax.set(title=f'{title}: {depth:g} µm · Oline={v.data["overlap"][row]:.3f}',xlabel='Detector x (µm)',xlim=xlimits,ylim=(0,1.22))
        ax.title.set_fontsize(10);ax.tick_params(labelsize=8);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Relative intensity');axes[0].legend(loc='center right',fontsize=7,frameon=False)
    fig.supxlabel('Shared experiment controls. Middle panel follows detector depth; reference planes remain at 0 and 80 µm.',fontsize=8)
    return fig


class GaussianLessonAids:
    """Lazy owned images, updated by the explorer without extra observers."""
    def __init__(self,explorer):
        import ipywidgets as w
        self.explorer=explorer
        self.coordinates=w.Image(format='png',layout=w.Layout(width='100%'))
        self.overlap=w.Image(format='png',layout=w.Layout(width='100%'))
        self._coordinate_panel=w.Accordion(children=[self.coordinates],selected_index=None)
        self._coordinate_panel.set_title(0,'Optional: local beam-coordinate geometry')
        self._coordinate_panel.observe(self._coordinate_visibility,names='selected_index')
        self._owned=w.VBox([self._coordinate_panel,self.overlap])
        self._active=[False,False];self._keys=[None,None];self.renders=[0,0];self._closed=False

    def coordinates_view(self):
        self._active[0]=True;self.refresh();return self.coordinates

    def coordinates_panel(self):
        return self._coordinate_panel

    def _coordinate_visibility(self,change):
        self._active[0]=self._coordinate_panel.selected_index is not None
        if self._active[0]:self.refresh()

    def overlap_view(self):
        self._active[1]=True;self.refresh();return self.overlap

    def refresh(self):
        if self._closed:return
        v=self.explorer;key=(v._key,v.detector.value)
        for index,(active,widget,builder) in enumerate(zip(self._active,(self.coordinates,self.overlap),(gaussian_order_coordinates_figure,gaussian_overlap_sequence_figure))):
            if active and key!=self._keys[index]:
                fig=builder(v)
                try:widget.value=figure_png(fig)
                finally:fig.clear()
                self._keys[index]=key;self.renders[index]+=1

    def close(self):
        if self._closed:return
        self._closed=True
        self._coordinate_panel.unobserve(self._coordinate_visibility,names='selected_index')
        _close_widget_tree(self._owned)
        self.coordinates.value=self.overlap.value=b'';self.explorer=None
