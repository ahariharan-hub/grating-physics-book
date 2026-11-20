"""Retained Plotly widget for analytical Gaussian beams; no field solver."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CAMERA = dict(eye=dict(x=1.45, y=-2., z=1.1), projection=dict(type='orthographic'))


def fitted_view_bounds(model, pair, depth):
    """Fit two-radius beam footprints plus 10% margin; display crop only.

    Match square window edges to the detector's 0.1/0.4 µm sample spacings.
    The cubic scene also contains the incident segment and propagation path.
    """
    c=model.configuration
    low,high=-2*c.waist,2*c.waist
    yhalf=2*c.waist
    for order in pair:
        direction=model.basis(order)[1]
        cosine=-direction[2]
        for s in (0.,float(depth)):
            center=s*direction[0]/cosine
            width=c.waist*np.sqrt(1+(s/cosine/c.rayleigh)**2)
            low=min(low,center-2*width/cosine)
            high=max(high,center+2*width/cosine)
            yhalf=max(yhalf,2*width)
    side=3.2*np.ceil(1.1*max(high-low,2*yhalf)/3.2)
    center=.1*np.round((low+high)/.2)
    half=side/2
    xmin,xmax=center-half,center+half
    # Display a short output segment even when the detector is at the grating.
    zlo,zhi=-max(20.,float(depth))-4.,36.
    span=3.2*np.ceil(max(zhi-zlo,side)/3.2)
    zcenter=(zlo+zhi)/2
    return dict(x=(xmin,xmax),y=(-half,half),side=side,center=center,
                scene_x=(center-span/2,center+span/2),scene_y=(-span/2,span/2),
                scene_z=(zcenter-span/2,zcenter+span/2),
                output_depth=max(20.,float(depth))+4.)


class GaussianOrderInteractiveView:
    """Use cached explorer data; preserve camera and traces between updates."""

    def __init__(self):
        figure = make_subplots(rows=2, cols=2,
            specs=[[dict(type='scene'), dict(type='xy')],
                   [dict(type='xy'), dict(type='xy')]],
            row_heights=[.67, .33], horizontal_spacing=.14, vertical_spacing=.15,
            subplot_titles=['Incident → grating → finite beams', 'XY detector · false colour',
                            'Detector profile at y = 0', 'Detector-line overlap'])
        self.roles = {}
        def add(role, trace, row=1, col=1):
            trace.meta = dict(role=role)
            figure.add_trace(trace, row=row, col=col)
            self.roles[role] = len(figure.data)-1
        for role, color in (('incident', '#17806b'), ('beam-m', '#2866a3'), ('beam-n', '#b85c17')):
            add(role, go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], color=color,
                opacity=.22 if role=='incident' else .16, showlegend=False,
                name=role+' · local 1/e² contour', hoverinfo='name',
                lighting=dict(ambient=1, diffuse=0)))
            add(role+'-arrow', go.Scatter3d(x=[], y=[], z=[], mode='lines',
                line=dict(color=color, width=6), showlegend=False, hoverinfo='skip'))
        add('grating', go.Surface(x=[-64,64], y=[-64,64], z=np.zeros((2,2)),
            colorscale=[[0,'#bac5d0'],[1,'#bac5d0']], showscale=False, opacity=.16,
            name='Grating reference plane', hoverinfo='name'))
        add('grating-lines', go.Scatter3d(x=[],y=[],z=[],mode='lines',
            line=dict(color='#73818f',width=1),showlegend=False,hoverinfo='skip'))
        add('incident-label', go.Scatter3d(x=[8],y=[0],z=[26],mode='text',
            text=['Incident'],textfont=dict(color='#17806b',size=11),showlegend=False,hoverinfo='skip'))
        add('detector', go.Surface(x=[-64,64],y=[-64,64],z=np.full((2,2),-40.),
            surfacecolor=np.zeros((2,2)),cmin=0,cmax=4,showscale=False,
            lighting=dict(ambient=1,diffuse=0,specular=0),
            hovertemplate='x=%{x:.1f} µm<br>y=%{y:.1f} µm<br>z=%{z:.1f} µm<br>I=%{surfacecolor:.3f}<extra></extra>'))
        add('border', go.Scatter3d(x=[-64,64,64,-64,-64],y=[-64,-64,64,64,-64],z=[-40]*5,
            mode='lines',line=dict(color='#7b858f',width=4),showlegend=False,hoverinfo='skip'))
        add('screen', go.Heatmap(x=[],y=[],z=[],zmin=0,zmax=4,zsmooth=False,
            colorbar=dict(title='Relative I',len=.46,y=.76,thickness=12),
            hovertemplate='x=%{x:.2f} µm<br>y=%{y:.2f} µm<br>I=%{z:.3f}<extra></extra>'), col=2)
        add('slice', go.Scatter(x=[-64,64],y=[0,0],mode='lines',
            line=dict(color='#9aa7b5',width=1,dash='dot'),showlegend=False,hoverinfo='skip'), col=2)
        add('nominal-ruler',go.Scatter(x=[],y=[],mode='lines+markers+text',
            line=dict(color='#63d9e6',width=2),marker=dict(symbol='line-ns',size=9),
            textfont=dict(color='#63d9e6',size=10),textposition='top center',
            showlegend=False,hoverinfo='text'),col=2)
        add('detector-local-ruler',go.Scatter(x=[],y=[],mode='lines+markers+text',
            line=dict(color='#e4a0ef',width=2),marker=dict(symbol='line-ns',size=9),
            textfont=dict(color='#e4a0ef',size=10),textposition='bottom center',
            showlegend=False,hoverinfo='text'),col=2)
        for role,color,name in [('profile-m','#2866a3','Iₘ'),('profile-n','#b85c17','Iₚ'),
                                ('profile-total','#243444','Combined'),('profile-cross','#79539e','Cross term')]:
            add(role,go.Scatter(x=[],y=[],mode='lines',name=name,line=dict(color=color,width=1.4)),row=2)
        add('overlap',go.Scatter(x=[],y=[],mode='lines',showlegend=False,
            line=dict(color='#17806b',width=2),hovertemplate='Depth=%{x:.0f} µm<br>O_line=%{y:.4f}<extra></extra>'),row=2,col=2)
        add('depth-marker',go.Scatter(x=[40,40],y=[0,1],mode='lines',
            line=dict(color='#299746',width=1),showlegend=False,hoverinfo='skip'),row=2,col=2)
        add('local-ruler',go.Scatter(x=[],y=[],mode='lines+markers+text',
            line=dict(color='#a13ab4',width=2),marker=dict(symbol='line-ns',size=9),
            textfont=dict(color='#a13ab4',size=10),textposition='top center',
            showlegend=False,hoverinfo='text'),row=2)
        figure.update_layout(template='plotly_white',autosize=True,height=750,
            margin=dict(l=45,r=30,t=45,b=65),uirevision='gaussian-orders-interactive',
            legend=dict(orientation='h',x=0,y=-.1,font=dict(size=10)),
            scene=dict(xaxis=dict(title='x (µm)',range=[-64,64]),
                yaxis=dict(title='y (µm)',range=[-64,64]),
                zaxis=dict(title='z (µm)',range=[-80,48]),aspectmode='cube',
                dragmode='orbit',uirevision='gaussian-orders-camera',camera=CAMERA))
        figure.update_xaxes(title_text='x (µm)',range=[-64,64],row=1,col=2)
        figure.update_yaxes(title_text='y (µm)',range=[-64,64],scaleanchor='x',
                           scaleratio=1,constrain='domain',row=1,col=2)
        figure.update_xaxes(title_text='x (µm)',range=[-64,64],row=2,col=1)
        figure.update_yaxes(title_text='Relative intensity',range=[-2.1,4.1],row=2,col=1)
        figure.update_xaxes(title_text='Depth s (µm)',range=[0,80],row=2,col=2)
        figure.update_yaxes(title_text='O_line (not coherence)',range=[0,1],row=2,col=2)
        self.figure = go.FigureWidget(figure)
        self.roles = {role:self.figure.data[index] for role,index in self.roles.items()}
        self._geometry_key = None
        spacing = make_subplots(rows=1,cols=2,horizontal_spacing=.14,
            subplot_titles=['Zoom: same envelopes, different phase models',
                            'Nominal versus local phase-gradient spacing'])
        for name,color,dash in [('Gaussian-field intensity','#243444','solid'),
                ('Same envelopes, nominal carrier','#b85c17','dash'),
                ('Envelope sum','#7d8791','dot')]:
            spacing.add_trace(go.Scatter(x=[],y=[],mode='lines',name=name,
                line=dict(color=color,width=1.5,dash=dash)),row=1,col=1)
        spacing.add_trace(go.Scatter(x=[],y=[],mode='lines',name='Local Gaussian Λ',
            line=dict(color='#a13ab4',width=2),
            hovertemplate='x=%{x:.2f} µm<br>Λ_local=%{y:.4f} µm<extra></extra>'),row=1,col=2)
        spacing.add_trace(go.Scatter(x=[],y=[],mode='lines',name='Nominal carrier Λ',
            line=dict(color='#17806b',width=1.5,dash='dash')),row=1,col=2)
        spacing.update_layout(template='plotly_white',height=340,autosize=True,
            margin=dict(l=45,r=20,t=45,b=85),uirevision='gaussian-spacing',
            legend=dict(orientation='h',y=-.28,font=dict(size=10)))
        spacing.update_xaxes(title_text='Detector x (µm)')
        spacing.update_yaxes(title_text='Relative intensity',row=1,col=1)
        spacing.update_yaxes(title_text='Λ (µm)',row=1,col=2)
        self.spacing_figure = go.FigureWidget(spacing)

    def _update_spacing(self, v):
        f = v.fringes
        center = f['center']
        if center is None:
            for role in ('nominal-ruler','local-ruler','detector-local-ruler'):
                self.roles[role].visible=False
            for trace in self.spacing_figure.data:
                trace.update(x=[],y=[])
            v.fringe_panel.set_title(0,'Fringe spacing — insufficient appreciable overlap at this detector')
            return
        xc = v.x[center]
        for role,length,height,label in (
                ('nominal-ruler',f['nominal'],.65*v.view_bounds['side']/2,'Plane-wave reference:'),
                ('local-ruler',f['local'],3.1,'Λ local ≈'),
                ('detector-local-ruler',f['local'],-.12*v.view_bounds['side'],'Gaussian local estimate: ≈')):
            self.roles[role].update(x=[xc-length/2,xc,xc+length/2],y=[height]*3,
                text=['',f'{label} {length:.2f} µm',''],visible=True)
        row = int(round(v.detector.value))
        half_width = max(3*f['nominal'],2*v.waist.value)
        lo,hi=max(-64,xc-half_width),min(64,xc+half_width)
        select=(v.x>=lo)&(v.x<=hi)
        phase=f['phase'][center]+np.deg2rad(v.phase.value)
        carrier=f['envelope']+f['cross_amplitude']*np.cos(
            2*np.pi*(v.x-xc)/f['nominal']+phase)
        with self.spacing_figure.batch_update():
            for trace,values in zip(self.spacing_figure.data[:3],
                    (v.combined[row],carrier,f['envelope'])):
                trace.update(x=v.x[select],y=values[select])
            self.spacing_figure.data[3].update(x=v.x[select],y=f['spacing'][select])
            self.spacing_figure.data[4].update(x=[lo,hi],y=[f['nominal']]*2)
            self.spacing_figure.update_xaxes(range=[lo,hi])
        v.fringe_panel.set_title(0,f'Fringe spacing — nominal {f["nominal"]:.2f} µm; '
                                 f'local {f["local"]:.2f} µm near strongest overlap')

    @staticmethod
    def _mesh(trace, faces):
        vertices, triangles = [], []
        for polygon in faces:
            offset = len(vertices)
            vertices.extend(polygon)
            triangles.extend((offset,offset+i,offset+i+1) for i in range(1,len(polygon)-1))
        points = np.asarray(vertices).reshape(-1,3)
        indices = np.asarray(triangles,dtype=int).reshape(-1,3)
        trace.update(x=points[:,0],y=points[:,1],z=points[:,2],
                     i=indices[:,0],j=indices[:,1],k=indices[:,2])

    @staticmethod
    def _segments(trace, segments):
        values = []
        for segment in segments:
            values.extend(segment)
            values.append([None,None,None])
        xyz = np.asarray(values,dtype=object)
        trace.update(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2])

    def update(self, explorer, shell_function, color_map):
        """Read the existing cached fields, without changing scene camera."""
        v = explorer
        bounds=v.view_bounds
        key = (v._key,v.detector.value)
        with self.figure.batch_update():
            if key != self._geometry_key:
                self._mesh(self.roles['incident'],shell_function(v.model,0,z_limits=(0.,32.)))
                self._segments(self.roles['incident-arrow'],v.incident_arrow._segments3d)
                for suffix,order,arrow in zip(('m','n'),v.pair.value,v.direction_arrows):
                    self._mesh(self.roles['beam-'+suffix],shell_function(v.model,order,
                               z_limits=(-bounds['output_depth'],0.)))
                    self.roles['beam-'+suffix].name=f'Order {order:+d} · local 1/e² contour'
                    transverse,direction=v.model.basis(order)
                    length=min(30.,.7*bounds['output_depth']/(-direction[2]))
                    start,tip=.35*length*direction,length*direction
                    head=.25*length
                    self._segments(self.roles['beam-'+suffix+'-arrow'],[
                        [start,tip],[tip,tip-head*direction+.4*head*transverse],
                        [tip,tip-head*direction-.4*head*transverse]])
                lo,hi=bounds['x'];bottom,top=bounds['y']
                positions=np.arange(np.ceil(lo/v.pitch.value),np.floor(hi/v.pitch.value)+1)*v.pitch.value
                self._segments(self.roles['grating-lines'],
                    [[(x,bottom,0),(x,top,0)] for x in positions])
                self.roles['grating'].update(x=[lo,hi],y=[bottom,top])
                self.roles['border'].update(x=[lo,hi,hi,lo,lo],y=[bottom,bottom,top,top,bottom])
                self.roles['slice'].x=[lo,hi]
                self.figure.layout.scene.xaxis.range=bounds['scene_x']
                self.figure.layout.scene.yaxis.range=bounds['scene_y']
                self.figure.layout.scene.zaxis.range=bounds['scene_z']
                self.figure.update_xaxes(range=[lo,hi],row=1,col=2)
                self.figure.update_yaxes(range=[bottom,top],row=1,col=2)
                self.figure.update_xaxes(range=[lo,hi],row=2,col=1)
                self._geometry_key = key
            scale = [[float(t),'rgb('+','.join(str(int(round(255*c))) for c in color_map(t)[:3])+')']
                     for t in np.linspace(0,1,65)]
            sy,sx = v._display_stride
            dx=(v.detector_x[::sx][:-1]+v.detector_x[::sx][1:])/2
            dy=(v.detector_y[::sy][:-1]+v.detector_y[::sy][1:])/2
            self.roles['detector'].update(x=dx,y=dy,
                z=np.full(v.detector_display_intensity.shape,-v.detector.value),
                surfacecolor=v.detector_display_intensity,colorscale=scale)
            self.roles['border'].z=[-v.detector.value]*5
            self.roles['screen'].update(x=v.detector_x,y=v.detector_y,
                                        z=v.detector_intensity,colorscale=scale)
            row=int(round(v.detector.value))
            for role,array in zip(('profile-m','profile-n','profile-total','profile-cross'),
                    (*v.data['intensities'],v.combined,v.cross)):
                self.roles[role].update(x=v.x,y=array[row])
            self.roles['overlap'].update(x=v.depths,y=v.data['overlap'])
            self.roles['depth-marker'].x=[v.detector.value]*2
            self.figure.layout.annotations[1].text=f'XY detector at z={-v.detector.value:g} µm · false colour'
            self._update_spacing(v)

    def reset_camera(self):
        self.figure.layout.scene.camera = CAMERA
