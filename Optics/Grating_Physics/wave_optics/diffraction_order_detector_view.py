"""Linked 3D geometry and XY detector for the prescribed two-order lesson."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .conical_plotting import _arrow, _grating, _trace_roles
from .plotting import geometry_line3d
from .superposition import two_wave_field


def clipped_order_crests(vector, coefficient_phase, xlim=(-8.,8.), zlim=(-8.,0.), ylim=(-8.,8.)):
    """Intersect constant-phase planes with all twelve edges of the display box."""
    from itertools import product, combinations
    vector = np.asarray(vector, dtype=float)
    normal = vector/np.linalg.norm(vector)
    corners = np.array(list(product(xlim, ylim, zlim)), dtype=float)
    edges = [(a,b) for a,b in combinations(range(8),2)
             if np.count_nonzero(corners[a] != corners[b]) == 1]
    phases = corners@vector+coefficient_phase
    tangent = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
    tangent /= np.linalg.norm(tangent)
    other = np.cross(normal, tangent)
    polygons = []
    for j in range(int(np.ceil(phases.min()/(2*np.pi)-1e-12)),
                   int(np.floor(phases.max()/(2*np.pi)+1e-12))+1):
        points = []
        for a,b in edges:
            da,db = phases[a]-2*np.pi*j,phases[b]-2*np.pi*j
            if abs(da)<1e-10: points.append(corners[a])
            if abs(db)<1e-10: points.append(corners[b])
            if da*db<0: points.append(corners[a]+da/(da-db)*(corners[b]-corners[a]))
        if not points: continue
        points = np.unique(np.round(points,12),axis=0)
        if len(points)<3: continue
        centred = points-points.mean(axis=0)
        angles = np.arctan2(centred@other,centred@tangent)
        polygon = points[np.argsort(angles)]
        if np.linalg.norm(np.cross(polygon[1]-polygon[0],polygon[2]-polygon[0]))<1e-12:
            continue
        polygons.append(polygon)
    return polygons


class TwoOrderDetectorView:
    """Retain Plotly traces and the camera while the shared controls change."""

    def __init__(self):
        self.x = np.linspace(-8, 8, 401)
        self.y = np.linspace(-8, 8, 401)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        figure = make_subplots(rows=2, cols=2,
            specs=[[{'type':'scene'}, {'type':'scene'}], [{'type':'xy','colspan':2},None]],
            column_widths=[.50,.50],row_heights=[.53,.47],
            horizontal_spacing=.10,vertical_spacing=.14,
            subplot_titles=['3D geometry and XY detector',
                '3D wavevector algebra: Δk = kₚ − kₘ', 'Face-on XY detector'])
        _grating(figure, {'period':2.}, extent=8.)
        _trace_roles(figure)['grating-bars'].opacity = .25
        # Direction arrows are schematic reference segments, never beam edges.
        for role, colour, name in (
                ('incident', '#243444', 'Incident direction'),
                ('order-m', '#2866a3', 'Order m direction'),
                ('order-n', '#b85c17', 'Order n direction')):
            _arrow(figure, np.zeros(3), np.array([0., 0., -2.]), colour,
                name, role=role)
        for suffix,colour in (('m','#2866a3'),('n','#b85c17')):
            figure.add_trace(go.Mesh3d(x=[],y=[],z=[],i=[],j=[],k=[],color=colour,
                opacity=.055,flatshading=True,lighting=dict(ambient=1,diffuse=0),
                showlegend=False,hoverinfo='skip',meta={'role':'crests-'+suffix}),row=1,col=1)
            figure.add_trace(go.Scatter3d(x=[],y=[],z=[],mode='lines',
                line=dict(color=colour,width=2),showlegend=True,
                name='Order '+suffix+' wavefronts (clipped infinite planes)',
                hoverinfo='name',meta={'role':'crest-edges-'+suffix}),row=1,col=1)
        for suffix,colour in (('m','#2866a3'),('n','#b85c17')):
            figure.add_trace(go.Scatter3d(x=[],y=[],z=[],mode='lines',
                line=dict(color=colour,width=5),showlegend=False,
                name='Right angle: wavevector normal to its wavefront',
                hoverinfo='name',meta={'role':'normal-cue-'+suffix}),row=1,col=1)
        figure.add_trace(go.Surface(x=self.xx, y=self.yy,
            z=np.full_like(self.xx, -3.), surfacecolor=np.zeros_like(self.xx),
            cmin=0, cmax=9, colorscale='Gray', showscale=False,
            lighting=dict(ambient=1, diffuse=0, specular=0),
            name='XY detector intensity', meta={'role':'detector'},
            hovertemplate='x=%{x:.2f} µm<br>y=%{y:.2f} µm<br>z=%{z:.2f} µm'
                          '<br>I=%{surfacecolor:.3f}<extra></extra>'), row=1, col=1)
        border = geometry_line3d(np.array([[-8,8,8,-8,-8],[-8,-8,8,8,-8],[-3]*5]),
            '#299746', 4, name='XY detector plane')
        border.update(meta={'role':'border'}, showlegend=True)
        figure.add_trace(border, row=1, col=1)
        figure.add_trace(go.Heatmap(x=self.x, y=self.y, z=np.zeros_like(self.xx),
            zmin=0, zmax=9, colorscale='Gray',
            colorbar=dict(title='Relative<br>intensity', thickness=12, len=.38, y=.22),
            meta={'role':'screen'},
            hovertemplate='x=%{x:.2f} µm<br>y=%{y:.2f} µm<br>I=%{z:.3f}<extra></extra>'),
            row=2, col=1)
        figure.add_trace(go.Scatter(x=[-8,8], y=[0,0], mode='lines',
            line=dict(color='#299746',width=1,dash='dot'), showlegend=False,
            name='Profile at y = 0', meta={'role':'profile-slice'}), row=2,col=1)
        figure.add_trace(go.Scatter(x=[], y=[], mode='lines+markers+text',
            line=dict(color='#b546d0',width=3), marker=dict(symbol='line-ns',size=12),
            textposition='bottom center', textfont=dict(color='#b546d0',size=12),
            showlegend=False, hoverinfo='skip', meta={'role':'ruler'}),row=2,col=1)
        for role,colour,name in (
                ('ky-m','#2866a3','kᵧ,ₘ'),('ky-n','#b85c17','kᵧ,ₚ'),
                ('delta-y','#79539e','Δkᵧ'),
                ('k-m','#2866a3','kₘ'),('kx-m','#2866a3','kₓ,ₘ'),('kz-m','#2866a3','k_z,ₘ'),
                ('k-n','#b85c17','kₚ'),('kx-n','#b85c17','kₓ,ₚ'),('kz-n','#b85c17','k_z,ₚ'),
                ('delta','#79539e','Δk: m → p'),('delta-x','#79539e','Δkₓ'),
                ('delta-z','#79539e','Δk_z')):
            _arrow(figure,np.zeros(3),np.array([1.,0.,-1.]),colour,name,
                role=role,row=1,col=2,legend=False,width=5 if role in ('k-m','k-n','delta') else 3)
            figure.add_trace(go.Scatter3d(x=[],y=[],z=[],mode='text',
                textfont=dict(color=colour,size=12),showlegend=False,hoverinfo='skip',
                meta={'role':role+'-label'}),row=1,col=2)
        figure.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode='markers',
            marker=dict(size=3,color='#243444'),showlegend=False,hoverinfo='skip'),row=1,col=2)
        figure.update_layout(template='plotly_white', width=950, height=950,
            margin=dict(l=15,r=35,t=60,b=125), uirevision='two-order-detector',
            legend=dict(orientation='h',y=-.12,font=dict(size=10)),
            scene=dict(xaxis=dict(title='x (µm)',range=[-8.5,8.5]),
                yaxis=dict(title='y (µm)',range=[-8.5,8.5]),
                zaxis=dict(title='z (µm)',range=[-8.5,2.5]),
                aspectmode='data',dragmode='orbit',uirevision='two-order-detector',
                camera=dict(eye=dict(x=.42,y=-3.0,z=.78),projection=dict(type='orthographic'))))
        figure.update_layout(scene2=dict(
            xaxis=dict(title='kₓ (rad/µm)'),yaxis=dict(title='kᵧ (rad/µm)'),
            zaxis=dict(title='k_z (rad/µm)'),aspectmode='manual',
            aspectratio=dict(x=1.3,y=1.3,z=1.),dragmode='orbit',
            uirevision='two-order-algebra',camera=dict(eye=dict(x=.4,y=-2.5,z=.5))))
        figure.update_xaxes(title_text='Detector x (µm)',range=[-8,8],row=2,col=1)
        figure.update_yaxes(title_text='Detector y (µm)',range=[-8,8],
            scaleanchor='x',scaleratio=1,constrain='domain',row=2,col=1)
        self.figure = go.FigureWidget(figure)
        self.roles = _trace_roles(self.figure)
        self._pitch = 2.
        self._crest_key = None

    def update(self, vectors, wavelength, pitch, angle, pair, ratio, phase, detector, azimuth=0.):
        """Use the same scalar fields and coefficient convention as the lesson equations."""
        with self.figure.batch_update():
            self._update_algebra(vectors)
            self._update_wavefronts(vectors,pair,ratio,phase)
            if pitch != self._pitch:
                from .conical_plotting import _grating_bars
                bars = _grating_bars({'period':pitch}, extent=8.)
                old = self.roles['grating-bars']
                old.update(x=bars.x,y=bars.y,z=bars.z,i=bars.i,j=bars.j,k=bars.k)
                self._pitch = pitch
            valid = vectors is not None
            for suffix in ('m','n'):self.roles['normal-cue-'+suffix].visible=valid and (suffix=='m' or ratio>0)
            for role in ('incident','order-m','order-n'):
                self.roles[role].visible = valid
                self.roles[role+'-head'].visible = valid
            if valid:
                k = 2*np.pi/wavelength
                theta,phi = np.deg2rad([angle,azimuth])
                incoming = np.array([np.sin(theta)*np.cos(phi),
                    np.sin(theta)*np.sin(phi),-np.cos(theta)])
                _arrow(self.figure,-2*incoming,2*incoming,'#243444','Incident direction',
                    role='incident',update=True)
                for vector, order, role, colour in zip(vectors,pair,
                        ('order-m','order-n'),('#2866a3','#b85c17')):
                    # A fixed reference length prevents grazing directions from
                    # expanding the plot; plane waves fill the whole detector.
                    direction=vector/k
                    coefficient_phase=0. if role=='order-m' else np.deg2rad(phase)
                    # First strictly outgoing crest on the normal through origin.
                    crest_index=int(np.floor(coefficient_phase/(2*np.pi)))+1
                    distance=(2*np.pi*crest_index-coefficient_phase)/k
                    anchor=distance*direction
                    tangent=np.array([-direction[2],0.,direction[0]])
                    tangent/=np.linalg.norm(tangent)
                    size=.35
                    # The tangent lies in the wavefront; the square corner spans
                    # the actual normal and tangent, never screen coordinates.
                    cue=np.array([anchor+size*direction,
                        anchor+size*(direction+tangent),anchor+size*tangent,
                        anchor,anchor+.85*tangent])
                    suffix=role[-1]
                    self.roles['normal-cue-'+suffix].update(
                        x=cue[:,0],y=cue[:,1],z=cue[:,2])
                    _arrow(self.figure,np.zeros(3),(distance+.85)*direction,colour,
                        f'Order {order:+d}: k perpendicular to wavefront (not beam edges)',
                        role=role,update=True,head_size=.34,width=6)
                points = np.stack((self.xx,self.yy,np.full_like(self.xx,detector)),axis=-1)
                intensity = two_wave_field(points,vectors,amplitude2=ratio,
                    phase_difference_deg=phase)['relative_intensity']
            else:
                intensity = np.full_like(self.xx,np.nan)
            self.intensity = intensity
            self.roles['detector'].update(z=np.full_like(self.xx,detector),surfacecolor=intensity)
            self.roles['border'].update(z=[detector]*5)
            self.roles['screen'].z = intensity
            self.figure.layout.annotations[2].text = (
                f'XY detector at z = {detector:.2f} µm' if valid else 'No propagating pair')
            ruler = self.roles['ruler']
            if valid and ratio > 0:
                delta = vectors[1]-vectors[0]
                period = 2*np.pi/abs(delta[0])
                root = -(delta[1]*(-3)+delta[2]*detector+np.deg2rad(phase))/delta[0]
                left = root+np.ceil((-7.4-root)/period)*period
                ruler.update(x=[left,left+period/2,left+period],y=[-3]*3,
                    text=['',f'Λₓ = {period:.3f} µm',''],visible=True)
            else:
                ruler.visible = False

    def _update_wavefronts(self,vectors,pair,ratio,phase):
        key=None if vectors is None else (tuple(vectors.ravel()),tuple(pair),float(phase),ratio>0)
        if key==self._crest_key:return
        self._crest_key=key
        for index,suffix in enumerate(('m','n')):
            mesh=self.roles['crests-'+suffix]
            edges=self.roles['crest-edges-'+suffix]
            visible=vectors is not None and (index==0 or ratio>0)
            mesh.visible=edges.visible=visible
            if not visible:
                mesh.update(x=[],y=[],z=[],i=[],j=[],k=[])
                edges.update(x=[],y=[],z=[])
                continue
            coefficient_phase=0. if index==0 else np.deg2rad(phase)
            polygons=clipped_order_crests(vectors[index],coefficient_phase)
            vertices=np.concatenate(polygons) if polygons else np.empty((0,3))
            triangles=[]
            offset=0
            for polygon in polygons:
                triangles.extend((offset,offset+j,offset+j+1) for j in range(1,len(polygon)-1))
                offset+=len(polygon)
            triangles=np.asarray(triangles,dtype=int).reshape(-1,3)
            mesh.update(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],
                i=triangles[:,0],j=triangles[:,1],k=triangles[:,2])
            lines=np.concatenate([np.vstack([p,p[:1],np.full((1,3),np.nan)]) for p in polygons]) if polygons else np.empty((0,3))
            edges.update(x=lines[:,0],y=lines[:,1],z=lines[:,2],
                name=f'Order {pair[index]:+d}: crests (clipped infinite planes)')

    def _update_algebra(self,vectors):
        roles=('k-m','kx-m','ky-m','kz-m','k-n','kx-n','ky-n','kz-n','delta','delta-x','delta-y','delta-z')
        if vectors is None:
            for role in roles:
                for suffix in ('','-head','-label'):self.roles[role+suffix].visible=False
            return
        m,n=vectors
        k=np.linalg.norm(m)
        origin=np.zeros(3)
        paths={}
        for vector,suffix,colour in ((m,'m','#2866a3'),(n,'n','#b85c17')):
            corner=np.array([vector[0],0.,0.])
            paths['k-'+suffix]=(origin,vector,colour)
            paths['kx-'+suffix]=(origin,corner,colour)
            xy=np.array([vector[0],vector[1],0.])
            paths['ky-'+suffix]=(corner,xy,colour)
            paths['kz-'+suffix]=(xy,vector,colour)
        corner=np.array([n[0],m[1],m[2]])
        xy=np.array([n[0],n[1],m[2]])
        paths.update({'delta':(m,n,'#79539e'),
            'delta-x':(m,corner,'#79539e'),'delta-y':(corner,xy,'#79539e'),
            'delta-z':(xy,n,'#79539e')})
        names={'k-m':'kₘ','k-n':'kₚ','kx-m':'kₓ,ₘ','kx-n':'kₓ,ₚ',
            'kz-m':'k_z,ₘ','kz-n':'k_z,ₚ','delta':'Δk (m → p)',
            'delta-x':'Δkₓ','delta-z':'Δk_z'}
        names.update({'ky-m':'kᵧ,ₘ','ky-n':'kᵧ,ₚ','delta-y':'Δkᵧ'})
        for role,(start,end,colour) in paths.items():
            vector=end-start
            visible=bool(np.linalg.norm(vector)>1e-12*k)
            _arrow(self.figure,start,vector,colour,names[role],role=role,
                update=True,head_size=.08*k,width=5 if role in ('k-m','k-n','delta') else 3)
            component=role not in ('k-m','k-n','delta')
            self.roles[role].update(visible=visible,line=dict(dash='dash' if component else 'solid'))
            self.roles[role+'-head'].visible=visible
            # Label the exact endpoint vector in hover; keep labels spatially separated.
            offsets={'k-m':(-.13,-.12,-.04),'k-n':(.13,-.12,-.04),
                'kx-m':(0.,0.,.14),'kx-n':(0.,0.,.14),
                'kz-m':(-.18,0.,0.),'kz-n':(.18,0.,0.),
                'delta':(0.,-.22,-.22),'delta-x':(0.,.08,-.12),
                'delta-z':(.22,.08,0.),'ky-m':(-.1,0.,.14),'ky-n':(.1,0.,.14),'delta-y':(.1,0.,.14)}
            point=(start+end)/2+k*np.array(offsets[role])
            label=names[role]
            label_visible=visible
            for suffix,vector in (('m',m),('n',n)):
                if np.linalg.norm(vector[:2])<1e-12*k:
                    if role=='kz-'+suffix:label_visible=False
                    if role=='k-'+suffix:label+=' = '+names['kz-'+suffix]
            self.roles[role+'-label'].update(x=[point[0]],y=[point[1]],z=[point[2]],
                text=[label],visible=label_visible)
            vector=end-start
            self.roles[role].hovertemplate=(names[role]+f'<br>({vector[0]:+.3f}, '
                f'{vector[1]:+.3f}, {vector[2]:+.3f}) rad/µm<extra></extra>')
        self.figure.layout.scene2.xaxis.range=[-1.3*k,1.3*k]
        self.figure.layout.scene2.yaxis.range=[-1.3*k,1.3*k]
        self.figure.layout.scene2.zaxis.range=[-1.3*k,.3*k]
        self.figure.layout.annotations[1].text=(
            f'3D wavevectors: Δk = kₚ − kₘ; Δkᵧ = {n[1]-m[1]:+.3f}'+
            ('; Δk_z = 0' if abs(m[2]-n[2])<1e-12*k else ''))

    def close(self):
        self.figure.close()
