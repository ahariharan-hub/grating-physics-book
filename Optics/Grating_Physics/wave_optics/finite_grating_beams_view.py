"""One owned, retained-artist explorer for Notebook 7.3A."""
from itertools import combinations
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .finite_grating_beams import FiniteGratingConfiguration,FiniteGratingExperiment
from .grating_intro import _close_widget_tree
from .plotting import figure_png


class FiniteGratingBeamsExplorer:
    def __init__(self):
        import ipywidgets as w
        self._closed=self._busy=False
        self.model_updates=self.updates=0
        self._model_key=self._pair_key=None
        def slider(label,value,lo,hi,step):
            return w.FloatSlider(description=label,value=value,min=lo,max=hi,step=step,
                continuous_update=False,style={'description_width':'initial'},layout=w.Layout(width='300px'))
        self.wavelength=slider('λ in medium (µm)',1.,.6,1.2,.1)
        self.pitch=slider('Grating pitch d (µm)',4.,2.,6.,.5)
        self.radius=slider('Envelope radius w₀ (µm)',8.,2.,16.,1.)
        self.angle=slider('Incidence θᵢ (°)',0.,-15.,15.,5.)
        self.detector=slider('Depth s = −zᴅ (µm)',60.,0.,80.,1.)
        self.pair=w.Dropdown(description='Orders (m,p)',options=[],style={'description_width':'initial'})
        self.reset_button=w.Button(description='Reset',icon='undo')
        self.picture=w.Image(format='png',layout=w.Layout(width='100%'))
        self.readout=w.HTML()
        self._inputs=(self.wavelength,self.pitch,self.radius,self.angle,self.detector,self.pair)
        self.figure=Figure(figsize=(14.,11.),dpi=105,layout='constrained')
        FigureCanvasAgg(self.figure)
        self.axes=self.figure.subplots(3,3)
        boundary,geometry,spectrum=self.axes[0]
        boundary.set(title='A: same grating, ideal / finite input',xlabel='x (µm)',ylabel='Boundary intensity',xlim=(-32,32),ylim=(-.12,1.15))
        boundary.axhline(1.,color='#727c86',ls='--',label='Ideal input: indefinite extent')
        self.input_line,=boundary.plot([],[],color='#17806b',label='Finite Gaussian input')
        self.mask_line,=boundary.plot([],[],color='#243444',alpha=.45,label='Binary mask t(x)')
        boundary.legend(fontsize=7,loc='upper right')
        geometry.set(title='A: incident envelope + full output',xlabel='x (µm)',ylabel='z (µm)',xlim=(-48,48),ylim=(-80,16))
        self.full_image=geometry.imshow(np.zeros((81,2)),extent=(-48,48,-80,0),origin='upper',aspect='auto',cmap='gray',vmin=0,vmax=1)
        self.incident_image=geometry.imshow(np.zeros((21,2)),extent=(-48,48,0,16),origin='lower',aspect='auto',cmap='Greens',vmin=0,vmax=1,alpha=.6)
        geometry.axhline(0,color='#243444',lw=1.5)
        self.grating,=geometry.plot([],[],color='#243444',lw=2)
        self.directions=[geometry.plot([],[],color=c,lw=1.3,ls='--')[0] for c in ('#2866a3','#b85c17')]
        spectrum.set(title='B: spectrum and selected windows',xlabel='kₓ (rad/µm)',ylabel='|spectrum| / max',ylim=(0,1.08))
        self.spectrum_line,=spectrum.plot([],[],color='#243444')
        self.window_lines=[spectrum.plot([],[],color=c,lw=2)[0] for c in ('#2866a3','#b85c17')]
        from matplotlib.patches import Rectangle
        self.window_patches=[]
        for colour in ('#2866a3','#b85c17'):
            patch=Rectangle((0,0),1,1,transform=spectrum.get_xaxis_transform(),
                facecolor=colour,edgecolor=colour,alpha=.12,lw=1)
            spectrum.add_patch(patch);self.window_patches.append(patch)
        from matplotlib.collections import LineCollection
        self.centers=LineCollection([],colors='#727c86',linestyles=':',linewidths=.8)
        spectrum.add_collection(self.centers)
        self.images=[]
        names=('C: selected order m intensity','C: selected order n intensity','C: coherent selected sum','D: signed cross term')
        for axis,name,cmap,signed in zip((self.axes[1,0],self.axes[1,1],self.axes[1,2],self.axes[2,0]),names,('Blues','Oranges','gray','RdBu_r'),(False,False,False,True)):
            axis.set(title=name,xlabel='x (µm)',ylabel='z (µm)',xlim=(-48,48),ylim=(-80,0))
            image=axis.imshow(np.zeros((81,2)),extent=(-48,48,-80,0),origin='upper',aspect='auto',cmap=cmap,vmin=-.5 if signed else 0,vmax=.5 if signed else 1.)
            self.images.append(image)
        self.figure.colorbar(self.images[2],ax=list(self.axes[1]),shrink=.7,label='Relative scalar intensity; common 0…1 scale')
        self.figure.colorbar(self.images[3],ax=self.axes[2,0],shrink=.7,label='2 Re(Uₘ* Uₚ)')
        self.detector_lines=[a.axhline(-60,color='#299746',lw=1.) for a in (geometry,*self.axes[1],self.axes[2,0])]
        profile=self.axes[2,1]
        profile.set(title='D: detector-plane profiles',xlabel='x (µm)',ylabel='Relative scalar intensity',xlim=(-48,48),ylim=(-.52,1.05))
        self.profiles=[profile.plot([],[],color=c,label=label,lw=1.2)[0] for c,label in (
            ('#2866a3','Iₘ'),('#b85c17','Iₚ'),('#243444','|Uₘ+Uₚ|²'),('#79539e','Cross term'))]
        profile.legend(fontsize=7)
        overlap=self.axes[2,2]
        overlap.set(title='D: spatial intensity overlap',xlabel='Depth s = −z (µm)',ylabel='O(s), not |γₘₚ|',xlim=(0,80),ylim=(-.02,1.02))
        self.overlap_line,=overlap.plot([],[],color='#17806b')
        self.overlap_marker=overlap.axvline(60,color='#299746',lw=1.)
        for a in self.axes.flat:a.tick_params(labelsize=8);a.title.set_fontsize(9);a.grid(alpha=.10)
        self.ui=w.VBox([w.HBox([self.wavelength,self.pitch,self.radius],layout=w.Layout(flex_flow='row wrap')),
            w.HBox([self.angle,self.pair,self.detector,self.reset_button],layout=w.Layout(flex_flow='row wrap')),
            self.readout,self.picture])
        for control in self._inputs:control.observe(self._changed,names='value')
        self.reset_button.on_click(self.reset)
        self._update()

    def _changed(self,change):
        if not self._busy and not self._closed:self._update()

    def _update(self):
        if self._closed:return
        self.updates+=1;self._busy=True
        try:
            key=(self.wavelength.value,self.pitch.value,self.radius.value,self.angle.value)
            if key!=self._model_key:
                self.model=FiniteGratingExperiment(FiniteGratingConfiguration(*key[:3],angle_deg=key[3]))
                self._model_key=key;self.model_updates+=1
                options=[(f'({m:+d}, {n:+d})',(int(m),int(n))) for m,n in combinations(self.model.orders,2)]
                old=self.pair.value
                self.pair.options=options
                legal=[x[1] for x in options]
                self.pair.value=old if old in legal else ((0,1) if (0,1) in legal else (legal[0] if legal else None))
                self.pair.disabled=not legal
            if self.pair.value is None:
                self.readout.value='No two nonzero propagating order centers. Change wavelength or pitch.'
                return
            pairkey=(key,self.pair.value)
            if pairkey!=self._pair_key:
                self.data=self.model.selected_fields(self.pair.value)
                self._pair_key=pairkey
            m=self.model;data=self.data
            show=abs(m.x)<=48
            x=m.x[show]
            self.input_line.set_data(m.x,abs(m.incident)**2)
            self.mask_line.set_data(m.x,m.mask.real)
            image_arrays=(abs(data['fields'][0])**2,abs(data['fields'][1])**2,abs(data['total'])**2,data['cross'])
            for artist,array in zip(self.images,image_arrays):artist.set_data(array[:,show])
            self.full_image.set_data(abs(data['full'][:,show])**2)
            above=np.linspace(0,16,21)
            gaussian=np.exp(-2*((x[None,:]+np.tan(np.deg2rad(self.angle.value))*above[:,None])/self.radius.value)**2)
            self.incident_image.set_data(gaussian)
            # The incident extension is a geometric envelope illustration;
            # only its boundary at z=0 is prescribed and propagated.
            positions=np.arange(-48,48.01,self.pitch.value)
            self.grating.set_data(np.ravel(np.column_stack((positions,positions,np.full_like(positions,np.nan)))),
                np.ravel(np.tile([-.5,.5,np.nan],(len(positions),1))))
            for line,order in zip(self.directions,data['pair']):
                k_x=m.k_x_i+order*m.G;k_z=-np.sqrt(m.k*m.k-k_x*k_x)
                line.set_data([0,-20*k_x/k_z],[0,-20])
            sorted_indices=np.argsort(m.k_x)
            amplitude=abs(m.spectrum)/max(abs(m.spectrum).max(),1e-300)
            self.spectrum_line.set_data(m.k_x[sorted_indices],amplitude[sorted_indices])
            for line,window in zip(self.window_lines,data['windows']):
                line.set_data(m.k_x[sorted_indices],np.where(window[sorted_indices],amplitude[sorted_indices],np.nan))
            for patch,order in zip(self.window_patches,data['pair']):
                center=m.k_x_i+order*m.G
                left=max(center-m.G/2,-m.k);right=min(center+m.G/2,m.k)
                patch.set_x(left);patch.set_width(right-left)
            self.centers.set_segments([[(c,0),(c,1.02)] for c in m.centers])
            self.axes[0,2].set_xlim(-m.k*1.1,m.k*1.1)
            row=int(round(self.detector.value))
            self.detector_fields=data['fields'][:,row]
            profiles=(image_arrays[0][row],image_arrays[1][row],image_arrays[2][row],data['cross'][row])
            for line,values in zip(self.profiles,profiles):line.set_data(m.x,values)
            self.overlap_line.set_data(data['depth'],data['overlap'])
            self.overlap_marker.set_xdata([self.detector.value]*2)
            for line in self.detector_lines:line.set_ydata([-self.detector.value]*2)
            self.axes[1,0].set_title(f'C: order m={data["pair"][0]:+d} intensity')
            self.axes[1,1].set_title(f'C: order p={data["pair"][1]:+d} intensity')
            diag=data['diagnostics']
            note='<br>'.join(data['warnings']) or 'Spectral replicas well separated by the Gaussian midpoint-tail diagnostic; check window/padding convergence for quantitative use.'
            self.readout.value=(f'<b>O(zᴅ) = {data["overlap"][row]:.3f}</b> (intensity overlap, not coherence). '
                f'FFT width {diag["fft_width"]:.0f} µm; dx={m.dx:.4f} µm; Δkₓ={diag["delta_k_x"]:.4f} rad/µm. '
                f'Outer-edge intensity fraction ≤ {diag["edge_fraction"]:.2e}.<br>'
                f'<span style="color:{"#a34c00" if data["warnings"] else "#526575"}">{note}</span>')
            self.picture.value=figure_png(self.figure)
        finally:self._busy=False

    def reset(self,button=None):
        if self._closed:return
        self._busy=True
        try:
            for control,value in ((self.wavelength,1.),(self.pitch,4.),(self.radius,8.),(self.angle,0.),(self.detector,60.)):
                control.value=value
            self._model_key=None
            self.pair.value=(0,1) if (0,1) in [v for _,v in self.pair.options] else self.pair.value
        finally:self._busy=False
        self._update()

    def close(self):
        if self._closed:return
        self._closed=True
        for control in self._inputs:control.unobserve(self._changed,names='value')
        self.reset_button.on_click(self.reset,remove=True)
        _close_widget_tree(self.ui)
        self.picture.value=b'';self.figure.clear();self.data=None
