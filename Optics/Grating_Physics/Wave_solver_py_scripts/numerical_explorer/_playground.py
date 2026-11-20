"""Shared presentation utilities for standalone experiments; no file exports."""
from pathlib import Path
import sys
import numpy as np

COURSE = next((base for base in Path(__file__).resolve().parents
               if (base/'wave_optics/scalar_grating.py').is_file()), None)
if COURSE is None:
    raise FileNotFoundError('Keep numerical_explorer inside the extracted Grating_Physics folder, beside wave_optics.')
if str(COURSE) not in sys.path:
    sys.path.insert(0,str(COURSE))
from wave_optics.scalar_propagation import transfer_function

BLUE='#0072B2'
ORANGE='#D55E00'
GREEN='#009E73'
_instances={}


def positive(value,name,allow_zero=False):
    value=float(value)
    if not np.isfinite(value) or (value<0 if allow_zero else value<=0):
        raise ValueError(name+' must be finite and '+('nonnegative' if allow_zero else 'positive'))
    return value


def integer(value,name,minimum=1):
    if isinstance(value,(bool,np.bool_)) or not np.isfinite(value) or int(value)!=value or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def phase(values,floor=1e-6,break_wraps=False):
    values=np.asarray(values)
    result=np.angle(values)
    result=np.where(np.isclose(abs(result),np.pi,atol=1e-12,rtol=0),np.pi,result)
    result=np.where(np.isclose(result,0.,atol=1e-12,rtol=0),0.,result)
    result=np.where(abs(values)>floor,result,np.nan)
    if break_wraps:
        result[np.flatnonzero(abs(np.diff(result))>np.pi)+1]=np.nan
    return result


def propagate_1d(boundary,dx,wavelength,depths,method='asm',selection=None):
    """Existing outgoing ASM/Fresnel multiplier; s>=0 is distance from a boundary."""
    boundary=np.asarray(boundary,dtype=complex)
    spectrum=np.fft.fft(boundary)
    fx=np.fft.fftfreq(boundary.size,dx)
    fields=[];edge_fraction=0.
    edge=max(1,boundary.size//50)
    for distance in depths:
        field=np.fft.ifft(spectrum*transfer_function(fx,0.,wavelength,float(distance),method))
        intensity=abs(field)**2
        total=float(np.sum(intensity))
        if total>0:
            edge_fraction=max(edge_fraction,float((sum(intensity[:edge])+sum(intensity[-edge:]))/total))
        fields.append(field if selection is None else field[selection])
    return np.asarray(fields),dict(spectrum=spectrum,fx=fx,edge_fraction=edge_fraction)


def image(figure,axis,values,extent,colorbars,*,title,label,cmap='inferno',**kwargs):
    im=axis.imshow(values,extent=extent,origin='lower',aspect='auto',cmap=cmap,interpolation='nearest',**kwargs)
    axis.set_title(title)
    colorbars.append(figure.colorbar(im,ax=axis,label=label,shrink=.85))
    return im


class Explorer:
    """Pending sliders; one compute on Apply, one on Reset; callbacks owned here."""
    def __init__(self,parameters,controls,compute,draw,shape,title,*,plot_bottom=.28,status_y=.215):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider,Button
        self.initial=dict(parameters);self.parameters=dict(parameters)
        self.compute,self.draw,self.title=compute,draw,title
        self.result=self.compute(self.parameters)
        self.closed=False;self.colorbars=[]
        self.figure,grid=plt.subplots(*shape,figsize=(14,9))
        self.axes=np.asarray(grid).ravel()
        self.figure.canvas.manager.set_window_title(title)
        self.figure.subplots_adjust(left=.07,right=.97,bottom=plot_bottom,top=.88,hspace=.65,wspace=.50)
        self.sliders={}
        for i,(key,label,low,high,step) in enumerate(controls):
            value=self.initial[key]
            axis=self.figure.add_axes([.15+(i%2)*.46,.19-(i//2)*.047,.28,.022])
            self.sliders[key]=Slider(axis,label,min(low,value),max(high,value),valinit=value,valstep=step)
        self.apply=Button(self.figure.add_axes([.37,.025,.12,.04]),'Apply')
        self.reset=Button(self.figure.add_axes([.52,.025,.12,.04]),'Reset')
        self.status=self.figure.text(.5,status_y,'',ha='center',va='center',fontsize=8 if plot_bottom>.28 else 9)
        self.apply_id=self.apply.on_clicked(self.update)
        self.reset_id=self.reset.on_clicked(self.restore)
        self.close_id=self.figure.canvas.mpl_connect('close_event',self.close)
        self.render()

    def render(self):
        for bar in self.colorbars:
            bar.remove()
        self.colorbars.clear()
        for axis in self.axes:
            axis.clear()
        self.draw(self.result,self.figure,self.axes,self.colorbars)
        self.figure.suptitle(self.title+'\n'+self.result.get('caption',''),fontsize=11)
        self.status.set_text(self.result.get('status','Adjust parameters, then Apply. No calculation while dragging.'))
        self.figure.canvas.draw_idle()

    def update(self,event=None):
        candidate=dict(self.initial)
        for key,slider in self.sliders.items():
            value=slider.val
            candidate[key]=int(round(value)) if isinstance(self.initial[key],int) else float(value)
        try:
            if candidate!=self.parameters:
                result=self.compute(candidate)
                self.parameters,self.result=candidate,result
            self.render()
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            self.status.set_text(str(error));self.figure.canvas.draw_idle()

    def restore(self,event=None):
        for slider in self.sliders.values():
            slider.reset()
        self.update()

    def close(self,event=None):
        if self.closed:return
        self.closed=True
        self.apply.disconnect(self.apply_id);self.reset.disconnect(self.reset_id)
        self.figure.canvas.mpl_disconnect(self.close_id)
        import matplotlib.pyplot as plt
        plt.close(self.figure)


def run(parameters,controls,compute,draw,shape,title,*,interactive=True,plot_backend='QtAgg',plot_bottom=.28,status_y=.215):
    import matplotlib
    if not interactive:matplotlib.use('Agg')
    elif plot_backend is not None:matplotlib.use(plot_backend)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'lines.linewidth':2.2,'lines.markersize':6,'font.size':10})
    if title in _instances:_instances[title].close()
    explorer=Explorer(parameters,controls,compute,draw,shape,title,plot_bottom=plot_bottom,status_y=status_y)
    _instances[title]=explorer
    print(title)
    print(explorer.result.get('caption',''))
    print(explorer.result.get('status',''))
    if 'table' in explorer.result:print(explorer.result['table'])
    if interactive:plt.show(block=True)
    else:explorer.close()
    return explorer
