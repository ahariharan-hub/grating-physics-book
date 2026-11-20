"""Known coherent components propagated toward -z, using positive depth s.

No scattering, pupil, FFT or lens model. Length units are shared by all inputs.
"""
import numpy as np
from .conical import conical_orders


def prescribed_order_data(orders=(0, 1), coefficients=None, *, wavelength0=2.,
                          index=1., period=4.):
    """Normal-incidence 1D grating geometry and explicitly prescribed C at z_ref.

    wavelength0 is vacuum wavelength; index is the lossless outgoing index.
    Coefficients use unit field amplitude, lateral origin x=y=0.
    """
    if not np.all(np.isfinite([wavelength0,index,period])) or min(wavelength0,index,period)<=0:
        raise ValueError('Use positive finite wavelength0, index and period.')
    labels=np.asarray(orders)
    if labels.ndim!=1 or not labels.size or len(np.unique(labels))!=labels.size:
        raise ValueError('Use a nonempty sequence of distinct integer orders.')
    geometry=conical_orders(wavelength=wavelength0/index,period=period,
                             orders=labels,selected_order=labels[0],alpha_i=0.,beta_i=0.)
    # conical_orders sorts labels; reorder coefficients by their given labels.
    c=np.ones(len(labels),complex) if coefficients is None else np.asarray(coefficients,complex)
    if c.shape!=labels.shape or not np.all(np.isfinite(c)):
        raise ValueError('Supply one finite complex coefficient per order.')
    c=c[np.argsort(labels)]
    return dict(orders=geometry['labels'], vectors=geometry['k']*geometry['components'],
                coefficients=c, status=geometry['states'], wavelength0=float(wavelength0),
                index=float(index),period=float(period))


def component_field(vectors, coefficients, x, y=0., s=0.):
    """Sum C_j exp(i[kx*x+ky*y-kz*s]); x, y, s broadcast, s >= 0.

    Coefficients are at the reference plane; s is positive distance toward -z,
    not a signed global z coordinate. Coordinates on a nonplanar surface may
    be supplied as arrays, provided every point is in the outgoing region.
    """
    v,c=np.asarray(vectors,complex),np.asarray(coefficients,complex)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)==0 or c.shape!=(len(v),):
        raise ValueError('Use vectors of shape (N,3) and N coefficients.')
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(c)):
        raise ValueError('Vectors and coefficients must be finite.')
    if np.any(v[:,:2].imag!=0) or np.any(v[:,2].real>0) or np.any(v[:,2].imag>0):
        raise ValueError('Use real transverse vectors and outgoing/decaying negative kz.')
    coords=[]
    for value in [x,y,s]:
        if np.iscomplexobj(value):raise ValueError('Coordinates must be real.')
        a=np.asarray(value,float)
        if not np.all(np.isfinite(a)):raise ValueError('Coordinates must be finite.')
        coords.append(a)
    x,y,s=np.broadcast_arrays(*coords)
    if np.any(s<0):raise ValueError('s must be nonnegative distance into the outgoing region.')
    phase=x[...,None]*v[:,0]+y[...,None]*v[:,1]-s[...,None]*v[:,2]
    return np.sum(c*np.exp(1j*phase),axis=-1)
