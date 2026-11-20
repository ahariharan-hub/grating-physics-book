"""Physical claims and control lifecycle of the introductory propagation lesson."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from wave_optics.field_propagation import prescribed_order_data, component_field
from wave_optics.scalar_propagation import transfer_function


class FieldPropagationTests(unittest.TestCase):
    def test_reference_field_and_coefficient_ordering(self):
        x=np.linspace(-8,8,801)
        d=prescribed_order_data((0,1))
        assert_allclose(component_field(d['vectors'],d['coefficients'],x),
                        1+np.exp(1j*np.pi*x/2),atol=1e-14)
        reverse=prescribed_order_data((1,0),(2j,1.))
        assert_allclose(component_field(reverse['vectors'],reverse['coefficients'],x),
                        1+2j*np.exp(1j*np.pi*x/2),atol=1e-14)

    def test_branch_factors_match_positive_distance_preview(self):
        d=prescribed_order_data((0,1,2,3))
        assert list(d['status'])==['propagating','propagating','grazing','evanescent']
        s=np.array([0.,.5,1.])
        factor=np.exp(-1j*s[:,None]*d['vectors'][:,2])
        assert_allclose(abs(factor[:,:2]),1.,atol=1e-14)
        assert_allclose(factor[:,2],1.)
        assert_allclose(factor[:,3],np.exp(-np.pi*np.sqrt(5)*s/2))
        self.assertTrue(np.all(np.diff(abs(factor[:,3]))<0))
        for depth,values in zip(s,factor):
            assert_allclose(values,transfer_function(d['vectors'][:,0].real/(2*np.pi),
                             0.,2.,depth),atol=1e-14)
        with self.assertRaises(ValueError):
            component_field(d['vectors'],d['coefficients'],0.,s=-1.)

    def test_interference_phase_sign_and_single_order(self):
        x=np.linspace(-8,8,801);s=2.
        d=prescribed_order_data((0,1))
        u=component_field(d['vectors'],d['coefficients'],x,s=s)
        delta=-np.pi*(1-np.sqrt(3)/2)*s
        assert_allclose(abs(u)**2,2+2*np.cos(np.pi*x/2+delta),atol=1e-13)
        x_peak=-delta/(np.pi/2)
        self.assertGreater(x_peak,0)
        self.assertAlmostEqual(abs(component_field(d['vectors'],d['coefficients'],x_peak,s=s))**2,4.)
        one=prescribed_order_data((1,))
        assert_allclose(abs(component_field(one['vectors'],one['coefficients'],x,s=s))**2,1.,atol=1e-14)

    def test_equal_reference_intensities_different_downstream(self):
        x=np.linspace(-8,8,801)
        a,b=[prescribed_order_data(o) for o in [(0,1),(0,-1)]]
        def intensity(d,s):return abs(component_field(d['vectors'],d['coefficients'],x,s=s))**2
        assert_allclose(intensity(a,0.),intensity(b,0.),atol=1e-14)
        self.assertGreater(np.max(abs(intensity(a,2.)-intensity(b,2.))),2.)

    def test_explicit_vacuum_medium_conversion(self):
        a=prescribed_order_data(wavelength0=3.,index=1.5)
        b=prescribed_order_data(wavelength0=2.,index=1.)
        assert_allclose(a['vectors'],b['vectors'],atol=1e-14)

    def test_controls_phase_mask_and_cleanup(self):
        from wave_optics.field_propagation_view import FieldPropagationExplorer,masked_phase
        from ipywidgets.widgets.widget import _instances
        before=set(_instances)
        explorer=FieldPropagationExplorer()
        try:
            self.assertTrue(bytes(explorer.picture.value).startswith(b'\x89PNG'))
            cached=explorer.evolution
            explorer.distance.value=0.
            assert_allclose(explorer.observed,explorer.reference)
            explorer.distance.value=4.
            self.assertIs(explorer.evolution,cached)
            explorer.mode.value='single'
            assert_allclose(abs(explorer.observed)**2,1.,atol=1e-14)
            self.assertTrue(np.ma.getmaskarray(masked_phase(np.array([0j,1j])))[0])
            explorer.reset.click()
            self.assertEqual(explorer.mode.value,'pair')
            self.assertEqual(explorer.distance.value,2.)
        finally:
            explorer.close();explorer.close()
        self.assertEqual(set(_instances),before)


if __name__=='__main__':unittest.main()
