"""Focused separation checks for 2D geometry and later objective collection."""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from wave_optics.lattice_grating import lattice_orders, collect_orders, reciprocal_basis


class LatticeGeometryTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([[0,0],[1,0],[0,1],[1,1],[2,0]])

    def test_geometry_requires_no_objective_and_coordinate_scales_agree(self):
        d=lattice_orders(np.diag([4.,5.]),self.labels,wavelength0=2.,incident=(.3,.4))
        self.assertNotIn('q',d)
        self.assertNotIn('collected',d)
        assert_allclose(d['f']*2*np.pi,d['k'])
        assert_allclose(d['components']*d['n_out']*d['k0'],d['k'])
        self.assertEqual(d['status'][3],'evanescent')

    def test_collection_preserves_geometry_labels_source_and_coefficients(self):
        d=lattice_orders(np.diag([4.,5.]),self.labels,wavelength0=2.)
        d.update(source_id=17,C=np.arange(5)+1j)
        c=collect_orders(d,.8)
        assert_array_equal(c['k'],d['k'])
        assert_array_equal(c['labels'],d['labels'])
        assert_array_equal(c['C'],d['C'])
        self.assertEqual(c['source_id'],17)
        assert_array_equal(c['collected'],[True,True,True,True,False])
        self.assertNotIn('q',d)
        legacy=lattice_orders(np.diag([4.,5.]),self.labels,wavelength0=2.,NA_obj=.8)
        assert_allclose(legacy['q'],c['q'])
        assert_array_equal(legacy['collected'],c['collected'])

    def test_grazing_and_evanescent_never_collected(self):
        d=lattice_orders(np.eye(2),[[0,0],[1,0],[2,0]],wavelength0=1.)
        assert_array_equal(d['status'],['propagating','grazing','evanescent'])
        assert_array_equal(collect_orders(d,1.)['collected'],[True,False,False])
        self.assertLess(d['k'][2,2].imag,0)
        self.assertLess(abs(np.exp(-.2j*d['k'][2,2])),1.)

    def test_na_validation_is_separate_from_geometry(self):
        d=lattice_orders(np.eye(2),self.labels,wavelength0=1.,n_out=1.33)
        for na in [0.,-1.,np.nan,np.inf,1.34]:
            with self.assertRaises(ValueError):collect_orders(d,na)

    def test_duality_rotation_and_incidence_translation(self):
        A=np.array([[4.,1.],[0.,5.]])
        angle=.3
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        assert_allclose(A.T@reciprocal_basis(A),2*np.pi*np.eye(2),atol=1e-12)
        assert_allclose(reciprocal_basis(R@A),R@reciprocal_basis(A),atol=1e-12)
        d0=lattice_orders(A,self.labels,wavelength0=2.)
        d1=lattice_orders(A,self.labels,wavelength0=2.,n_in=1.5,incident=(.1,.2))
        assert_allclose(d1['k'][:,:2]-d0['k'][:,:2],
                        np.tile(1.5*np.pi*np.array([.1,.2]),(5,1)),atol=1e-12)

class RelocatedNotebookChecks(unittest.TestCase):
    """Physical checks formerly embedded in Lessons 5.3 and 5.4."""

    def test_one_dimensional_row_matches_conical_helper(self):
        from wave_optics.conical import conical_orders
        labels=np.column_stack([np.arange(-3,4),np.zeros(7,dtype=int)])
        row=lattice_orders(np.diag([4.,5.]),labels,wavelength0=2.,incident=(.3,.4))
        old=conical_orders(wavelength=2.,period=4.,alpha_i=.3,beta_i=.4,orders=np.arange(-3,4))
        assert_allclose(row['k']/np.pi,old['components'],atol=1e-12)

    def test_outgoing_dispersion_and_decay(self):
        d=lattice_orders(np.eye(2),[[0,0],[1,0],[2,0]],wavelength0=1.)
        assert_allclose(np.sum(d['k']**2,axis=1),(2*np.pi)**2,atol=1e-12)
        depths=np.array([0.,.1,.2])
        amplitude=abs(np.exp(-1j*d['k'][2,2]*depths))
        assert_allclose(amplitude,np.exp(-2*np.pi*np.sqrt(3)*depths))
        self.assertTrue(np.all(np.diff(amplitude)<0))

    def test_output_index_changes_propagation_not_transverse_increments(self):
        v=np.arange(-6,7)
        mm,nn=np.meshgrid(v,v)
        labels=np.column_stack([mm.ravel(),nn.ravel()])
        args=dict(direct=np.diag([4.,5.]),labels=labels,wavelength0=2.,incident=(.3,.4))
        air=lattice_orders(**args)
        glass=lattice_orders(**args,n_out=1.5)
        assert_allclose(air['k'][:,:2],glass['k'][:,:2])
        self.assertGreater(sum(glass['status']=='propagating'),sum(air['status']=='propagating'))

    def test_source_labels_and_declared_complex_pupil(self):
        labels=np.array([[0,0],[1,0],[2,0]])
        records=[collect_orders(dict(lattice_orders(np.diag([4.,5.]),labels,
                    wavelength0=2.,incident=inc),source_id=i),.8)
                 for i,inc in enumerate([(0.,0.),(.1,-.1)])]
        for i,d in enumerate(records):
            assert_array_equal(d['labels'],labels)
            self.assertEqual(d['source_id'],i)
            expected=((d['status']=='propagating') &
                      (np.sum(d['q']**2,axis=1)<=1+1e-12))
            assert_array_equal(d['collected'],expected)
            H=d['collected']*np.exp(.3j*np.sum(d['q']**2,axis=1))
            assert_allclose(abs(H[d['collected']]),1.)
            assert_array_equal(H[~d['collected']],0.)


if __name__=='__main__':unittest.main()
