"""Three-dimensional phase matching, geometry and linked notebook controls."""

import gc
import re
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.conical import conical_orders, conical_phasor, detector_intersections
from wave_optics.diffraction import diffraction_orders
from wave_optics.evanescent import evanescent_order


class ConicalPhysicsTests(unittest.TestCase):
    def test_detector_intersections_match_rays_and_notebook_two(self):
        from wave_optics.grating_intro import order_surface_intersections
        for alpha, beta in [(0, 0), (.3, .4), (.3, .6), (.3, .8), (-.3, -.4), (.3, .599999)]:
            data = conical_orders(alpha_i=alpha, beta_i=beta)
            near = detector_intersections(data, 3.)
            far = detector_intersections(data, 6.)
            hits = near['reaches']
            assert_allclose(near['points'][hits, 2], -3.)
            assert_allclose(np.cross(near['points'][hits], data['components'][hits].real), 0, atol=1e-10)
            assert_allclose(far['points'][hits], 2*near['points'][hits])
            self.assertTrue(np.all(np.isnan(near['points'][~hits])))
            self.assertTrue(np.all(near['in_patch'] <= hits))
            old = order_surface_intersections(data['k']*data['directions'], 4., 3.)
            assert_allclose(near['points'][data['valid']], old['plane'], equal_nan=True)
            np.testing.assert_array_equal(near['reaches'][data['valid']], old['reaches_plane'])
        # Approaching grazing sends the plane intersection to infinity, not the patch edge.
        i = data['selected_index']
        self.assertTrue(near['reaches'][i])
        self.assertFalse(near['in_patch'][i])
        self.assertGreater(near['points'][i, 0], 1000)
        for bad in (0., -1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                detector_intersections(data, bad)

    def test_reference_transition_and_selection(self):
        expected = { .4:([-2, -1, 0, 1], 'propagating'),
                     .6:([-2, -1, 0, 1], 'grazing'),
                     .8:([-1, 0], 'evanescent') }
        for beta, (labels, status) in expected.items():
            data = conical_orders(beta_i=beta)
            self.assertEqual(data['allowed'].tolist(), labels)
            self.assertEqual(data['states'][data['selected_index']], status)
            self.assertEqual(data['chosen'], 1)
        ev = conical_orders(beta_i=.8)
        assert_allclose(ev['components'][ev['selected_index'], 2], -1j*np.sqrt(.28))
        envelope = abs(conical_phasor(.17, .29, -np.arange(3), ev))
        assert_allclose(envelope, np.exp(-np.pi*np.sqrt(.28)*np.arange(3)))

    def test_planar_limit_matches_notebook_four(self):
        for angle in [-60., 0., 30.]:
            alpha = np.sin(np.deg2rad(angle))
            for m in range(-3, 4):
                data = conical_orders(2., 2., alpha, 0., m)
                old = evanescent_order(2., 2., angle, m)
                index = data['selected_index']
                self.assertEqual(data['states'][index], old['status'])
                assert_allclose(data['k']*data['components'][index],
                                [old['kx'], 0, old['kz']], atol=1e-12)

    def test_three_dimensional_phase_and_helmholtz(self):
        # Finite differences of fields, independent of the displayed vectors.
        h = 1e-4
        r = np.array([.17, .29, -.31])
        for m in [-2, 0, 1, 2]:
            data = conical_orders(beta_i=.4, selected_order=m)
            field = conical_phasor(*r, data)
            translated_x = conical_phasor(*(r+[data['period'], 0, 0]), data)
            translated_y = conical_phasor(*(r+[0, .73, 0]), data)
            assert_allclose(translated_x/field,
                np.exp(1j*data['k']*data['incident'][0]*data['period']), atol=1e-12)
            assert_allclose(translated_y/field,
                np.exp(1j*data['k']*data['incident'][1]*.73), atol=1e-12)
            laplacian = sum((conical_phasor(*(r+h*axis), data)
                            +conical_phasor(*(r-h*axis), data)-2*field)/h**2
                           for axis in np.eye(3))
            self.assertLess(abs(laplacian+data['k']**2*field), 3e-6)
            assert_allclose(np.sum(data['components']**2, axis=1), 1, atol=1e-12)

    def test_matches_older_geometry_and_ewald_constraints(self):
        labels = np.arange(-4, 5)
        for theta, phi in [(0, 0), (30, 53.13), (70, -120), (40, 90)]:
            old = diffraction_orders(1.7, 3.2, theta, phi, orders=labels)
            data = conical_orders(1.7, 3.2, *old['incident'][:2], orders=labels)
            assert_allclose(data['directions'], old['directions']*[1, 1, -1], atol=1e-12)
            assert_allclose(data['changes']+data['incident'], data['directions'], atol=1e-12)
            assert_allclose(data['changes'][:, 0], data['allowed']*data['spacing'], atol=1e-12)
            assert_allclose(data['changes'][:, 1], 0, atol=1e-12)
            assert_allclose(np.linalg.norm(data['changes']+data['incident'], axis=1), 1)
            self.assertTrue(np.all(data['directions'][:, 2] <= 0))
        zero = conical_orders(beta_i=.4, selected_order=0)
        assert_allclose(zero['components'][zero['selected_index']], zero['incident'])

    def test_invalid_inputs_and_retained_distant_candidate(self):
        for kwargs in ({'wavelength':0}, {'period':np.nan}, {'alpha_i':1},
                       {'alpha_i':.8, 'beta_i':.8}, {'selected_order':.5},
                       {'orders':[]}, {'orders':[.5, 1]}, {'orders':[0]}):
            with self.assertRaises(ValueError):
                conical_orders(**kwargs)
        data = conical_orders(selected_order=20)
        self.assertIn(20, data['labels'])
        self.assertEqual(data['states'][data['selected_index']], 'evanescent')
        for point in [(0, 0, .1), (0, np.inf, 0)]:
            with self.assertRaises(ValueError):
                conical_phasor(*point, data)


class ConicalViewTests(unittest.TestCase):
    @staticmethod
    def roles(figure):
        return {trace.meta['role']:trace for trace in figure.data
                if isinstance(trace.meta, dict) and 'role' in trace.meta}

    def test_coordinate_scales_preserve_geometry_phase_and_order_states(self):
        from wave_optics.conical_plotting import representation_figures
        for beta in (.4, .6, .8):
            data = conical_orders(beta_i=beta)
            baseline, ewald = representation_figures(data)
            for scale, factor, unit in [('frequency', .5, 'cycles/µm'),
                                        ('wavevector', np.pi, 'rad/µm')]:
                figure, unchanged_ewald = representation_figures(data, scale)
                self.assertEqual(ewald.to_json(), unchanged_ewald.to_json())
                # Rescale every guide and marker, not just the selected point.
                for before, after in zip(baseline.data, figure.data, strict=True):
                    for axis in ('xyz' if before.type == 'scatter3d' else 'xy'):
                        assert_allclose(np.asarray(getattr(after, axis), dtype=float),
                                        factor*np.asarray(getattr(before, axis), dtype=float))
                    self.assertEqual(before.visible, after.visible)
                    if before.customdata is not None:
                        np.testing.assert_array_equal(before.customdata, after.customdata)
                roles = self.roles(figure)
                self.assertIn(unit, figure.layout.xaxis.title.text)
                self.assertIn(unit, roles['selected-disk'].hovertemplate)
                self.assertEqual(roles['selected-direction'].visible, beta <= .6)
                self.assertEqual(roles['selected-ray'].visible, beta <= .6)
                assert_allclose(figure.layout.xaxis.range,
                                factor*np.array(baseline.layout.xaxis.range))
                assert_allclose(figure.layout.scene.zaxis.range,
                                factor*np.array(baseline.layout.scene.zaxis.range))
                if scale == 'frequency':
                    fx, fy = roles['selected-disk'].x[0], roles['selected-disk'].y[0]
                    rad = 1/data['wavelength']**2-fx**2-fy**2
                    if abs(rad) < 1e-12:
                        rad = 0.
                    fz = -np.sqrt(rad+0j)
                    point = np.array([.17, .29, -.31])
                    # Independent field reconstruction from the plotted frequencies.
                    assert_allclose(np.exp(2j*np.pi*np.dot([fx, fy, fz], point)),
                                    conical_phasor(*point, data))

    def test_live_coordinate_switch_keeps_selection_camera_and_frequency_step(self):
        from wave_optics.conical_view import ConicalDiffractionExplorer
        explorer = ConicalDiffractionExplorer()
        try:
            _ = explorer.harvey_ui
            live = explorer._figures['harvey']
            camera = dict(eye=dict(x=-1., y=2., z=.7))
            live.layout.scene.camera = camera
            explorer.skew.click()
            for scale in ('frequency', 'wavevector', 'direction', 'frequency'):
                explorer.coordinate_scale.value = scale
                factor = {'frequency':.5, 'wavevector':np.pi, 'direction':1.}[scale]
                selected = self.roles(live)['selected-disk']
                assert_allclose([selected.x[0], selected.y[0]], factor*np.array([.8, .4]))
                self.assertEqual(explorer.order.value, 1)
                self.assertEqual(live.layout.scene.camera.to_plotly_json(), camera)
            for wavelength, period in [(1.,4.), (4.,4.), (.5,6.)]:
                explorer.wavelength.value = wavelength
                explorer.period.value = period
                row = next(t for t in live.data if t.type == 'scatter'
                           and t.mode == 'markers+text')
                labels = np.asarray(row.customdata)[:, 0].astype(int)
                # The phase-cycle step depends on period, not wavelength.
                assert_allclose(np.diff(row.x)/np.diff(labels), 1/period)
                assert_allclose(row.y, explorer.beta.value/wavelength)
                self.assertEqual(live.layout.scene.camera.to_plotly_json(), camera)
            for button, state in [(explorer.cutoff, 'grazing'), (explorer.beyond, 'evanescent')]:
                button.click()
                self.assertEqual(explorer.coordinate_scale.value, 'frequency')
                self.assertEqual(explorer.data['states'][explorer.data['selected_index']], state)
                self.assertEqual(self.roles(live)['selected-ray'].visible, state == 'grazing')
            self.assertIn('imaginary', explorer.coordinate_summary.value)
        finally:
            explorer.close()
        del explorer
        gc.collect()

    def test_selected_component_projections_and_complex_kz(self):
        from wave_optics.conical_plotting import spatial_figure, detector_readout
        for alpha, beta, m in [(0, 0, 0), (.3, .4, 1), (.3, .6, 1),
                               (.3, .8, 1), (-.3, -.4, -1)]:
            data = conical_orders(alpha_i=alpha, beta_i=beta, selected_order=m)
            figure = spatial_figure(data)
            roles = self.roles(figure)
            vector = data['k']*data['components'][data['selected_index']]
            total = np.zeros(3)
            for j, axis in enumerate('xyz'):
                trace = roles[f'component-{axis}']
                if abs(vector[j].real) < 1e-12:
                    self.assertEqual(len(trace.x), 0)
                else:
                    start = np.array([trace.x[0], trace.y[0], trace.z[0]])
                    end = np.array([trace.x[1], trace.y[1], trace.z[1]])
                    assert_allclose(start, 0)
                    expected = np.eye(3)[j]*vector[j].real
                    assert_allclose(end-start, expected)
                    total += end-start
            full = roles['selected-wavevector']
            assert_allclose([full.x[-1], full.y[-1], full.z[-1]], total)
            assert_allclose(total, vector.real)
            if vector[2].imag:
                self.assertIn('Re(k', re.sub(r'<[^>]+>', '', full.name))
                self.assertIn('imaginary', figure.layout.annotations[-1].text)
                self.assertIn('no real outgoing ray', detector_readout(data))
            if data['grazing'][data['selected_index']]:
                self.assertIn('parallel', detector_readout(data))
            hits = detector_intersections(data)
            points = roles['detector-hits']
            assert_allclose(np.column_stack([points.x, points.y, points.z]),
                            hits['points'][hits['in_patch']])
            np.testing.assert_array_equal(points.customdata, data['labels'][hits['in_patch']])

    def test_detector_distance_leaves_wavevector_and_both_cameras_unchanged(self):
        from wave_optics.conical_view import ConicalDiffractionExplorer
        explorer = ConicalDiffractionExplorer()
        try:
            _ = explorer.orders_ui
            explorer.skew.click()
            live = explorer._figures['orders']
            count = len(live.data)
            first = self.roles(live)
            vector = first['selected-wavevector'].to_plotly_json()
            endpoints = first['reference-endpoints'].to_plotly_json()
            camera = dict(eye=dict(x=-1., y=2., z=.7))
            camera2 = dict(eye=dict(x=1., y=-1., z=1.2))
            live.layout.scene.camera, live.layout.scene2.camera = camera, camera2
            explorer.detector_distance.value = 6.
            updated = self.roles(live)
            assert_allclose(updated['detector-plane'].z, -6.)
            for axis in 'xyz':
                assert_allclose(updated['selected-wavevector'][axis], vector[axis])
                assert_allclose(updated['reference-endpoints'][axis], endpoints[axis])
            self.assertIn('outside the drawn patch', explorer.detector_summary.value)
            explorer.detector.value = False
            self.assertFalse(self.roles(live)['detector-plane'].visible)
            self.assertEqual(len(self.roles(live)['detector-hits'].x), 0)
            explorer.detector.value = True
            for button in (explorer.cutoff, explorer.beyond, explorer.reset):
                button.click()
                self.assertEqual(len(live.data), count)
                self.assertEqual(live.layout.scene.camera.to_plotly_json(), camera)
                self.assertEqual(live.layout.scene2.camera.to_plotly_json(), camera2)
        finally:
            explorer.close()
        del explorer
        gc.collect()

    def test_incident_wavefront_planes_and_surface_intersections(self):
        from wave_optics.conical_plotting import incidence_figure
        for alpha, beta, wavelength in [(0, 0, 2), (.3, 0, 2), (.3, .4, 2),
                                        (-.3, -.4, 2), (.95, .31, .5), (0, .8, 4)]:
            data = conical_orders(wavelength=wavelength, alpha_i=alpha, beta_i=beta)
            roles = self.roles(incidence_figure(data))
            mesh = roles['incident-wavefronts']
            vertices = np.column_stack([mesh.x, mesh.y, mesh.z])
            # Every rendered plane is a crest of the actual incident phasor.
            phase = data['k']*(vertices @ data['incident'])
            assert_allclose(phase, 2*np.pi*np.asarray(mesh.customdata), atol=1e-10)
            self.assertTrue(np.all(vertices[:, 2] >= 0))
            self.assertTrue(np.all(vertices[:, 2] <= 6))
            for i, j, k in zip(mesh.i, mesh.j, mesh.k, strict=True):
                normal = np.cross(vertices[j]-vertices[i], vertices[k]-vertices[i])
                self.assertGreater(np.linalg.norm(normal), 1e-10)
                assert_allclose(abs(np.dot(normal/np.linalg.norm(normal), data['incident'])), 1)
            footprint = roles['wavefront-footprints']
            points = np.array([[x,y,z] for x,y,z in zip(footprint.x, footprint.y, footprint.z)
                               if x is not None]).reshape(-1, 3)
            assert_allclose(points[:, 2], 0)
            cycles = points @ data['incident']/wavelength
            assert_allclose(cycles, np.round(cycles), atol=1e-10)
            self.assertEqual(tuple(roles['wavefront-footprints-map'].x), tuple(footprint.x))
            self.assertEqual(tuple(roles['wavefront-footprints-map'].y), tuple(footprint.y))
            surface = roles['incident-surface-field']
            expected_phase = data['k']*(alpha*np.asarray(surface.x)[None, :]
                                       +beta*np.asarray(surface.y)[:, None])
            assert_allclose(surface.customdata, expected_phase)
            assert_allclose(surface.z, np.cos(expected_phase))
            if alpha == beta == 0:
                assert_allclose(np.unique(mesh.z), [2, 4, 6])
                self.assertEqual(len(points), 0)
                assert_allclose(surface.z, 1)

    def test_incident_presets_live_geometry_and_camera(self):
        from wave_optics.conical_view import ConicalDiffractionExplorer
        explorer = ConicalDiffractionExplorer()
        try:
            _ = explorer.incidence_ui
            live = explorer._figures['incidence']
            camera = dict(eye=dict(x=-1., y=2., z=.7))
            live.layout.scene.camera = camera
            count = len(live.data)
            for button, alpha, beta in [(explorer.normal, 0, 0),
                                        (explorer.reset, .3, 0), (explorer.skew, .3, .4)]:
                button.click()
                assert_allclose(explorer.data['incident'][:2], [alpha, beta])
                mesh = self.roles(live)['incident-wavefronts']
                vertices = np.column_stack([mesh.x, mesh.y, mesh.z])
                assert_allclose(vertices @ explorer.data['incident'],
                                explorer.data['wavelength']*np.asarray(mesh.customdata), atol=1e-10)
                self.assertEqual(live.layout.scene.camera.to_plotly_json(), camera)
                self.assertEqual(len(live.data), count)
            for wavelength in (.5, 4):
                explorer.wavelength.value = wavelength
                mesh = self.roles(live)['incident-wavefronts']
                assert_allclose(np.column_stack([mesh.x, mesh.y, mesh.z]) @ explorer.data['incident'],
                                wavelength*np.asarray(mesh.customdata), atol=1e-10)
            self.assertIn('phase now changes along', explorer.incident_summary.value)
            self.assertNotIn(explorer.cutoff, explorer.incident_controls.children[-1].children)
        finally:
            explorer.close()
        del explorer
        gc.collect()

    def test_rendered_hemisphere_vector_triangle_and_evanescent_marker(self):
        from wave_optics.conical_plotting import representation_figures, spatial_figure
        for beta, order in [(.4, 1), (.6, 1), (.8, 1), (.4, 0), (-.4, -2)]:
            data = conical_orders(beta_i=beta, selected_order=order)
            harvey, ewald = representation_figures(data)
            h, e = self.roles(harvey), self.roles(ewald)
            index = data['selected_index']
            self.assertLess(harvey.layout.scene.zaxis.range[0], -1)
            assert_allclose([h['selected-disk'].x[0], h['selected-disk'].y[0]],
                            data['components'][index, :2].real)
            assert_allclose(e['selected-rod'].x, order*data['spacing'])
            if data['valid'][index]:
                q = data['components'][index].real-data['incident']
                assert_allclose([e['selected-ewald'].x[0], e['selected-ewald'].y[0],
                                 e['selected-ewald'].z[0]], q)
                vector = e['outgoing-vector']
                start = np.array([vector.x[0], vector.y[0], vector.z[0]])
                end = np.array([vector.x[1], vector.y[1], vector.z[1]])
                assert_allclose(start, -data['incident'])
                assert_allclose(end-start, data['components'][index].real)
                if order == 0:
                    self.assertEqual(len(e['change-head'].x), 0)
            else:
                self.assertFalse(h['selected-ray'].visible)
                self.assertFalse(e['outgoing-vector'].visible)
                self.assertFalse(e['selected-ewald'].visible)
            spatial = spatial_figure(data, show_cone=True)
            cone = next(t for t in spatial.data if t.type == 'surface')
            radii = np.sqrt(np.asarray(cone.x)**2+np.asarray(cone.y)**2+np.asarray(cone.z)**2)
            mask = radii > 0
            assert_allclose(np.asarray(cone.y)[mask]/radii[mask], beta, atol=1e-12)
            self.assertTrue(np.all(np.asarray(cone.z) <= 0))

    def test_linked_presets_camera_limits_and_disposal(self):
        from ipywidgets.widgets.widget import _instances
        from wave_optics.conical_view import ConicalDiffractionExplorer
        before = set(_instances)
        explorer = ConicalDiffractionExplorer()
        try:
            self.assertEqual(explorer._figures, {})
            for attr in ['incidence_ui', 'orders_ui', 'harvey_ui', 'ewald_ui']:
                view = getattr(explorer, attr)
                self.assertIs(view, getattr(explorer, attr))
            camera = dict(eye=dict(x=2., y=1., z=.3))
            explorer._figures['harvey'].layout.scene.camera = camera
            for button, status in [(explorer.skew,'propagating'),
                                   (explorer.cutoff,'grazing'), (explorer.beyond,'evanescent')]:
                button.click()
                self.assertEqual(explorer.order.value, 1)
                self.assertEqual(explorer.data['states'][explorer.data['selected_index']], status)
            self.assertEqual(explorer._figures['harvey'].layout.scene.camera.to_plotly_json(), camera)
            explorer.alpha.value = .95
            self.assertLess(explorer.alpha.value**2+explorer.beta.value**2, 1)
            self.assertIn('adjusted', explorer.notice.value)
            explorer.skew.click()
            explorer.cone.value = True
            self.assertTrue(next(t for t in explorer._figures['orders'].data
                                 if t.type == 'surface').visible)
            explorer.order.value = 0
            self.assertIn('collapses', explorer.ewald_summary.value)
            # Stress a changing order count and retention beyond the automatic row.
            explorer.wavelength.value = .5
            explorer.period.value = 6
            explorer.order.value = -10
            explorer.wavelength.value = 4
            explorer.period.value = 1
            self.assertEqual(explorer.order.value, -10)
            explorer.reset.click()
            self.assertEqual(explorer.order.value, 1)
            self.assertFalse(explorer.cone.value)
        finally:
            explorer.close()
            explorer.close()
        self.assertEqual(set(_instances), before)
        with self.assertRaises(RuntimeError):
            _ = explorer.harvey_ui
        del view, explorer
        gc.collect()

    def test_order_labels_follow_coordinates_when_order_count_changes(self):
        from wave_optics.conical_plotting import RADIUS
        from wave_optics.conical_view import ConicalDiffractionExplorer

        def check_labels(explorer):
            data = explorer.data
            rays = next(t for t in explorer._figures['orders'].data
                        if t.type == 'scatter3d' and t.mode == 'markers+text')
            self.assertEqual(len(rays.text), len(rays.x))
            # Recover each integer from its plotted position, independently of
            # the label array. Hover labels must agree even on crowded plots.
            ray_orders = np.rint((np.asarray(rays.x)/RADIUS-data['incident'][0])
                                 /data['spacing']).astype(int)
            assert_allclose(rays.customdata[:, 0], ray_orders)
            for label, order in zip(rays.text, ray_orders, strict=True):
                if label:
                    self.assertEqual(int(label.removeprefix('m=')), order)
            zero = list(ray_orders).index(0)
            assert_allclose([rays.x[zero], rays.y[zero], rays.z[zero]],
                            RADIUS*data['incident'])
            self.assertEqual(rays.text[zero], 'm=+0')
            if data['chosen'] in ray_orders:
                selected = list(ray_orders).index(data['chosen'])
                self.assertEqual(rays.text[selected], f'm={data["chosen"]:+d}')
            if len(ray_orders) <= 13:
                self.assertTrue(all(rays.text))

            rods = explorer._figures['ewald'].data[-1]
            self.assertEqual(len(rods.text), len(rods.x))
            rod_orders = np.rint(np.asarray(rods.x)/data['spacing']).astype(int)
            for label, order in zip(rods.text, rod_orders, strict=True):
                if label:
                    self.assertEqual(int(label), order)
            for order in (0, data['chosen']):
                self.assertEqual(rods.text[list(rod_orders).index(order)], f'{order:+d}')
            if len(rod_orders) <= 15:
                self.assertTrue(all(rods.text))

        explorer = ConicalDiffractionExplorer()
        try:
            _ = explorer.orders_ui, explorer.ewald_ui
            check_labels(explorer)
            explorer.skew.click()
            explorer.wavelength.value = .5
            explorer.period.value = 6
            check_labels(explorer)
            explorer.order.value = -10
            check_labels(explorer)
            explorer.wavelength.value = 4
            explorer.period.value = 1
            check_labels(explorer)  # Retained selection is now evanescent.
            explorer.reset.click()
            check_labels(explorer)
        finally:
            explorer.close()
        del explorer
        gc.collect()


if __name__ == '__main__':
    unittest.main()
