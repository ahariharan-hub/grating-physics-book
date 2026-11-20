"""Physical invariants and interactive behaviour retained by the extraction."""

import gc
import json
from pathlib import Path
import unittest
import weakref

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from wave_optics import (
    diffraction_orders, direction_cosines, field_from_phasor,
    plane_wave_phasor, scalar_plane_wave,
    sphere_residuals, wavenumber, wavevector,
)


class WavePhysicsTests(unittest.TestCase):
    def test_phasor_reconstructs_real_waves_in_each_dimension(self):
        for vector in ([2.3], [1.2, -2.1], [0.7, -1.4, 2.2]):
            with self.subTest(dimensions=len(vector)):
                points = np.arange(4 * len(vector)).reshape(4, len(vector)) / 7
                phasor = plane_wave_phasor(points, vector, amplitude=1.7, phase=0.4)
                assert_allclose(np.abs(phasor), 1.7)
                for time in [0, 0.3, 1.1]:
                    assert_allclose(
                        field_from_phasor(phasor, time, omega=2.5),
                        scalar_plane_wave(points, vector, time, omega=2.5,
                                          amplitude=1.7, phase=0.4), atol=1e-14)

    def test_phasor_phase_and_time_convention(self):
        # Moving perpendicular to k stays on one complex phase plane.
        values = plane_wave_phasor([[0, 0], [-4, 3]], [3, 4])
        assert_allclose(values, [1, 1], atol=1e-14)
        # At omega*t=pi/2, a phasor i rotates clockwise onto the positive real axis.
        self.assertAlmostEqual(field_from_phasor(1j, time=0.25, omega=2*np.pi), 1)

    def test_wavevector_conventions(self):
        assert_allclose(wavevector(2, 0), [0, 0, np.pi], atol=1e-15)
        assert_allclose(direction_cosines(90, 90), [0, 1, 0], atol=1e-15)
        assert_allclose(direction_cosines(-90), [-1, 0, 0], atol=1e-15)
        self.assertAlmostEqual(np.linalg.norm(wavevector(2, 57, -83)), np.pi)

    def test_equal_phase_planes_and_periodicity(self):
        k = wavevector(2, 90, 0)
        # Motion transverse to k stays on one wavefront; lambda restores phase.
        points = [[0, 0, 0], [0, 3, 0], [2, 0, 0], [1, 0, 0]]
        assert_allclose(scalar_plane_wave(points, k), [1, 1, 1, -1], atol=1e-14)
        # For omega=pi, the period is two time units.
        self.assertAlmostEqual(scalar_plane_wave([0, 0, 0], k, time=1, omega=np.pi), -1)
        self.assertAlmostEqual(scalar_plane_wave([0, 0, 0], k, time=2, omega=np.pi), 1)

    def test_normal_incidence_grazing_and_evanescent_orders(self):
        data = diffraction_orders(500, 1000, 0, 0)
        assert_array_equal(data['allowed'], [-2, -1, 0, 1, 2])
        assert_array_equal(data['labels'][data['grazing']], [-2, 2])
        assert_array_equal(data['labels'][~data['valid']], [-3, 3])
        assert_allclose(data['directions'][[0, -1]], [[-1, 0, 0], [1, 0, 0]])

    def test_oblique_grating_and_both_sphere_constraints(self):
        data = diffraction_orders(193, 800, 45, -45)
        assert_array_equal(data['allowed'], np.arange(-5, 2))
        assert_allclose(data['directions'][:, 0], 0.5 + data['allowed'] * 193 / 800)
        assert_allclose(data['directions'][:, 1], -0.5)
        assert_allclose(data['changes'][:, 1], 0, atol=1e-15)
        assert_allclose(data['changes'][data['allowed'] == 0], 0, atol=1e-15)
        assert_allclose(sphere_residuals(data), 0, atol=1e-14)

    def test_explicit_orders_with_no_real_directions(self):
        data = diffraction_orders(500, 1000, 0, 0, orders=[-4, 4])
        self.assertEqual(data['directions'].shape, (0, 3))
        self.assertIsNone(data['chosen'])
        self.assertIsNone(sphere_residuals(data))

    def test_invalid_physical_inputs(self):
        for wavelength in [0, -1, float('nan'), float('inf')]:
            with self.subTest(wavelength=wavelength), self.assertRaises(ValueError):
                wavenumber(wavelength)
        for arguments in [(0, 1000, 30, 30), (500, -1, 30, 30), (500, 1000, 90, 0)]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                diffraction_orders(*arguments)
        with self.assertRaises(ValueError):
            diffraction_orders(500, 1000, 0, 0, orders=[0, 0.5])


class InteractiveGeometryTests(unittest.TestCase):
    def test_selection_colours_camera_and_order_disappearance(self):
        from wave_optics.plotting import geometry_order_colour
        from wave_optics.widgets import DiffractionGeometryExplorer

        explorer = DiffractionGeometryExplorer(500, 1000, 30, 30)
        try:
            before = explorer.data['directions'].copy()
            explorer.directions.layout.scene.camera.eye = dict(x=2.1, y=-1.4, z=0.9)
            camera = explorer.directions.layout.scene.camera.to_plotly_json()
            explorer.order.value = -1
            assert_array_equal(explorer.data['directions'], before)
            for figure in [explorer.directions, explorer.ewald]:
                selections = [trace for trace in figure.data
                              if isinstance(trace.meta, dict)
                              and trace.meta.get('role', '').startswith('selected-')
                              and trace.meta['role'] != 'selected-ray']
                self.assertEqual(len(selections), 2)
                for trace in selections:
                    self.assertEqual(trace.customdata[0][0], -1)
            for order in range(1, 8):
                self.assertEqual(geometry_order_colour(order), geometry_order_colour(-order))
            explorer.order.value = 0
            change_head = next(trace for trace in explorer.ewald.data
                               if isinstance(trace.meta, dict) and trace.meta.get('role') == 'change-head')
            self.assertEqual(len(change_head.x), 0)
            explorer.order.value = -2
            explorer.wavelength.value = 1500
            self.assertEqual(explorer.order.value, 0)
            self.assertEqual(explorer.directions.layout.scene.camera.to_plotly_json(), camera)
        finally:
            explorer.directions.close()
            explorer.ewald.close()
            explorer.ui.close()


class InteractivePhasorTests(unittest.TestCase):
    def test_shared_clock_position_and_dimension_controls(self):
        from wave_optics.widgets import PlaneWaveExplorer, ScalarPlaneWave3D

        for dimensions in [1, 2, 3]:
            with self.subTest(dimensions=dimensions):
                explorer = (ScalarPlaneWave3D() if dimensions == 3
                            else PlaneWaveExplorer(dimensions))
                try:
                    self.assertEqual(len(explorer.phasor.position), dimensions)
                    explorer.views.selected_index = 1
                    initial = explorer.phasor.plot.phasor
                    old_image = explorer.phasor.picture.value
                    explorer.time.value = 15
                    self.assertEqual(explorer.phasor.plot.phasor, initial)
                    assert_allclose(explorer.phasor.plot.instantaneous, -1j * initial, atol=1e-14)
                    self.assertNotEqual(explorer.phasor.picture.value, old_image)
                    explorer.phasor.coordinates['x'].value += 0.2
                    self.assertNotAlmostEqual(abs(explorer.phasor.plot.phasor - initial), 0)
                    # Hidden views pick up changed wave parameters when opened again.
                    explorer.views.selected_index = 0
                    explorer.wavelength.value = 3.0
                    if dimensions == 2:
                        explorer.direction.value = -60
                    elif dimensions == 3:
                        explorer.polar.value = 70
                        explorer.azimuth.value = -45
                    explorer.views.selected_index = 1
                    panel = explorer.phasor
                    expected = scalar_plane_wave(panel.position, panel.k_vector,
                                                 phase=-2*np.pi*explorer.fraction)
                    assert_allclose(panel.plot.real_value, expected, atol=1e-14)
                    self.assertTrue(bytes(panel.picture.value).startswith(b'\x89PNG'))
                    # Each spatial sample gets the same time rotation.
                    explorer.views.selected_index = 2
                    grid = explorer.spatial_phasors
                    fixed = [values.copy() for values in grid.plot.phasors]
                    explorer.time.value = 30
                    for phasor, value in zip(fixed, grid.plot.values):
                        assert_allclose(value, -phasor, atol=1e-14)
                    grid.include_time.value = False
                    frozen = bytes(grid.picture.value)
                    explorer.time.value = 45
                    self.assertEqual(bytes(grid.picture.value), frozen)
                    for phasor, value in zip(fixed, grid.plot.values):
                        assert_allclose(value, phasor, atol=1e-14)
                    # Opening a previously hidden grid must show the latest direction.
                    explorer.views.selected_index = 0
                    explorer.wavelength.value = 2.5
                    explorer.views.selected_index = 2
                    for positions, values in zip(grid.plot.positions, grid.plot.values):
                        assert_allclose(values, plane_wave_phasor(positions, grid.k_vector))
                finally:
                    explorer.close()

    def test_close_releases_only_owned_widgets_and_detaches_callbacks(self):
        # The registry holds live widget comms, including layouts and styles.
        from ipywidgets.widgets.widget import _instances
        from wave_optics.widgets import PlaneWaveExplorer, ScalarPlaneWave3D

        for dimensions in (1, 2, 3):
            with self.subTest(dimensions=dimensions):
                baseline = set(_instances)
                for _ in range(2):
                    explorer = (ScalarPlaneWave3D() if dimensions == 3
                                else PlaneWaveExplorer(dimensions))
                    try:
                        explorer.play.playing = True
                        figures = [weakref.ref(panel.plot.figure) for panel in
                                   (explorer, explorer.phasor, explorer.spatial_phasors)]
                        self.assertGreater(len(_instances), len(baseline))
                        explorer.close()
                        explorer.close()  # Closing twice must be harmless.
                        self.assertFalse(explorer.play.playing)
                        self.assertIsNone(explorer.time_link.comm)
                        self.assertEqual(set(_instances), baseline)

                        # Retained controls must not redraw a disposed figure.
                        explorer.time.value = 15
                        explorer.wavelength.value = 3
                        explorer.views.selected_index = 1
                        explorer.phasor.coordinates['x'].value += 0.1
                        explorer.spatial_phasors.include_time.value = False
                        explorer._restart_control.click()
                        self.assertEqual(explorer.time.value, 15)
                        # A callback already queued at disposal must also be inert.
                        explorer._update()
                        explorer._change_view({'new': 1})
                        explorer._restart(None)
                        explorer.phasor._redraw()
                        explorer.spatial_phasors._redraw()
                        for panel in (explorer, explorer.phasor, explorer.spatial_phasors):
                            self.assertEqual(bytes(panel.picture.value), b'')
                            self.assertIsNone(panel.plot)
                        self.assertIsNone(explorer.figure)
                        gc.collect()
                        self.assertTrue(all(figure() is None for figure in figures))
                    finally:
                        explorer.close()

    def test_notebook_cells_can_be_replaced_without_affecting_other_explorers(self):
        from ipywidgets.widgets.widget import _instances
        from wave_optics.widgets import (
            PlaneWaveExplorer, ScalarPlaneWave3D, close_wave_explorer,
        )

        notebook = json.loads((Path(__file__).resolve().parents[2]
                              / 'supplement/1_Plane_Waves_and_Wavevectors_Original.ipynb').read_text(encoding='utf-8'))
        names = ('wave_1d', 'wave_2d', 'wave_3d')
        cells = {}
        for cell in notebook['cells']:
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'code':
                for name in names:
                    if source.startswith(f"if '{name}' in globals():"):
                        cells[name] = compile(source, f'notebook:{name}', 'exec')
        self.assertEqual(set(cells), set(names))
        namespace = dict(PlaneWaveExplorer=PlaneWaveExplorer,
                         ScalarPlaneWave3D=ScalarPlaneWave3D,
                         close_wave_explorer=close_wave_explorer, display=lambda _: None)
        baseline = set(_instances)
        try:
            for code in cells.values():
                exec(code, namespace)
            live_count = len(_instances)
            for _ in range(2):
                for name, code in cells.items():
                    previous = namespace[name]
                    previous.play.playing = True
                    neighbours = [namespace[key] for key in names if key != name]
                    for neighbour in neighbours:
                        neighbour.play.playing = True
                    neighbour_comms = [item.ui.model_id for item in neighbours]
                    neighbour_images = [bytes(item.picture.value) for item in neighbours]
                    exec(code, namespace)
                    self.assertFalse(previous.play.playing)
                    self.assertIsNone(previous.ui.comm)
                    self.assertIsNone(previous.time_link.comm)
                    self.assertEqual(len(_instances), live_count)
                    self.assertEqual([item.ui.model_id for item in neighbours], neighbour_comms)
                    self.assertEqual([bytes(item.picture.value) for item in neighbours], neighbour_images)
                    for neighbour in neighbours:
                        self.assertTrue(neighbour.play.playing)
                        image_before = bytes(neighbour.picture.value)
                        neighbour.time.value = (neighbour.time.value + 7) % neighbour.frames_per_period
                        self.assertNotEqual(bytes(neighbour.picture.value), image_before)
        finally:
            for name in names:
                if name in namespace:
                    namespace[name].close()
        self.assertEqual(set(_instances), baseline)

    def test_notebook_recovers_cached_classes_and_pre_close_instances(self):
        import importlib
        from ipywidgets.widgets.widget import _instances
        import wave_optics.widgets as backend

        notebook = json.loads((Path(__file__).resolve().parents[2]
                              / 'supplement/1_Plane_Waves_and_Wavevectors_Original.ipynb').read_text(encoding='utf-8'))
        code = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']
                if cell['cell_type'] == 'code'}
        names = ('wave_1d', 'wave_2d', 'wave_3d')
        builders = ('6cdb2192', '27bee8ce', 'dbcd7e3b')
        baseline = set(_instances)
        namespace = {}
        previous = []
        try:
            # Reproduce pre-cleanup instances: no base class providing close(),
            # and neither of the callback bookkeeping attributes added later.
            legacy_classes = []
            for current in (backend.PlaneWaveExplorer, backend.ScalarPlaneWave3D):
                members = {key: value for key, value in vars(current).items()
                           if key not in ('__dict__', '__weakref__')}
                legacy_classes.append(type(current.__name__, (), members))
            for name, dimensions in zip(names, (1, 2, 3)):
                explorer = (legacy_classes[1]() if dimensions == 3
                            else legacy_classes[0](dimensions))
                namespace[name] = explorer
                previous.append(explorer)
                del explorer._controls, explorer._restart_control
                self.assertFalse(hasattr(explorer, 'close'))
                explorer.play.playing = True

            # A plain import would reuse these cached definitions. The actual
            # setup cell must load the current definitions from disk instead.
            backend.PlaneWaveExplorer, backend.ScalarPlaneWave3D = legacy_classes
            namespace.update(PlaneWaveExplorer=legacy_classes[0],
                             ScalarPlaneWave3D=legacy_classes[1])
            population = len(_instances)
            exec(code['288c5c98'], namespace)
            self.assertTrue(callable(namespace['PlaneWaveExplorer'].close))
            self.assertTrue(callable(namespace['ScalarPlaneWave3D'].close))
            namespace['display'] = lambda _: None
            for name, builder, old in zip(names, builders, previous):
                others = [namespace[key] for key in names if key != name]
                other_ids = [item.ui.model_id for item in others]
                old_time = old.time.value
                exec(code[builder], namespace)
                # Notebook one teaches physical waves and their wavevectors;
                # its explorers omit the optional complex-plane panels.
                self.assertLess(len(_instances), population)
                population = len(_instances)
                self.assertIsNone(namespace[name].phasor)
                self.assertEqual([item.ui.model_id for item in others], other_ids)
                self.assertFalse(old.play.playing)
                self.assertIsNone(old.ui.comm)
                self.assertIsNone(old.time_link.comm)
                self.assertIsNone(old.figure)
                for control in old._controls:
                    self.assertNotIn(old._update,
                                     control._trait_notifiers.get('value', {}).get('change', []))
                namespace['close_wave_explorer'](old)  # Legacy path is idempotent too.
                old.time.value = old_time + 1
                old.phasor.coordinates['x'].value += .1
                old.spatial_phasors.include_time.value = False
                old.views.selected_index = 1
                old._restart_control.click()
                self.assertEqual(old.time.value, old_time + 1)
                self.assertEqual(bytes(old.picture.value), b'')

            # Reloading once more must also allow current live objects to close.
            exec(code['288c5c98'], namespace)
            namespace['display'] = lambda _: None
            for builder in builders:
                exec(code[builder], namespace)
            self.assertEqual(len(_instances), population)
        finally:
            importlib.reload(backend)
            for explorer in previous + [namespace[name] for name in names if name in namespace]:
                backend.close_wave_explorer(explorer)
        self.assertEqual(set(_instances), baseline)

    def test_notebook_setup_refreshes_dependencies_before_widgets(self):
        import importlib
        from ipywidgets.widgets.widget import _instances

        modules = [importlib.import_module(name) for name in (
            'wave_optics.waves', 'wave_optics.diffraction', 'wave_optics.coordinates',
            'wave_optics.plotting', 'wave_optics.widgets', 'wave_optics')]
        waves, diffraction, coordinates, plotting, backend, package = modules
        notebook = json.loads((Path(__file__).resolve().parents[2]
                              / 'supplement/1_Plane_Waves_and_Wavevectors_Original.ipynb').read_text(encoding='utf-8'))
        code = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']
                if cell['cell_type'] == 'code'}
        baseline = set(_instances)
        namespace = {}
        neighbour = backend.ScalarPlaneWave3D()
        try:
            neighbour.play.playing = True
            neighbour_id = neighbour.ui.model_id
            original_image = bytes(neighbour.picture.value)
            population = set(_instances)

            # Reproduce the reported partial reload: the cached plotting module
            # predates functions now imported by widgets.py on disk.
            del plotting.coordinate_figure, plotting.coordinate_readout
            with self.assertRaisesRegex(ImportError, 'coordinate_figure'):
                importlib.reload(backend)

            # Also model dependencies and package exports from earlier versions.
            # Reloading plotting alone would fail against these cached modules.
            del waves.plane_wave_phasor
            del diffraction.sphere_residuals
            del coordinates.wavevector_coordinates
            del backend.close_wave_explorer
            del package.plane_wave_phasor
            exec(code['288c5c98'], namespace)

            self.assertIs(backend.coordinate_figure, plotting.coordinate_figure)
            self.assertIs(backend.coordinate_readout, plotting.coordinate_readout)
            self.assertIs(backend.PlaneWavePlot, plotting.PlaneWavePlot)
            self.assertIs(plotting.plane_wave_phasor, waves.plane_wave_phasor)
            self.assertIs(plotting.sphere_residuals, diffraction.sphere_residuals)
            self.assertIs(diffraction.direction_cosines, waves.direction_cosines)
            self.assertIs(coordinates.wavenumber, waves.wavenumber)
            self.assertIs(backend.wavevector_coordinates, coordinates.wavevector_coordinates)
            self.assertIs(package.plane_wave_phasor, waves.plane_wave_phasor)
            self.assertIs(package.wavevector_coordinates, coordinates.wavevector_coordinates)
            self.assertTrue(callable(namespace['close_wave_explorer']))
            self.assertEqual(set(_instances), population)
            self.assertEqual(neighbour.ui.model_id, neighbour_id)
            self.assertTrue(neighbour.play.playing)
            self.assertEqual(bytes(neighbour.picture.value), original_image)
            neighbour.time.value = 7
            self.assertNotEqual(bytes(neighbour.picture.value), original_image)

            namespace['display'] = lambda _: None
            exec(code['6cdb2192'], namespace)
            self.assertIsNone(namespace['wave_1d'].views)
            namespace['wave_1d'].time.value = 15
            self.assertTrue(bytes(namespace['wave_1d'].picture.value).startswith(b'\x89PNG'))
            self.assertTrue(neighbour.play.playing)
        finally:
            for module in modules:
                importlib.reload(module)
            backend.close_wave_explorer(namespace.get('wave_1d'))
            neighbour.close()
        self.assertEqual(set(_instances), baseline)

    def test_wavelength_change_preserves_normalized_time_and_rescales_physical_time(self):
        from wave_optics.widgets import PlaneWaveExplorer, ScalarPlaneWave3D

        for dimensions in (1, 2, 3):
            with self.subTest(dimensions=dimensions):
                explorer = (ScalarPlaneWave3D() if dimensions == 3
                            else PlaneWaveExplorer(dimensions))
                try:
                    self.assertEqual(explorer.c, 1)
                    explorer.views.selected_index = 1
                    for coordinate in explorer.phasor.coordinates.values():
                        coordinate.value = 0
                    explorer.time.value = explorer.frames_per_period // 4
                    playback = (explorer.play.interval, explorer.play.max, explorer.play.step)
                    for wavelength in (1.0, 3.0):
                        explorer.wavelength.value = wavelength
                        self.assertEqual(explorer.fraction, 0.25)
                        self.assertEqual(explorer.period, wavelength / explorer.c)
                        self.assertEqual((explorer.play.interval, explorer.play.max,
                                          explorer.play.step), playback)
                        self.assertIn(f'T = {wavelength:.2f}; t = {wavelength/4:.2f}',
                                      explorer.readout.value)
                        # A quarter-cycle at the origin is the same field value
                        # despite the changed physical period and frequency.
                        self.assertAlmostEqual(explorer.phasor.plot.real_value, 0)
                        self.assertAlmostEqual(explorer.plot.c * explorer.k *
                                               (explorer.fraction * explorer.period), np.pi/2)
                finally:
                    explorer.close()


class SpatialPhasorGridTests(unittest.TestCase):
    def test_equal_phase_rows_and_columns_in_2d(self):
        from wave_optics.plotting import SpatialPhasorGridPlot

        grid = SpatialPhasorGridPlot(2)
        grid.render([0, np.pi], include_time=False)
        values = grid.values[0].reshape(7, 7)
        assert_allclose(values, np.broadcast_to(values[:, :1], values.shape))
        self.assertGreater(abs(values[1, 0] - values[0, 0]), 0.1)
        grid.render([np.pi, 0], include_time=False)
        values = grid.values[0].reshape(7, 7)
        assert_allclose(values, np.broadcast_to(values[:1, :], values.shape))

    def test_3d_slices_sample_one_complex_field(self):
        from wave_optics.plotting import SpatialPhasorGridPlot

        grid = SpatialPhasorGridPlot(3)
        grid.render([0, np.pi, 0], include_time=False)
        self.assertEqual(sum(len(points) for points in grid.positions), 147)
        for values, expected in zip(grid.values, [-1j, 1, 1j]):
            assert_allclose(values, expected, atol=1e-14)
        vector = wavevector(2, 50, 30)
        grid.render(vector, fraction=0.2)
        for points, values in zip(grid.positions, grid.values):
            assert_allclose(np.abs(values), 1)
            assert_allclose(values.real, scalar_plane_wave(points, vector, phase=-0.4*np.pi), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
