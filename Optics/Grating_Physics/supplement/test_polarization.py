"""Regression checks for transverse bases, fields, and order projections."""

from contextlib import redirect_stdout
import io
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import VKG


class PolarizationBasisTests(unittest.TestCase):
    def assert_transverse_basis(self, k, s, p):
        k = np.asarray(k, dtype=float)
        k = k / np.linalg.norm(k)
        basis = np.array([k, s, p])
        self.assertTrue(np.all(np.isfinite(basis)))
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.cross(s, p), -k, atol=1e-12)

    def test_normal_incidence_for_rotated_normals_and_parallel_references(self):
        directions = list(np.eye(3)) + list(-np.eye(3)) + [np.array([1., 2., 3.])]
        for k in directions:
            for sign in (1, -1):
                for reference in (None, k, -k):
                    with self.subTest(k=k, sign=sign, reference=reference):
                        s, p = VKG.compute_polarization_basis(k, sign * k, reference)
                        self.assert_transverse_basis(k, s, p)

    def test_default_normal_incidence_preserves_existing_basis(self):
        s, p = VKG.compute_polarization_basis([0, 0, -1], [0, 0, -1])
        np.testing.assert_allclose(s, [0, 1, 0])
        np.testing.assert_allclose(p, [-1, 0, 0])

    def test_nearly_parallel_reference_stays_transverse(self):
        k = np.array([1., 1e-8, 0.])
        s, p = VKG.compute_polarization_basis(k, k, [1, 0, 0])
        self.assert_transverse_basis(k, s, p)

    def test_near_normal_incidence_uses_stable_reference(self):
        k = np.array([1e-14, 0., -1.])
        s, p = VKG.compute_polarization_basis(k, [0, 0, -1])
        self.assert_transverse_basis(k, s, p)
        np.testing.assert_allclose(s, [0, 1, 0])

    def test_oblique_incidence_preserves_sign_convention(self):
        k = np.array([1., 0., -1.]) / np.sqrt(2)
        s, p = VKG.compute_polarization_basis(k, [0, 0, -1])
        self.assert_transverse_basis(k, s, p)
        np.testing.assert_allclose(s, [0, 1, 0])
        np.testing.assert_allclose(p, np.array([-1, 0, -1]) / np.sqrt(2))

    def test_incident_plot_handles_normal_propagation_along_y(self):
        fig = plt.figure()
        try:
            ax = fig.add_subplot(projection="3d")
            k, s, p, _ = VKG.plot_incident_plane_wave(
                ax, 90, 90, 1, n_hat=[0, 1, 0]
            )
            np.testing.assert_allclose(k, [0, 1, 0], atol=1e-12)
            self.assert_transverse_basis(k, s, p)
            fig.canvas.draw()
        finally:
            plt.close(fig)

    def test_order_basis_handles_parallel_reference_and_evanescent_order(self):
        orders = VKG.compute_diffraction_orders(
            [1, 0, 0], 193, 800, [0, 1, 0], [1, 0, 0], orders=[0, 5]
        )
        results = VKG.compute_order_polarization_bases(orders, [1, 0, 0], [1, 0, 0])
        self.assert_transverse_basis(
            results[0]["k_hat"], results[0]["s_hat"], results[0]["p_hat"]
        )
        self.assertIsNone(results[1]["s_hat"])
        self.assertIsNone(results[1]["p_hat"])
        self.assertNotIn("s_hat", orders[0])

    def test_field_projections_match_basis_matrices(self):
        cases = [
            ([0, 0, -1], [0, 0, -1], [1, 0, 0], None),
            ([1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]),
            ([0.5, -0.5, -np.sqrt(0.5)], [0, 0, -1], [1, 0, 0], None),
        ]
        amplitudes = np.array([1.0, 1.0j]) / np.sqrt(2)
        for k, n, g, reference in cases:
            with self.subTest(k=k):
                s, p = VKG.compute_polarization_basis(k, n, reference)
                orders = VKG.compute_diffraction_orders(k, 193, 800, g, n)
                orders = VKG.compute_order_polarization_bases(orders, n, reference)
                orders = VKG.compute_order_rotation_matrices(orders, s, p)
                results = VKG.transform_field_to_order_bases(
                    orders, s, p, *amplitudes
                )
                incident_field = amplitudes[0] * s + amplitudes[1] * p
                for result in results:
                    np.testing.assert_allclose(result["E_inc"], incident_field)
                    if not result["propagating"]:
                        self.assertIsNone(result["R_sp"])
                        for key in ("Es_proj", "Ep_proj", "E_sp_proj", "E_proj", "E_vec"):
                            self.assertIsNone(result[key])
                        continue
                    np.testing.assert_allclose(
                        result["E_sp_proj"], result["R_sp"] @ amplitudes,
                        atol=1e-12,
                    )
                    direction = result["k_hat"]
                    transverse_field = (
                        incident_field - direction * np.dot(direction, incident_field)
                    )
                    for key in ("E_proj", "E_vec"):
                        np.testing.assert_allclose(
                            result[key], transverse_field, atol=1e-12
                        )
                        np.testing.assert_allclose(
                            np.dot(direction, result[key]), 0, atol=1e-12
                        )
                    self.assertLessEqual(
                        np.linalg.norm(result["E_proj"]),
                        np.linalg.norm(incident_field) + 1e-12,
                    )
                    if result["order"] == 0:
                        np.testing.assert_allclose(result["R_sp"], np.eye(2), atol=1e-12)
                        np.testing.assert_allclose(result["E_sp_proj"], amplitudes, atol=1e-12)
                        np.testing.assert_allclose(result["E_vec"], incident_field, atol=1e-12)
                self.assertNotIn("E_inc", orders[0])
                self.assertNotIn("E_proj", orders[0])

    def test_projection_diagnostics_accept_grazing_and_negative_determinants(self):
        n = [0, 0, -1]
        cases = [
            ([0, 0, -1], 193, [0, 1, 2], [1.0, 0.0]),
            ([0.8, 0, -0.6], 193 / 1.6, [0, -1], [1.0, -0.28]),
        ]
        for k, period, order_numbers, expected in cases:
            with self.subTest(k=k):
                s, p = VKG.compute_polarization_basis(k, n)
                orders = VKG.compute_diffraction_orders(
                    k, 193, period, [1, 0, 0], n, orders=order_numbers
                )
                orders = VKG.compute_order_polarization_bases(orders, n)
                orders = VKG.compute_order_rotation_matrices(orders, s, p)
                with redirect_stdout(io.StringIO()):
                    diagnostics = VKG.check_order_rotation_matrices(orders)
                np.testing.assert_allclose(
                    [d["determinant"] for d in diagnostics], expected, atol=1e-12
                )
                for diagnostic in diagnostics:
                    self.assertLess(diagnostic["determinant_error"], 1e-12)
                    self.assertLess(diagnostic["singular_value_error"], 1e-12)

    def test_projection_diagnostics_detect_sign_and_magnitude_errors(self):
        k = [0, 0, -1]
        s, p = VKG.compute_polarization_basis(k, k)
        orders = VKG.compute_diffraction_orders(
            k, 193, 193, [1, 0, 0], k, orders=[0, 1]
        )
        orders = VKG.compute_order_polarization_bases(orders, k)
        orders = VKG.compute_order_rotation_matrices(orders, s, p)
        # An incorrect handedness keeps singular values but changes det(R).
        orders[0]["R_sp"][0, :] *= -1
        # At grazing, an incorrect scale keeps det(R) = 0 but changes the norm.
        orders[1]["R_sp"] *= 2
        with redirect_stdout(io.StringIO()):
            diagnostics = VKG.check_order_rotation_matrices(orders)
        self.assertAlmostEqual(diagnostics[0]["determinant_error"], 2.0)
        self.assertLess(diagnostics[0]["singular_value_error"], 1e-12)
        self.assertLess(diagnostics[1]["determinant_error"], 1e-12)
        self.assertAlmostEqual(diagnostics[1]["singular_value_error"], 1.0)


if __name__ == "__main__":
    unittest.main()
