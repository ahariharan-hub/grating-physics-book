# Worked-out Python examples

Four independent numerical implementations accompany each teaching notebook.
Run a whole file or its `# %%` sections in order. Each file contains inputs,
setup, the calculation itself, a numerical check, graphics and a summary.
The `_common.py` helper only locates the backend and manages plotting.

Install the requirements and keep the downloaded folder structure together.

<a id="chapter-0-cutoffs"></a>
## Chapter 0 numerical cutoffs

The physical-scattering worked scripts for 0.5-0.7 start with `M = 8`
(17 harmonics); the corresponding teaching notebooks use `M = 24`
(49 harmonics). Physical material and illumination defaults agree, but these
are different numerical truncations. The notebooks display only orders -7
through +7 while reconstructing fields and calculating power with all 49.

To reproduce notebook values, set `M = 24` in a script and keep sufficient
sampling (more than `2*M` endpoint-excluded samples per period for the retained
finite sum). A projection or FFT check verifies that script's own truncated
field; it does not certify convergence of the scattering solve. Compare larger
cutoffs for the coefficient or field region you need. Neither default is changed.

For example, the default exit coefficient C_1 differs by about 0.0557% between
M=8 and M=24; a successful projection check alone will not expose that difference.
The independent ideal-mask coefficient exercise has no material-solver cutoff.

<a id="notebook-0-1"></a>
## Notebook 0.1

- [0_1_01_medium_wavelength.py](chapter_0/0_1_01_medium_wavelength.py) — Convert vacuum wavelength to medium wavelength.
- [0_1_02_phasor_and_real_field.py](chapter_0/0_1_02_phasor_and_real_field.py) — Recover a real field from a complex phasor.
- [0_1_03_helmholtz_trial_residual.py](chapter_0/0_1_03_helmholtz_trial_residual.py) — Test an allowed and an incorrect spatial wavenumber.
- [0_1_04_finite_difference_helmholtz.py](chapter_0/0_1_04_finite_difference_helmholtz.py) — Check the convergence of a numerical second derivative.

<a id="notebook-0-2"></a>
## Notebook 0.2

- [0_2_01_travelling_and_standing.py](chapter_0/0_2_01_travelling_and_standing.py) — Build a standing wave from opposite travelling waves.
- [0_2_02_radial_helmholtz.py](chapter_0/0_2_02_radial_helmholtz.py) — Verify spherical and cylindrical radial solutions.
- [0_2_03_bessel_cone_sum.py](chapter_0/0_2_03_bessel_cone_sum.py) — Construct a Bessel beam by angular quadrature.
- [0_2_04_paraxial_gaussian_power.py](chapter_0/0_2_04_paraxial_gaussian_power.py) — Implement Gaussian spreading and check transverse power.

<a id="notebook-0-3"></a>
## Notebook 0.3

- [0_3_01_superposition_residual.py](chapter_0/0_3_01_superposition_residual.py) — Check linear superposition with complex amplitudes.
- [0_3_02_boundary_nonuniqueness.py](chapter_0/0_3_02_boundary_nonuniqueness.py) — Compare fields that share the same boundary value.
- [0_3_03_slit_phasor_quadrature.py](chapter_0/0_3_03_slit_phasor_quadrature.py) — Integrate a slit far field and recover the sinc pattern.
- [0_3_04_periodic_boundary_field.py](chapter_0/0_3_04_periodic_boundary_field.py) — Prescribe an illuminated grating boundary and test its translation.

<a id="notebook-0-4"></a>
## Notebook 0.4

- [0_4_01_fft_reconstruction_phase.py](chapter_0/0_4_01_fft_reconstruction_phase.py) — Reconstruct a complex field with and without spectral phase.
- [0_4_02_component_multiplier.py](chapter_0/0_4_02_component_multiplier.py) — Choose the outgoing square-root branch for each component.
- [0_4_03_asm_slit_from_scratch.py](chapter_0/0_4_03_asm_slit_from_scratch.py) — Propagate a sampled slit by the angular-spectrum method.
- [0_4_04_sampling_and_padding.py](chapter_0/0_4_04_sampling_and_padding.py) — Separate FFT resolution from window-padding effects.

<a id="notebook-0-5"></a>
## Notebook 0.5

- [0_5_01_material_geometry.py](chapter_0/0_5_01_material_geometry.py) — Discretize an absorbing grating geometry.
- [0_5_02_solve_and_power_balance.py](chapter_0/0_5_02_solve_and_power_balance.py) — Solve a physical grating and inspect its power budget.
- [0_5_03_basis_convergence.py](chapter_0/0_5_03_basis_convergence.py) — Compare a propagating-only cutoff with larger Fourier bases.
- [0_5_04_explicit_harmonic_sum.py](chapter_0/0_5_04_explicit_harmonic_sum.py) — Rebuild the solved field with an explicit Fourier sum.

<a id="notebook-0-6"></a>
## Notebook 0.6

- [0_6_01_project_complex_coefficient.py](chapter_0/0_6_01_project_complex_coefficient.py) — Recover one coefficient by numerical projection.
- [0_6_02_reference_plane_evolution.py](chapter_0/0_6_02_reference_plane_evolution.py) — Advance complex coefficients to a new reference plane.
- [0_6_03_flux_weights.py](chapter_0/0_6_03_flux_weights.py) — Distinguish coefficient magnitude squared from normal power.
- [0_6_04_parseval_field_and_orders.py](chapter_0/0_6_04_parseval_field_and_orders.py) — Check Parseval and the importance of complex order phase.

<a id="notebook-0-7"></a>
## Notebook 0.7

- [0_7_01_ideal_mask_coefficients.py](chapter_0/0_7_01_ideal_mask_coefficients.py) — Integrate ideal binary-mask Fourier coefficients.
- [0_7_02_common_phase_reference.py](chapter_0/0_7_02_common_phase_reference.py) — Compare the physical exit field with a thin mask propagated from z=0.
- [0_7_03_absorber_thickness_scan.py](chapter_0/0_7_03_absorber_thickness_scan.py) — Test whether greater absorber thickness approaches an ideal mask.
- [0_7_04_compare_at_common_depth.py](chapter_0/0_7_04_compare_at_common_depth.py) — Propagate physical and ideal coefficients to the same observation plane.

<a id="notebook-1-1"></a>
## Notebook 1.1

- [1_1_01_time_oscillation.py](chapter_1/1_1_01_time_oscillation.py) — Sample one optical oscillation in time.
- [1_1_02_moving_snapshot.py](chapter_1/1_1_02_moving_snapshot.py) — Compute travelling-wave snapshots in both directions.
- [1_1_03_recover_complex_phase.py](chapter_1/1_1_03_recover_complex_phase.py) — Recover amplitude and phase from quadrature samples.
- [1_1_04_fft_temporal_frequency.py](chapter_1/1_1_04_fft_temporal_frequency.py) — Estimate oscillation frequency with a sampled Fourier transform.

<a id="notebook-1-2"></a>
## Notebook 1.2

- [1_2_01_finite_window_power.py](chapter_1/1_2_01_finite_window_power.py) — Integrate uniform and Gaussian illumination over finite windows.
- [1_2_02_beam_angular_spectrum.py](chapter_1/1_2_02_beam_angular_spectrum.py) — Calculate the angular spectrum of a finite Gaussian beam.
- [1_2_03_finite_beam_propagation.py](chapter_1/1_2_03_finite_beam_propagation.py) — Compare finite-beam propagation with a uniform plane wave.
- [1_2_04_beam_local_phase_gradient.py](chapter_1/1_2_04_beam_local_phase_gradient.py) — Estimate local directions from a paraxial Gaussian phase.

<a id="notebook-1-3"></a>
## Notebook 1.3

- [1_3_01_wavefront_map.py](chapter_1/1_3_01_wavefront_map.py) — Draw constant-phase lines and the perpendicular wavevector.
- [1_3_02_directional_phase_change.py](chapter_1/1_3_02_directional_phase_change.py) — Measure phase change along and across a wavevector.
- [1_3_03_retrieve_phase_gradient.py](chapter_1/1_3_03_retrieve_phase_gradient.py) — Recover a wavevector from sampled complex phase.
- [1_3_04_fft_wavevector.py](chapter_1/1_3_04_fft_wavevector.py) — Find a plane-wave direction using a two-dimensional FFT.

<a id="notebook-1-4"></a>
## Notebook 1.4

- [1_4_01_direction_cosines.py](chapter_1/1_4_01_direction_cosines.py) — Construct a 3D wavevector from polar and azimuthal angles.
- [1_4_02_three_dimensional_samples.py](chapter_1/1_4_02_three_dimensional_samples.py) — Evaluate a 3D field using matrix multiplication and explicit loops.
- [1_4_03_recover_three_components.py](chapter_1/1_4_03_recover_three_components.py) — Retrieve all three wavevector components from phase samples.
- [1_4_04_rotate_coordinates.py](chapter_1/1_4_04_rotate_coordinates.py) — Rotate the field geometry without changing its physical phase.

<a id="notebook-2-1"></a>
## Notebook 2.1

- [2_1_01_single_slit_integral.py](chapter_2/2_1_01_single_slit_integral.py) — Resolve a single opening by direct far-field quadrature.
- [2_1_02_finite_array_sum.py](chapter_2/2_1_02_finite_array_sum.py) — Sum equally driven openings and recover the finite-array factor.
- [2_1_03_slit_envelope_times_array.py](chapter_2/2_1_03_slit_envelope_times_array.py) — Combine opening width with the finite-array phase sum.
- [2_1_04_finite_mask_fft.py](chapter_2/2_1_04_finite_mask_fft.py) — Compute a finite grating spectrum from sampled real-space openings.

<a id="notebook-2-2"></a>
## Notebook 2.2

- [2_2_01_neighbouring_opening_phases.py](chapter_2/2_2_01_neighbouring_opening_phases.py) — Evaluate incident complex phase at neighbouring openings.
- [2_2_02_mask_times_illumination.py](chapter_2/2_2_02_mask_times_illumination.py) — Multiply a binary transmission by a tilted complex field.
- [2_2_03_bloch_translation.py](chapter_2/2_2_03_bloch_translation.py) — Verify the Bloch translation law for illuminated openings.
- [2_2_04_illumination_spectral_shift.py](chapter_2/2_2_04_illumination_spectral_shift.py) — Measure the spectral shift caused by an incident phase ramp.

<a id="notebook-2-3"></a>
## Notebook 2.3

- [2_3_01_enumerate_order_candidates.py](chapter_2/2_3_01_enumerate_order_candidates.py) — Enumerate every retained order before classifying it.
- [2_3_02_angle_and_wavevector.py](chapter_2/2_3_02_angle_and_wavevector.py) — Recover the grating-angle law from vector components.
- [2_3_03_rayleigh_threshold.py](chapter_2/2_3_03_rayleigh_threshold.py) — Locate a diffraction-order cutoff in wavelength.
- [2_3_04_selected_outgoing_field.py](chapter_2/2_3_04_selected_outgoing_field.py) — Construct a selected outgoing wave without assigning an efficiency.

<a id="notebook-2-4"></a>
## Notebook 2.4

- [2_4_01_coordinate_conversions.py](chapter_2/2_4_01_coordinate_conversions.py) — Convert one set of orders between k, frequency and direction cosine.
- [2_4_02_sphere_and_plane.py](chapter_2/2_4_02_sphere_and_plane.py) — Intersect outgoing rays with a sphere and a detector plane.
- [2_4_03_spacing_is_coordinate_dependent.py](chapter_2/2_4_03_spacing_is_coordinate_dependent.py) — Compare uniform transverse spacing with nonuniform angle spacing.
- [2_4_04_ewald_harvey_coordinates.py](chapter_2/2_4_04_ewald_harvey_coordinates.py) — Translate a direction-circle representation into scattering-vector coordinates.

<a id="notebook-3-1"></a>
## Notebook 3.1

- [3_1_01_phase_across_period.py](chapter_3/3_1_01_phase_across_period.py) — Calculate the incident phase difference across one pitch.
- [3_1_02_integer_translation_rule.py](chapter_3/3_1_02_integer_translation_rule.py) — Recover allowed transverse components from discrete translation.
- [3_1_03_fit_surface_phase_slope.py](chapter_3/3_1_03_fit_surface_phase_slope.py) — Estimate incident direction by unwrapping sampled phase.
- [3_1_04_modulo_phase_ambiguity.py](chapter_3/3_1_04_modulo_phase_ambiguity.py) — Show why one complex translation factor cannot count phase cycles.

<a id="notebook-3-2"></a>
## Notebook 3.2

- [3_2_01_matched_order_field.py](chapter_3/3_2_01_matched_order_field.py) — Build a selected-order wave and check the surface seam.
- [3_2_02_noninteger_seam_error.py](chapter_3/3_2_02_noninteger_seam_error.py) — Measure phase mismatch for a noninteger trial order.
- [3_2_03_dispersion_after_matching.py](chapter_3/3_2_03_dispersion_after_matching.py) — Complete the longitudinal component after tangential matching.
- [3_2_04_bloch_fourier_projection.py](chapter_3/3_2_04_bloch_fourier_projection.py) — Remove the incident carrier before projecting order labels.

<a id="notebook-3-3"></a>
## Notebook 3.3

- [3_3_01_projected_path_difference.py](chapter_3/3_3_01_projected_path_difference.py) — Convert incident and outgoing path projections into phase.
- [3_3_02_two_opening_path_sum.py](chapter_3/3_3_02_two_opening_path_sum.py) — Add two path contributions as complex amplitudes.
- [3_3_03_invert_path_projection.py](chapter_3/3_3_03_invert_path_projection.py) — Recover a diffraction angle from an integer path increment.
- [3_3_04_many_path_interference.py](chapter_3/3_3_04_many_path_interference.py) — Build grating maxima from a finite set of path differences.

<a id="notebook-4-1"></a>
## Notebook 4.1

- [4_1_01_complex_longitudinal_root.py](chapter_4/4_1_01_complex_longitudinal_root.py) — Compute propagating and evanescent longitudinal components.
- [4_1_02_construct_decaying_field.py](chapter_4/4_1_02_construct_decaying_field.py) — Construct and sample an evanescent field in space.
- [4_1_03_evanescent_helmholtz_residual.py](chapter_4/4_1_03_evanescent_helmholtz_residual.py) — Verify that a decaying oscillatory field still satisfies Helmholtz.
- [4_1_04_mixed_spectrum_propagation.py](chapter_4/4_1_04_mixed_spectrum_propagation.py) — Propagate a boundary containing both real and imaginary longitudinal components.

<a id="notebook-4-2"></a>
## Notebook 4.2

- [4_2_01_decay_lengths.py](chapter_4/4_2_01_decay_lengths.py) — Distinguish amplitude decay length from intensity decay length.
- [4_2_02_grazing_transition.py](chapter_4/4_2_02_grazing_transition.py) — Follow one order continuously through grazing.
- [4_2_03_reactive_scalar_flux.py](chapter_4/4_2_03_reactive_scalar_flux.py) — Separate real normal scalar flux from a reactive component.
- [4_2_04_period_averaged_flux.py](chapter_4/4_2_04_period_averaged_flux.py) — Check constant average normal flux while the near-field intensity changes.

<a id="notebook-5-1"></a>
## Notebook 5.1

- [5_1_01_incident_three_components.py](chapter_5/5_1_01_incident_three_components.py) — Complete an incident wavevector with a groove-parallel component.
- [5_1_02_two_translation_symmetries.py](chapter_5/5_1_02_two_translation_symmetries.py) — Test discrete x translation and continuous y translation.
- [5_1_03_conical_order_directions.py](chapter_5/5_1_03_conical_order_directions.py) — Construct conical outgoing directions and check their common cone angle.
- [5_1_04_groove_component_cutoff.py](chapter_5/5_1_04_groove_component_cutoff.py) — Track the loss of propagating orders as groove-parallel incidence increases.

<a id="notebook-5-2"></a>
## Notebook 5.2

- [5_2_01_one_order_three_scales.py](chapter_5/5_2_01_one_order_three_scales.py) — Represent identical orders in k, spatial frequency and direction coordinates.
- [5_2_02_spatial_frequency_disk.py](chapter_5/5_2_02_spatial_frequency_disk.py) — Calculate the propagating spatial-frequency disk in different media.
- [5_2_03_sphere_projection_plane_intersection.py](chapter_5/5_2_03_sphere_projection_plane_intersection.py) — Compare 3D sphere projections and detector-plane intersections.
- [5_2_04_translated_sphere_residuals.py](chapter_5/5_2_04_translated_sphere_residuals.py) — Verify Harvey and Ewald sphere equations for the same 3D orders.

<a id="notebook-5-3"></a>
## Notebook 5.3

- [5_3_01_reciprocal_lattice_basis.py](chapter_5/5_3_01_reciprocal_lattice_basis.py) — Construct a reciprocal basis for a nonorthogonal real-space lattice.
- [5_3_02_two_dimensional_order_lattice.py](chapter_5/5_3_02_two_dimensional_order_lattice.py) — Classify a two-integer order lattice against the output propagation disk.
- [5_3_03_rotate_lattice.py](chapter_5/5_3_03_rotate_lattice.py) — Rotate direct and reciprocal lattices consistently.
- [5_3_04_evanescent_lattice_order.py](chapter_5/5_3_04_evanescent_lattice_order.py) — Propagate one evanescent two-dimensional order.

<a id="notebook-5-4"></a>
## Notebook 5.4

- [5_4_01_sigma_normalization.py](chapter_5/5_4_01_sigma_normalization.py) — Normalize diffraction wavevectors using objective numerical aperture.
- [5_4_02_collect_orders_with_na.py](chapter_5/5_4_02_collect_orders_with_na.py) — Apply a circular angular collection aperture to order candidates.
- [5_4_03_source_position_and_size.py](chapter_5/5_4_03_source_position_and_size.py) — Distinguish moving a source point from widening a source disk.
- [5_4_04_sine_mapping_vs_plane.py](chapter_5/5_4_04_sine_mapping_vs_plane.py) — Compare an assumed sine-condition pupil map with a flat propagation plane.

<a id="notebook-6-1"></a>
## Notebook 6.1

- [6_1_01_finite_exposure_average.py](chapter_6/6_1_01_finite_exposure_average.py) — Integrate the real-field square over a finite detector exposure.
- [6_1_02_amplitude_intensity_scaling.py](chapter_6/6_1_02_amplitude_intensity_scaling.py) — Verify the square-law relation with a fixed reference.
- [6_1_03_time_quadrature_convergence.py](chapter_6/6_1_03_time_quadrature_convergence.py) — Check detector time-sampling convergence.
- [6_1_04_intensity_loses_phase.py](chapter_6/6_1_04_intensity_loses_phase.py) — Show that phase-distinct fields can have equal recorded intensity.

<a id="notebook-6-2"></a>
## Notebook 6.2

- [6_2_01_three_detector_points.py](chapter_6/6_2_01_three_detector_points.py) — Sample a tilted order at three detector locations.
- [6_2_02_detector_phase_gradient.py](chapter_6/6_2_02_detector_phase_gradient.py) — Measure spatial phase while detector intensity remains uniform.
- [6_2_03_prescribed_detector_envelope.py](chapter_6/6_2_03_prescribed_detector_envelope.py) — Compare a uniform order with a finite detector-plane envelope.
- [6_2_04_intensity_and_normal_flux.py](chapter_6/6_2_04_intensity_and_normal_flux.py) — Compare field intensity with flux normal to a detector.

<a id="notebook-7-1"></a>
## Notebook 7.1

- [7_1_01_add_fields_before_squaring.py](chapter_7/7_1_01_add_fields_before_squaring.py) — Evaluate the cross term in two-wave addition.
- [7_1_02_ensemble_mutual_coherence.py](chapter_7/7_1_02_ensemble_mutual_coherence.py) — Compute mutual coherence from a sampled phase ensemble.
- [7_1_03_visibility_balance_and_coherence.py](chapter_7/7_1_03_visibility_balance_and_coherence.py) — Separate amplitude imbalance from reduced mutual coherence.
- [7_1_04_eo_phase_translation.py](chapter_7/7_1_04_eo_phase_translation.py) — Translate fringes by rotating a complex correlation.

<a id="notebook-7-2"></a>
## Notebook 7.2

- [7_2_01_two_order_wavevectors.py](chapter_7/7_2_01_two_order_wavevectors.py) — Construct two grating-order vectors and verify their magnitudes.
- [7_2_02_fft_fringe_period.py](chapter_7/7_2_02_fft_fringe_period.py) — Synthesize grating-order fringes and measure their period with an FFT.
- [7_2_03_detector_depth_phase.py](chapter_7/7_2_03_detector_depth_phase.py) — Predict fringe displacement when the detector moves downstream.
- [7_2_04_tilted_conical_detector.py](chapter_7/7_2_04_tilted_conical_detector.py) — Project conical-order interference onto a tilted detector.

<a id="notebook-7-3"></a>
## Notebook 7.3

- [7_3_01_rotated_gaussian_implementation.py](chapter_7/7_3_01_rotated_gaussian_implementation.py) — Implement a Gaussian beam in its own rotated coordinate frame.
- [7_3_02_overlap_quadrature.py](chapter_7/7_3_02_overlap_quadrature.py) — Integrate detector-line overlap for two spreading order beams.
- [7_3_03_local_fringe_phase.py](chapter_7/7_3_03_local_fringe_phase.py) — Calculate finite-beam fringe phase and its local spatial frequency.
- [7_3_04_local_visibility_and_overlap.py](chapter_7/7_3_04_local_visibility_and_overlap.py) — Distinguish local fringe visibility from integrated beam overlap.

<a id="notebook-7-4"></a>
## Notebook 7.4

- [7_4_01_fixed_directions_variable_coefficients.py](chapter_7/7_4_01_fixed_directions_variable_coefficients.py) — Hold order directions fixed and vary complex coefficients.
- [7_4_02_mask_translation_phase.py](chapter_7/7_4_02_mask_translation_phase.py) — Derive complex mask coefficients and their translation phase.
- [7_4_03_physical_field_projection.py](chapter_7/7_4_03_physical_field_projection.py) — Recover solved complex scattering amplitudes from sampled exit fields.
- [7_4_04_coefficients_vs_efficiency.py](chapter_7/7_4_04_coefficients_vs_efficiency.py) — Apply the longitudinal flux factor for unequal exterior indices.

<a id="notebook-8-1"></a>
## Notebook 8.1

- [8_1_01_green_helmholtz_residual.py](chapter_8/8_1_01_green_helmholtz_residual.py) — Check the outgoing point Green response away from its source.
- [8_1_02_point_to_point_response.py](chapter_8/8_1_02_point_to_point_response.py) — Calculate distance-dependent Green amplitude and phase.
- [8_1_03_normal_derivative.py](chapter_8/8_1_03_normal_derivative.py) — Implement and finite-difference-check the Green normal derivative.
- [8_1_04_closed_boundary_green_identity.py](chapter_8/8_1_04_closed_boundary_green_identity.py) — Reconstruct a source-free interior field from a closed boundary.

<a id="notebook-8-2"></a>
## Notebook 8.2

- [8_2_01_rectangular_rs_integral.py](chapter_8/8_2_01_rectangular_rs_integral.py) — Integrate a rectangular aperture using explicit two-dimensional Simpson weights.
- [8_2_02_rs_quadrature_convergence.py](chapter_8/8_2_02_rs_quadrature_convergence.py) — Refine the aperture quadrature independently of observation sampling.
- [8_2_03_weyl_plane_wave_integral.py](chapter_8/8_2_03_weyl_plane_wave_integral.py) — Rebuild a spherical Green response from propagating and evanescent spectra.
- [8_2_04_rs_integral_vs_asm.py](chapter_8/8_2_04_rs_integral_vs_asm.py) — Compare a direct RS aperture integral with FFT angular-spectrum propagation.

<a id="notebook-8-3"></a>
## Notebook 8.3

- [8_3_01_two_dimensional_asm_fft.py](chapter_8/8_3_01_two_dimensional_asm_fft.py) — Implement two-dimensional ASM using FFT frequencies and the outgoing branch.
- [8_3_02_fresnel_integral_vs_fft.py](chapter_8/8_3_02_fresnel_integral_vs_fft.py) — Evaluate Fresnel propagation by a spatial integral and a Fourier multiplier.
- [8_3_03_fraunhofer_rectangle_fft.py](chapter_8/8_3_03_fraunhofer_rectangle_fft.py) — Recover rectangular-aperture Fraunhofer diffraction with physical Fourier scaling.
- [8_3_04_kernel_approximation_errors.py](chapter_8/8_3_04_kernel_approximation_errors.py) — Compare full RS I and approximate spatial kernels at common points.

<a id="notebook-9-1"></a>
## Notebook 9.1

- [9_1_01_mask_and_incident_field.py](chapter_9/9_1_01_mask_and_incident_field.py) — Construct a complex exit field from a prescribed mask and illumination.
- [9_1_02_amplitude_and_phase_masks.py](chapter_9/9_1_02_amplitude_and_phase_masks.py) — Compare two masks with different complex transmission but simple geometry.
- [9_1_03_bloch_carrier_and_fft.py](chapter_9/9_1_03_bloch_carrier_and_fft.py) — Separate periodic transmission coefficients from a tilted exit field.
- [9_1_04_physical_exit_reference.py](chapter_9/9_1_04_physical_exit_reference.py) — Extract a physical exit field and move its reference plane.

<a id="notebook-9-2"></a>
## Notebook 9.2

- [9_2_01_fft_origin_correction.py](chapter_9/9_2_01_fft_origin_correction.py) — Recover complex Fourier coefficients with the sampling-origin phase correction.
- [9_2_02_aliasing_collision.py](chapter_9/9_2_02_aliasing_collision.py) — Demonstrate that a perfect FFT roundtrip can hide an aliased field.
- [9_2_03_multiple_period_windows.py](chapter_9/9_2_03_multiple_period_windows.py) — Track FFT-bin changes when the window contains several periods.
- [9_2_04_two_dimensional_fft_coefficients.py](chapter_9/9_2_04_two_dimensional_fft_coefficients.py) — Recover a two-dimensional complex order lattice with FFT2.

<a id="notebook-9-3"></a>
## Notebook 9.3

- [9_3_01_thin_mask_at_object_plane.py](chapter_9/9_3_01_thin_mask_at_object_plane.py) — Prescribe the thin field at the object plane and compare its propagating projection.
- [9_3_02_thick_exit_to_object_plane.py](chapter_9/9_3_02_thick_exit_to_object_plane.py) — Bring the thick exit field back to the object plane.
- [9_3_03_compare_complex_orders.py](chapter_9/9_3_03_compare_complex_orders.py) — Compare complex thin and thick coefficients at the object plane.
- [9_3_04_common_observation_plane.py](chapter_9/9_3_04_common_observation_plane.py) — Propagate both predictions to a common observation plane and verify the thick replay.
