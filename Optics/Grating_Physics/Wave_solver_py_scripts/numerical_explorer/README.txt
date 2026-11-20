NUMERICAL EXPLORERS

Interactive experiments backed by wave_optics: editable parameter blocks,
focused Matplotlib plots, sliders, Apply and Reset. There is no lesson-navigation
window. Edit the numbered example and its setup at the top of a chapter file,
then run that whole file with F5 in Spyder or with Python.

INSTALL AND RUN AFTER DOWNLOADING
1. Extract the entire ZIP; keep wave_optics and Wave_solver_py_scripts together.
2. Open a terminal in the extracted Grating_Physics folder.
3. Install: python -m pip install -r requirements.txt
4. Run: python Wave_solver_py_scripts/numerical_explorer/1_plane_waves.py

In Spyder, use the same Python environment into which the requirements were
installed. QtAgg opens a separate interactive plot window. Restart the kernel
if a retained inline/Agg backend prevents interactive controls. plot_backend=None
respects your existing Matplotlib backend. No Jupyter kernel is required.

NUMERICAL-EXPLORER NUMBERING
0_scalar_foundations.py        0.1-0.7: fields, boundary data, material/mask models
1_plane_waves.py               1.1-1.4: oscillation, finite beams, wavefronts, 3D
2_grating_orders.py            2.1-2.4: slits, illumination and allowed orders
3_phase_matching.py            3.1-3.3: phase cycles, continuity and path difference
4_evanescent_orders.py         4.1-4.2: evanescent fields, grazing and decay
5_diffraction_coordinates.py   5.1-5.4: conical orders, lattices and pupils
6_field_to_intensity.py        6.1-6.2: averaging and a single-order detector
7_interference.py              7.1-7.4: coherence, order interference and overlap
8_field_propagation.py         8.1-8.3: Green response, RS, Weyl, kernels and ASM
8_3a_scalar_asm_propagation.py  Extended 8.3: full 2D prescribed-aperture propagation
8_3b_talbot_propagation.py      Extended 8.3: periodic thin-mask recurrence
A1_1_scalar_fmm_grating.py     Appendix 1: lossless scalar/TE FMM
A1_2_absorbing_fmm_grating.py  Appendix 1: absorber and ideal-mask comparison
A1_3_fmm_convergence.py        Appendix 1: Fourier-basis convergence
A1_4_grating_parameter_sweep.py Appendix 1: independent material-parameter solves
A2_1_finite_grating_propagation.py Appendix 2: finite ideal openings and ASM

Chapter files offer multiple numbered setups; their comments identify the
companion subnotebook. Edit example=1, example=2, etc., before running. Every
setup lists all its adjustable parameters. Sliders remain pending until Apply;
Reset restores the setup loaded from the parameter block. The terminal prints
the physical model and a short interpretation. A1/A2 tools retain their original
specialized plots and parameters; see SOLVER_NOTES.txt for their detailed notes.

Chapter-only CLI options:
  python Wave_solver_py_scripts/numerical_explorer/8_field_propagation.py --headless --all
  python Wave_solver_py_scripts/numerical_explorer/8_field_propagation.py --headless --example 5
  python Wave_solver_py_scripts/numerical_explorer/8_field_propagation.py --headless --all --render-dir previews
Alternatively set interactive=False in any explorer, including specialized ones.
No files are saved unless you request --render-dir or use the plot toolbar.

CONVENTIONS
Lengths are um except RS/Weyl setups explicitly measured in wavelengths.
Wave plots in Chapter 1 use +z; grating outputs use downstream s=-z.
Time convention exp(-i omega t). Most chapter setups use medium wavelength;
material models explicitly use vacuum wavelength. Material n+i*kappa values
are illustrative, not calibrated data. Kinematics and prescribed amplitudes
are distinct from solved scattering and flux efficiencies. Check Fourier cutoff,
quadrature, FFT sampling and padding as appropriate; conservation alone does
not certify convergence. Angular pupil collection alone does not form an image.

The other download, scripts, provides four section-wise worked implementations
per teaching notebook (0.1-9.2), with notebook/example filenames such as
chapter_0/0_1_01_medium_wavelength.py. It also retains the seven specialist solver
scripts with matching 8_3a/8_3b/A1/A2 names. Each download includes its own
wave_optics copy and runs independently.
