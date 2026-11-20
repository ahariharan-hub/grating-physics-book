# Download Python experiments

Choose the way you want to work. Both collections use the `wave_optics` backend,
include their own copy of that package, and run independently after extraction.
Keep the extracted folder structure together.

| Collection | What you get | Download |
| --- | --- | --- |
| Numerical explorers | Editable parameters, focused plots, sliders, Apply and Reset | {download}`Download numerical explorers <downloads/grating_physics_numerical_explorer.zip>` |
| Worked-out Python scripts | Four section-wise implementations per teaching notebook: inputs, grids, numerical calculation, checks and static plots | {download}`Download Python scripts <downloads/grating_physics_scripts.zip>` |

## Numbering and coverage

The numerical-explorer collection has 16 interactive programs. Its chapter files are:

| Chapter | Filename | Topic |
| --- | --- | --- |
| 0 | `0_scalar_foundations.py` | Scalar fields, Helmholtz and boundary data |
| 1 | `1_plane_waves.py` | Plane waves and wavevectors |
| 2 | `2_grating_orders.py` | Finite arrays and diffraction orders |
| 3 | `3_phase_matching.py` | Translation phase and order matching |
| 4 | `4_evanescent_orders.py` | Evanescent fields and decay |
| 5 | `5_diffraction_coordinates.py` | Three-dimensional orders, lattices and pupils |
| 6 | `6_field_to_intensity.py` | Exposure averaging and detector intensity |
| 7 | `7_interference.py` | Coherence, grating-order interference and overlap |
| 8 | `8_field_propagation.py` | Scalar propagation kernels and Fourier components |

The numerical chapter explorers provide 36 numbered setups linked to subnotebooks
0.1–8.3. The scripts collection instead provides **140 worked examples**, four for
each teaching notebook from **0.1 through 9.2**, plus seven specialist solver scripts.
It is organized by chapter, with filenames identifying notebook and example:

```text
scripts/
  chapter_0/0_1_01_medium_wavelength.py
  chapter_0/0_1_02_phasor_and_real_field.py
  chapter_0/0_1_03_helmholtz_trial_residual.py
  chapter_0/0_1_04_finite_difference_helmholtz.py
  ...
  chapter_8/8_2_01_rectangular_rs_integral.py
  chapter_8/8_2_03_weyl_plane_wave_integral.py
  ...
  chapter_9/9_2_04_two_dimensional_fft_coefficients.py
```

Each teaching notebook has a **Worked-out examples** section naming and linking
its four scripts. The [complete script index](Wave_solver_py_scripts/scripts/INDEX.md)
lists the numerical task of every example. Chapter introductions link the relevant
index entries; they do not have four separate introductory scripts.

These are readable implementations, not wrappers around an explorer. The
`# %%` sections expose input parameters, coordinate grids, calculations, numerical
checks, plotting and a summary. Standard NumPy/SciPy operations implement the
mathematics directly where practical; physical material-grating solves can use
`wave_optics`. The plotting helper contains no numerical physics.

Additional programs in both collections are `8_3a_scalar_asm_propagation.py` and
`8_3b_talbot_propagation.py`. The `a` and `b` suffixes identify extended experiments
for 8.3, rather than new notebook chapters. Appendix 1 has four `A1_` programs for
lossless FMM, absorption, convergence and parameter sweeps. Appendix 2 has
`A2_1_finite_grating_propagation.py` for finite prescribed openings.

## Start after downloading

1. Extract the entire ZIP.
2. Open a terminal in the extracted `Grating_Physics` directory.
3. Install the included requirements:

   ```console
   python -m pip install -r requirements.txt
   ```

4. Open a numbered file in Spyder and run it with **F5**, or use Python:

   ```console
   python Wave_solver_py_scripts/numerical_explorer/1_plane_waves.py
   ```

   For the ordinary scripts:

   ```console
   python Wave_solver_py_scripts/scripts/chapter_1/1_1_01_time_oscillation.py
   ```

Use the Python environment where the requirements were installed. The numerical
explorers use a desktop Qt window. Standard scripts need no Qt binding; add
`--no-show` to calculate without opening a plot. Neither collection needs a
Jupyter kernel. Figures are saved only when you explicitly request it. To save a
worked-example figure without opening a window:

```console
python Wave_solver_py_scripts/scripts/chapter_8/8_2_01_rectangular_rs_integral.py --no-show --save-dir figures
```

Read the model and units at the top of each file. Prescribed fields, allowed order
directions and physical scattering solves answer different questions; intensity
and normal power efficiency also differ. Check numerical convergence and sampling
before interpreting a result quantitatively.

[Return to the course introduction](0_0_Foundations_of_Scalar_Optics.ipynb).
