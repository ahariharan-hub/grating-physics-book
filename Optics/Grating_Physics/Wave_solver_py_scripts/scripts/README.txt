WORKED-OUT PYTHON EXAMPLES

140 section-wise examples accompany 35 teaching notebooks, 0.1 through 9.2.
Each notebook has four separate programs for its relevant numerical topics.
See INDEX.md for the complete catalogue. Seven specialist solver scripts are
also retained at this folder's top level for deeper material/propagation work.

STRUCTURE
  chapter_0/0_1_01_medium_wavelength.py
  chapter_0/0_1_02_phasor_and_real_field.py
  chapter_0/0_1_03_helmholtz_trial_residual.py
  chapter_0/0_1_04_finite_difference_helmholtz.py
  ...
  chapter_8/8_2_01_rectangular_rs_integral.py
  chapter_8/8_2_03_weyl_plane_wave_integral.py
  ...
  chapter_9/9_2_04_two_dimensional_fft_coefficients.py

The prefix identifies chapter, notebook and example. The examples are individual
readable implementations, not four parameter settings of one hidden solver.
Every file has these Spyder-compatible # %% sections:
  INPUT PARAMETERS
  SETUP: coordinates, sampling and constants
  NUMERICAL CALCULATION
  CHECK AND INTERPRET
  GRAPHICS
  NUMERICAL SUMMARY

NumPy/SciPy arithmetic, loops, integration, finite differences, FFTs and reference
comparisons are visible in the files. Physical material scattering can use
wave_optics; reconstruction, coefficient projection and other teaching steps
remain explicit. _common.py only locates the backend and handles plotting.
No explorer, notebook kernel, sliders or lesson dispatcher is needed.
The supplied classroom-style example informed the sectioned layout; these are
new implementations for this course's models, units and conventions.

INSTALL AND RUN
1. Extract the entire ZIP and keep the folder structure together.
2. Open a terminal in the extracted Grating_Physics folder.
3. Install: python -m pip install -r requirements.txt
4. Run, for example:
   python Wave_solver_py_scripts/scripts/chapter_8/8_2_01_rectangular_rs_integral.py
Or open a file in Spyder and run it with F5. To study its implementation, execute
its # %% sections in order and inspect the intermediate arrays.

Set show_plot=False or add --no-show for headless calculations:
  python Wave_solver_py_scripts/scripts/chapter_0/0_1_01_medium_wavelength.py --no-show
To explicitly save a worked-example PNG:
  python Wave_solver_py_scripts/scripts/chapter_8/8_2_01_rectangular_rs_integral.py --no-show --save-dir figures
No figures are saved by default. Importing a new worked-example file executes
its numerical sections and checks, but its graphics block runs only when the
file is executed as a script. The seven older specialist scripts instead expose
compute()/plot() functions and retain their existing parameter blocks.

HOW TO LEARN FROM AN EXAMPLE
Read the model, units and assumptions in its docstring. Change one input, inspect
the grid and intermediate quantities, and compare the result with its analytical
or independently computed reference. Assertions check the stated case and
numerical accuracy; intentionally unresolved grids or different cases may fail
a check. Read that check before changing its tolerance or expected result.
Convergence comparisons use a numerical reference, not an exact solution.

The main notebooks have a Worked-out examples section naming their four files.
Chapter introductions serve as indexes; they do not get four duplicate examples.
The numerical_explorer download remains a separate collection for exploration.

Requirements: NumPy, SciPy, Matplotlib, and ipywidgets because some existing
propagation helper imports use it. These examples create no widget interfaces
and need no Qt binding or Jupyter kernel. Length and wavelength conventions are
stated in each program; material indices are illustrative rather than calibrated.
