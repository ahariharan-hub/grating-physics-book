WAVE OPTICS PYTHON DOWNLOADS

numerical_explorer/  16 interactive tools with focused plots and Apply/Reset.
scripts/             140 section-wise worked examples for 35 teaching notebooks,
                     plus seven specialist solver scripts.

Each teaching notebook (0.1-9.2) links four worked scripts in a Worked-out
examples section. Chapter introductions link the script index. Examples expose
inputs, grids, numerical calculation, checks, plots and a summary as # %% sections.
The filenames identify notebook and example, e.g. chapter_0/0_1_01_medium_wavelength.py.
See scripts/INDEX.md and each collection's README.txt for installation and use.

The course Downloads.md page links two independent ZIP bundles in downloads/.
Each ZIP includes its selected collection, wave_optics, requirements and README.
Environments, caches, tests and notebook outputs are excluded.

Rebuild after changing scripts or backend:
  python Wave_solver_py_scripts/build_downloads.py

A future Jupyter Book should include Downloads.md in its table of contents and
retain the downloads folder beside it. No book has been configured or published
here yet. Its MyST download roles copy the bundles into the built site.

CHECKS
  python -m pytest supplement/tests/test_worked_examples.py
  python -m pytest supplement/tests/test_python_downloads.py
  python -m pytest supplement/tests/test_chapter_explorers.py
  python -m pytest supplement/tests/test_wave_solver_playgrounds.py supplement/tests/test_fmm_script.py
