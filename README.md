# Grating Physics

**Scalar Theory of Grating Diffraction**

An interactive textbook exploring optical physics through **physical intuition, mathematical derivations, and numerical experiments**.

**[Read the online textbook](https://ahariharan-hub.github.io/grating-physics-book/)**

## About this book

How does a periodic structure transform an incident optical wave into diffracted fields?

This book explores that question, starting with scalar wave theory and progressing through diffraction orders, evanescent waves, interference, coherence, and field propagation.

The emphasis is on understanding the physics by connecting equations to interactive visualizations and executable Python examples.

**Predict what will happen. Explore the mathematics. Change a parameter. Observe the result.**

## Contents

The textbook currently contains **51 Jupyter notebooks**, organized into:

| Chapter | Topic |
|---|---|
| 0 | Foundations of Scalar Optics |
| 1 | Plane Waves and Wavevectors |
| 2 | A Plane Wave Meets a Grating |
| 3 | Why Diffraction Orders Appear |
| 4 | Evanescent Diffraction Orders |
| 5 | Grating Diffraction in Three Dimensions |
| 6 | Amplitude and Intensity |
| 7 | Interference and Beam Overlap |
| 8 | Scalar Field Propagation |
| 9 | Complex Exit Fields and Numerical Experiments |

The book also includes a closing synthesis connecting diffraction to imaging, five mathematical and computational appendices, and downloadable Python experiments.

*Chapter 9 currently contains lessons 9.0–9.2; additional lessons are planned.*

## Interactive notebooks

The mathematical explanations, equations, and static figures can be read directly in your browser.

To run the interactive examples:

1. Open a notebook in the [online textbook](https://ahariharan-hub.github.io/grating-physics-book/).
2. Click the **power button** near the top of the page to activate a Binder-hosted Python session.
3. Select **Run all cells** if necessary.
4. Use the sliders, buttons, and plots to explore the models.

**No Python installation is required.** Binder sessions are temporary, and initial startup may take some time.

See the [Reader's Guide](Optics/Grating_Physics/reader-guide.md) for additional instructions.

## Python code and numerical experiments

The repository includes:

- Jupyter notebooks containing explanations, equations, and interactive demonstrations.
- The `wave_optics` Python backend implementing shared optical models.
- Standalone numerical explorers and worked Python scripts.
- Downloadable experiment collections.

The teaching sources are located in [`Optics/Grating_Physics/`](Optics/Grating_Physics/).

See [Downloads](Optics/Grating_Physics/Downloads.md) for ready-to-run experiment collections.

## Build locally

Use Python 3.12 and the recorded dependency versions.

```bash
cd Optics/Grating_Physics
python -m pip install -r book/requirements-tested.txt

python book/pilot.py execute
python book/check_solutions.py
python book/pilot.py build
python book/check_build.py
```

For an interactive local preview:

```bash
python book/pilot.py preview
```

Open `http://localhost:3010/` in your browser.

GitHub Actions executes and validates the notebooks in the published table of contents before deploying the website to GitHub Pages.

Further information is available in the [book implementation notes](Optics/Grating_Physics/book/README.md).

## Feedback and contributions

Questions, corrections, and suggestions are welcome through [GitHub Issues](https://github.com/ahariharan-hub/grating-physics-book/issues).

When reporting a numerical or interactive issue, include the notebook number, parameter settings, and steps to reproduce it.

---

**[Start reading Grating Physics](https://ahariharan-hub.github.io/grating-physics-book/)**
