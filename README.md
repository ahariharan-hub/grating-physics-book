# Grating Physics

[Read the book](https://ahariharan-hub.github.io/grating-physics-book/)

The Jupyter Book pilot contains **0.0: Foundations of Scalar Optics**
and **0.1: Scalar Fields and Helmholtz**, and **0.2: Solutions of the Helmholtz equation**. Other teaching sources are included for
future chapters; they are not part of the published table of contents.

Click the power button at the top of notebook 0.1 to activate its interactive
examples. The public book uses a Binder-hosted Python kernel; readers do not
need to install Python. Startup can take time and sessions are temporary.

Sources, the `wave_optics` backend, numerical explorers and worked Python
examples are in `Optics/Grating_Physics/`.

## Build locally

Use Python 3.12 and the recorded requirements:

```console
cd Optics/Grating_Physics
python -m pip install -r book/requirements-tested.txt
python book/pilot.py execute
python book/check_solutions.py
python book/pilot.py build
python book/check_build.py
```

For a private local interactive preview, run `python book/pilot.py preview`.
Open http://localhost:3010/. Connection credentials and logs stay under ignored
`_build/`; the public deployment uses only the separately built public artifact.

GitHub Actions executes and publishes only the three pilot pages. See
[`book/README.md`](Optics/Grating_Physics/book/README.md) for validation details.
