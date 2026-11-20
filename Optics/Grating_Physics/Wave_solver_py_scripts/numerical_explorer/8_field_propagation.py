"""Chapter 8: focused numerical experiments. Edit a setup below; run with F5.
Choose example by its numbered entry. Sliders are pending until Apply.
No lesson-navigation GUI, Jupyter kernel, or automatic file exports.
Lengths um unless explicitly stated; exp(-i omega t).
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chapter_playground import run_chapter, calculate

# EDIT PARAMETERS
example = 1
interactive = True
plot_backend = 'QtAgg'  # None respects your IDE backend.
examples = {
    # 1: notebook 8.1 - Propagation between two points
    1: dict(
        distance=8.0,
        wavelength=1.0,
    ),
    # 2: notebook 8.2 - Sum Rayleigh-Sommerfeld patches
    2: dict(
        distance=8.0,
        width=2.0,
        patches=25,
        phase=0.0,
    ),
    # 3: notebook 8.2 - Weyl: rebuild a spherical field
    3: dict(
        weyl_order=24,
        distance=8.0,
        width=2.0,
    ),
    # 4: notebook 8.3 - Compare spatial kernels first
    4: dict(
        distance=8.0,
        width=2.0,
        wavelength=1.0,
    ),
    # 5: notebook 8.3 - Follow one Fourier component
    5: dict(
        component=0.5,
        distance=8.0,
        wavelength=1.0,
    ),
    # 6: notebook 8.3 - Then propagate a full aperture
    6: dict(
        distance=8.0,
        width=2.0,
        wavelength=1.0,
        samples=256,
        window=32.0,
    ),
}


def compute_example(number=example):
    return calculate(8, number, examples[number])


if __name__ == '__main__':
    experiment = run_chapter(8, examples, example=example,
                             interactive=interactive, plot_backend=plot_backend)
