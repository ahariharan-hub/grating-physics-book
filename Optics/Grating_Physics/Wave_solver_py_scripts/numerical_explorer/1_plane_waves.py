"""Chapter 1: focused numerical experiments. Edit a setup below; run with F5.
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
    # 1: notebook 1.1 - Oscillation, wavelength and phase
    1: dict(
        wavelength=1.0,
        time=0.0,
        phase=0.0,
        angle=0.0,
        azimuth=0.0,
    ),
    # 2: notebook 1.2 - Infinite plane wave and finite beam
    2: dict(
        width=4.0,
        distance=8.0,
        wavelength=1.0,
    ),
    # 3: notebook 1.3 - Wavefronts are normal to k
    3: dict(
        angle=15.0,
        time=0.0,
        wavelength=1.0,
        phase=0.0,
        azimuth=0.0,
    ),
    # 4: notebook 1.4 - Rotate a plane wave in 3D
    4: dict(
        angle=15.0,
        azimuth=30.0,
        wavelength=1.0,
        time=0.0,
        phase=0.0,
    ),
}


def compute_example(number=example):
    return calculate(1, number, examples[number])


if __name__ == '__main__':
    experiment = run_chapter(1, examples, example=example,
                             interactive=interactive, plot_backend=plot_backend)
