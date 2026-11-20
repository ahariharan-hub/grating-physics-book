"""Chapter 5: focused numerical experiments. Edit a setup below; run with F5.
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
    # 1: notebook 5.1 - Conical orders in 3D
    1: dict(
        angle=15.0,
        azimuth=30.0,
        period=4.0,
        wavelength=1.0,
        order=1,
    ),
    # 2: notebook 5.2 - One order on three coordinate scales
    2: dict(
        wavelength=1.0,
        azimuth=30.0,
        period=4.0,
        angle=15.0,
        order=1,
    ),
    # 3: notebook 5.3 - Two periods, two order integers
    3: dict(
        period=4.0,
        pitch_y=5.0,
        wavelength=1.0,
        angle=15.0,
        azimuth=30.0,
    ),
    # 4: notebook 5.4 - From order directions to pupil collection
    4: dict(
        na=0.7,
        angle=15.0,
        period=4.0,
        wavelength=1.0,
        azimuth=30.0,
    ),
}


def compute_example(number=example):
    return calculate(5, number, examples[number])


if __name__ == '__main__':
    experiment = run_chapter(5, examples, example=example,
                             interactive=interactive, plot_backend=plot_backend)
