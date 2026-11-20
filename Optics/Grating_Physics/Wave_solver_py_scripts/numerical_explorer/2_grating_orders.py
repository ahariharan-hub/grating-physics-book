"""Chapter 2: focused numerical experiments. Edit a setup below; run with F5.
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
    # 1: notebook 2.1 - From one slit to many
    1: dict(
        count=5,
        width=2.0,
        period=4.0,
        wavelength=1.0,
    ),
    # 2: notebook 2.2 - Illumination writes a surface phase
    2: dict(
        angle=15.0,
        wavelength=1.0,
        period=4.0,
        duty=0.5,
    ),
    # 3: notebook 2.3 - Which orders can propagate?
    3: dict(
        period=4.0,
        angle=15.0,
        wavelength=1.0,
        azimuth=0.0,
        order=1,
    ),
    # 4: notebook 2.4 - Equal spacing, different coordinate scales
    4: dict(
        period=4.0,
        wavelength=1.0,
        angle=15.0,
        azimuth=30.0,
        order=1,
    ),
}


def compute_example(number=example):
    return calculate(2, number, examples[number])


if __name__ == '__main__':
    experiment = run_chapter(2, examples, example=example,
                             interactive=interactive, plot_backend=plot_backend)
