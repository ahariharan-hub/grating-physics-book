"""Chapter 3: focused numerical experiments. Edit a setup below; run with F5.
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
    # 1: notebook 3.1 - Close the phase cycle
    1: dict(
        component=0.5,
        order=1,
        period=4.0,
        wavelength=1.0,
        angle=15.0,
    ),
    # 2: notebook 3.2 - Follow a selected order
    2: dict(
        order=1,
        angle=15.0,
        period=4.0,
        wavelength=1.0,
        component=0.5,
    ),
    # 3: notebook 3.3 - Phase cycles and path difference
    3: dict(
        order=1,
        angle=15.0,
        period=4.0,
        wavelength=1.0,
    ),
}


def compute_example(number=example):
    return calculate(3, number, examples[number])


if __name__ == '__main__':
    experiment = run_chapter(3, examples, example=example,
                             interactive=interactive, plot_backend=plot_backend)
