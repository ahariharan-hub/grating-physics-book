"""Chapter 6: focused numerical experiments. Edit a setup below; run with F5.
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
    # 1: notebook 6.1 - What does the detector average?
    1: dict(
        cycles=3.0,
        phase=0.0,
        amplitude=1.0,
    ),
    # 2: notebook 6.2 - A tilted field across a detector
    2: dict(
        angle=15.0,
        time=0.0,
        wavelength=1.0,
        amplitude=1.0,
        phase=0.0,
    ),
}


def compute_example(number=example):
    return calculate(6, number, examples[number])


if __name__ == '__main__':
    experiment = run_chapter(6, examples, example=example,
                             interactive=interactive, plot_backend=plot_backend)
