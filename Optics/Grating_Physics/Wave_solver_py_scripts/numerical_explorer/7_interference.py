"""Chapter 7: focused numerical experiments. Edit a setup below; run with F5.
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
    # 1: notebook 7.1 - Phase, balance and mutual coherence
    1: dict(
        phase=0.0,
        coherence=1.0,
        balance=1.0,
        period=4.0,
        distance=8.0,
    ),
    # 2: notebook 7.2 - Interference of grating orders
    2: dict(
        order=1,
        distance=8.0,
        period=4.0,
        phase=0.0,
        balance=1.0,
        coherence=1.0,
    ),
    # 3: notebook 7.3 - Finite beams must overlap
    3: dict(
        distance=8.0,
        waist=6.0,
        phase=0.0,
        balance=1.0,
    ),
    # 4: notebook 7.4 - Directions do not determine coefficients
    4: dict(
        balance=1.0,
        phase=0.0,
        period=4.0,
        order=1,
        coherence=1.0,
        distance=8.0,
    ),
}


def compute_example(number=example):
    return calculate(7, number, examples[number])


if __name__ == '__main__':
    experiment = run_chapter(7, examples, example=example,
                             interactive=interactive, plot_backend=plot_backend)
