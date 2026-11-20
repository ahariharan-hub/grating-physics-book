"""Chapter 0: focused numerical experiments. Edit a setup below; run with F5.
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
    # 1: notebook 0.1 - Does this field satisfy Helmholtz?
    1: dict(
        ratio=1.0,
        wavelength=1.0,
    ),
    # 2: notebook 0.2 - Compare solution families
    2: dict(
        family=0,
        wavelength=1.0,
        time=0.0,
    ),
    # 3: notebook 0.3 - The same boundary value, different fields
    3: dict(
        backward=0.0,
        phase=0.0,
        wavelength=1.0,
    ),
    # 4: notebook 0.4 - Propagate a known aperture field
    4: dict(
        distance=8.0,
        width=2.0,
        wavelength=1.0,
        samples=256,
        window=32.0,
    ),
    # 5: notebook 0.5 - Solve a physical grating
    5: dict(
        thickness=0.3,
        loss=0.8,
        duty=0.5,
        cutoff=12,
    ),
    # 6: notebook 0.6 - One field, two representations
    6: dict(
        distance=0.5,
        cutoff=12,
        thickness=0.3,
        loss=0.8,
        duty=0.5,
    ),
    # 7: notebook 0.7 - Ideal mask versus physical absorber
    7: dict(
        thickness=0.3,
        loss=0.8,
        duty=0.5,
        cutoff=12,
    ),
}


def compute_example(number=example):
    return calculate(0, number, examples[number])


if __name__ == '__main__':
    experiment = run_chapter(0, examples, example=example,
                             interactive=interactive, plot_backend=plot_backend)
