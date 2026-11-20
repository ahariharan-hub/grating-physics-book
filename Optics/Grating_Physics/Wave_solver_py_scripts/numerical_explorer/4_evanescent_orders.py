"""Chapter 4: focused numerical experiments. Edit a setup below; run with F5.
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
    # 1: notebook 4.1 - Build a decaying order
    1: dict(
        period=1.0,
        distance=8.0,
        wavelength=1.0,
        angle=30.0,
        order=1,
        time=0.0,
    ),
    # 2: notebook 4.2 - Through grazing: phase, decay and flux
    2: dict(
        period=1.5,
        order=1,
        wavelength=1.0,
        angle=30.0,
        distance=8.0,
        time=0.0,
    ),
}


def compute_example(number=example):
    return calculate(4, number, examples[number])


if __name__ == '__main__':
    experiment = run_chapter(4, examples, example=example,
                             interactive=interactive, plot_backend=plot_backend)
