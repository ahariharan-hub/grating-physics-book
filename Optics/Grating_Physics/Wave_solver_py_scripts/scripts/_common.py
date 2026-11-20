"""Locate wave_optics and choose static plotting; no GUI widgets or solvers here."""
from pathlib import Path
import sys

COURSE = next((base for base in Path(__file__).resolve().parents
               if (base/'wave_optics/__init__.py').is_file()), None)
if COURSE is None:
    raise FileNotFoundError('Keep scripts in the extracted Grating_Physics folder beside wave_optics.')
if str(COURSE) not in sys.path:
    sys.path.insert(0,str(COURSE))


def pyplot(show=True):
    import matplotlib
    if not show or '--no-show' in sys.argv:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def finish(figure, show=True):
    import matplotlib.pyplot as plt
    if '--save-dir' in sys.argv:
        position = sys.argv.index('--save-dir')
        if position + 1 >= len(sys.argv):
            raise ValueError('Provide an output directory after --save-dir.')
        destination = Path(sys.argv[position + 1]).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination / (Path(sys.argv[0]).stem + '.png'), dpi=130)
    if show and '--no-show' not in sys.argv:
        plt.show()
    else:
        plt.close(figure)
