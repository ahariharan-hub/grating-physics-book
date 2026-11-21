"""Shared, fail-closed selection and reader-link checks for public teaching files."""
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

# Only the active main series is public. Adding a lesson requires an explicit
# selection here; the visible MyST TOC may continue to contain a smaller subset.
PUBLIC_NOTEBOOKS = (
    '0_0_Foundations_of_Scalar_Optics.ipynb',
    '0_1_Scalar_Fields_and_Helmholtz.ipynb',
    '0_2_Solutions_of_the_Helmholtz_Equation.ipynb',
    '0_3_Boundary_Conditions_and_Diffraction.ipynb',
    '0_4_Propagation_of_Scalar_Optical_Fields.ipynb',
    '0_5_From_Helmholtz_BVP_to_Diffracted_Field.ipynb',
    '0_6_From_Complex_Fields_to_Diffraction_Orders.ipynb',
    '0_7_Ideal_Binary_Masks_vs_Physical_Gratings.ipynb',
    '1_0_Plane_Waves_and_Wavevectors.ipynb',
    '1_1_Oscillation_and_Wavelength.ipynb',
    '1_2_Ideal_Plane_Waves_versus_Finite_Optical_Beams.ipynb',
    '1_3_Wavefronts_and_Wavevectors.ipynb',
    '1_4_Plane_Waves_in_Three_Dimensions.ipynb',
    '2_0_Plane_Wave_Meets_a_Grating.ipynb',
    '2_1_From_a_Slit_to_a_Grating.ipynb',
    '2_2_Illumination_and_the_Surface_Field.ipynb',
    '2_3_Allowed_Diffraction_Orders.ipynb',
    '2_4_Order_Coordinates_and_Spacing.ipynb',
    '3_0_Why_Diffraction_Orders_Appear.ipynb',
    '3_1_Phase_Change_and_the_Order_Rule.ipynb',
    '3_2_Selected_Orders_and_Phase_Continuity.ipynb',
    '3_3_Phase_Cycles_and_Path_Differences.ipynb',
    '4_0_Evanescent_Diffraction_Orders.ipynb',
    '4_1_Constructing_an_Evanescent_Field.ipynb',
    '4_2_Decay_Length_Grazing_and_Energy_Flow.ipynb',
    '5_0_Grating_Diffraction_in_Three_Dimensions.ipynb',
    '5_1_Three_Dimensional_Orders_and_Conical_Diffraction.ipynb',
    '5_2_Direction_Cosines_and_Spatial_Frequency_Coordinates.ipynb',
    '5_3_Two_Dimensional_Gratings_and_Diffraction_Order_Lattices.ipynb',
    '5_4_From_Diffraction_Coordinates_to_Pupils.ipynb',
    '6_0_Amplitude_and_Intensity_of_a_Scalar_Wave.ipynb',
    '6_1_Time_Averaging_and_Relative_Intensity.ipynb',
    '6_2_One_Order_Across_a_Detector.ipynb',
    '7_0_Introduction_to_Interference_and_Beam_Overlap.ipynb',
    '7_1_Interference_and_Mutual_Coherence_of_Two_Plane_Waves.ipynb',
    '7_2_Interference_of_Diffraction_Orders_from_a_Grating.ipynb',
    '7_3_Finite_Diffraction_Beams_and_Experimental_Recombination.ipynb',
    '7_4_Diffraction_Kinematics_and_Complex_Order_Coefficients.ipynb',
    '8_0_Introduction_to_Field_Propagation.ipynb',
    '8_1_Physical_Pictures_Behind_Scalar_Field_Propagation.ipynb',
    '8_2_Rayleigh_Sommerfeld_Propagation.ipynb',
    '8_3_ASM_Fresnel_and_Fraunhofer_Propagation.ipynb',
    '9_0_From_a_Physical_Model_to_a_Numerical_Experiment.ipynb',
    '9_1_Defining_the_Complex_Exit_Field.ipynb',
    '9_2_Sampling_Complex_Fields_and_Their_Spectra.ipynb',
    'Closing_0_From_Diffraction_to_Imaging.ipynb',
    'Appendix_1_Fourier_Modal_Method_for_Scalar_Grating_Scattering.ipynb',
    'Appendix_2_Numerical_Diffraction_of_Finite_Grating_Illumination.ipynb',
    'Appendix_3_Historical_Context_and_Original_Sources.ipynb',
    'Appendix_4_How_to_Use_FFTs_in_Python.ipynb',
    'Appendix_5_Formulae_Sheet.ipynb',
)
PUBLIC_MARKDOWN = ('index.md', 'preface.md', 'reader-guide.md', 'Downloads.md')
PUBLIC_HELPERS = {'pilot.py', 'public_content.py', 'validation_tests.py', 'check_build.py',
                  'check_solutions.py', 'check_boundary.py', 'check_propagation.py',
                  'check_scattering.py', 'check_orders.py', 'check_masks.py', 'requirements-tested.txt'}
ASSET_TYPES = {'.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.css', '.woff', '.woff2', '.ico'}


def is_public_source(path):
    path = Path(path)
    parts = path.parts
    if not parts or path.is_absolute() or '..' in parts:
        return False
    if any(p.lower() in ('supplement', 'tests', 'validation') for p in parts) or path.name.lower().startswith('test_'):
        return False
    if any(p.startswith('.') or p == '__pycache__' for p in parts):
        return path.as_posix() == '.gitignore'
    if len(parts) == 1:
        return path.name in (*PUBLIC_NOTEBOOKS, *PUBLIC_MARKDOWN, 'myst.yml')
    if parts[0] == 'book':
        return len(parts) == 2 and path.name in PUBLIC_HELPERS
    if parts[0] == 'wave_optics':
        return path.suffix == '.py' and 'tests' not in parts and not path.name.startswith('test_')
    if parts[0] == 'Wave_solver_py_scripts':
        return (len(parts) == 2 and path.name == 'build_downloads.py') or (
            len(parts) > 2 and parts[1] in ('scripts', 'numerical_explorer')
            and path.suffix in ('.py', '.md', '.txt') and 'tests' not in parts
            and not path.name.startswith('test_'))
    if parts[0] == 'downloads':
        return len(parts) == 2 and path.name in ('grating_physics_scripts.zip',
            'grating_physics_numerical_explorer.zip', 'manifest.json')
    return parts[0] == 'assets' and path.suffix.lower() in ASSET_TYPES


def source_files(root):
    """Select sources for both public and local staging, without private trees."""
    root = Path(root)
    candidates = list(root.iterdir())
    for directory in ('book', 'wave_optics', 'Wave_solver_py_scripts', 'downloads', 'assets'):
        if (root / directory).is_dir():
            candidates.extend((root / directory).rglob('*'))
    selected = sorted(p for p in candidates if p.is_file() and is_public_source(p.relative_to(root)))
    for path in selected:
        if path.is_symlink() or root.resolve() not in path.resolve().parents:
            raise RuntimeError('Do not stage a symlink or source outside the book: ' + str(path))
    return selected


def markdown_links(text):
    """Read balanced inline Markdown destinations, including parenthesized URLs."""
    start = re.compile(r'!?\[(?:\\.|[^\[\]\n])*\]\(')
    pos = 0
    while match := start.search(text, pos):
        i, depth = match.end(), 1
        begin = i
        while i < len(text) and depth:
            if text[i] == '(' and text[i-1] != '\\':
                depth += 1
            elif text[i] == ')' and text[i-1] != '\\':
                depth -= 1
            i += 1
        if depth:
            raise RuntimeError('Unclosed Markdown link near: ' + text[match.start():match.start()+90])
        yield text[begin:i-1]
        pos = i
    yield from re.findall(r'(?:src|href)=[\"\']([^\"\']+)[\"\']', text, re.I)
    yield from re.findall(r'\{download\}`[^`<>]*<([^>]+)>`', text)


def check_readers(root, available=None):
    """Reject internal-document links before staging or any public sync writes."""
    root = Path(root).resolve()
    available = set(available) if available is not None else None
    problems = []
    names = [*PUBLIC_NOTEBOOKS, *PUBLIC_MARKDOWN]
    names.extend(p.relative_to(root).as_posix() for p in (root/'Wave_solver_py_scripts').rglob('*.md')
                 if is_public_source(p.relative_to(root)))
    for name in names:
        path = root / name
        if not path.is_file():
            continue  # Small isolated fixtures may contain only one public lesson.
        if path.suffix == '.ipynb':
            notebook = json.loads(path.read_text(encoding='utf-8'))
            text = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown')
        else:
            text = path.read_text(encoding='utf-8')
        if re.search(r'original unsplit notebook|original foundations notebook|archived versions|reading_sources\.json|supplement[\\/]', text, re.I):
            problems.append((name, 'internal supplementary/development reference'))
        prose = re.sub(r'^(`{3,}|~{3,})[^\n]*\n.*?^\1[ \t]*$|`+[^`\n]*`+|\$\$.*?\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$',
                       '', text, flags=re.M | re.S)
        if re.search(r'(?<![\w/])\d+_\d+(?!\w|\.(?:ipynb|py|md|json|png|svg)\b)', prose):
            problems.append((name, 'use dotted lesson numbers and link the named lesson'))
        for url in markdown_links(text):
            parsed = urlsplit(url.strip('<>'))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            try:
                rel = target.relative_to(root).as_posix()
            except ValueError:
                problems.append((name, 'link leaves the public book: ' + url))
                continue
            if not is_public_source(rel) or (available is not None and rel not in available) or not target.is_file():
                problems.append((name, 'non-public or missing destination: ' + url))
    if problems:
        raise RuntimeError('Public reader-content check failed:\n' + '\n'.join(name + ': ' + why for name, why in problems))
    return len([n for n in names if (root/n).is_file()])
