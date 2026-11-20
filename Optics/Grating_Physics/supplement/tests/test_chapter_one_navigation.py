"""The inserted lesson must precede geometry without redirecting old subject links."""
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]


def prose(name):
    book = json.loads((ROOT/name).read_text(encoding='utf-8'))
    return '\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='markdown')


class ChapterOneNavigationTests(unittest.TestCase):
    def test_overview_and_forward_route_visit_the_four_lessons_in_order(self):
        names = ['1_1_Oscillation_and_Wavelength.ipynb',
            '1_2_Ideal_Plane_Waves_versus_Finite_Optical_Beams.ipynb',
            '1_3_Wavefronts_and_Wavevectors.ipynb',
            '1_4_Plane_Waves_in_Three_Dimensions.ipynb']
        overview = prose('1_0_Plane_Waves_and_Wavevectors.ipynb')
        locations = [overview.index(']('+name+')') for name in names]
        self.assertEqual(locations, sorted(locations))
        for name, following in zip(names,names[1:]):
            self.assertIn(']('+following+')',prose(name))
        for number in range(1,5):
            self.assertEqual(len(list(ROOT.glob(f'1_{number}_*.ipynb'))),1)

    def test_all_active_course_notebook_local_link_paths_resolve(self):
        for path in ROOT.glob('*.ipynb'):
            book = json.loads(path.read_text(encoding='utf-8'))
            for cell in book['cells']:
                if cell['cell_type'] != 'markdown':
                    continue
                for target in re.findall(r'\]\(([^)]+)\)', ''.join(cell['source'])):
                    if target.startswith('attachment:'):
                        self.assertIn(target.split(':', 1)[1], cell.get('attachments', {}))
                        continue
                    if '://' in target or target.startswith(('mailto:', '#')):
                        continue
                    local = unquote(target.split('#')[0])
                    if local:
                        with self.subTest(notebook=path.name, target=target):
                            self.assertTrue((path.parent/local).exists())


if __name__ == '__main__': unittest.main()
