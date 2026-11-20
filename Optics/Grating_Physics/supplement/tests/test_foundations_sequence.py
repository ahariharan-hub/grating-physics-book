"""Preservation and navigation checks for the independent scalar overview lessons."""

import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit

import nbformat


ROOT = Path(__file__).resolve().parents[2]
LESSONS = sorted(ROOT.glob('0_[1-7]_*.ipynb'))
INVENTORY = ROOT/'supplement/validation/notebook_0_split_inventory.json'


class FoundationsSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.books = {p.name: nbformat.read(p, 4) for p in LESSONS}
        cls.inventory = json.loads(INVENTORY.read_text(encoding='utf-8'))

    def test_consistent_introductions_and_clean_executable_cells(self):
        self.assertEqual(len(self.books), 7)
        for name, book in self.books.items():
            with self.subTest(notebook=name):
                nbformat.validate(book)
                intro = book.cells[0].source
                for label in ('Motivation and prerequisites', '**Motivation.', '**Essential prerequisites.',
                              '**Optional advanced mathematical background.', '**First reading.', '**Next lesson.'):
                    self.assertIn(label, intro)
                first_reading = intro.split('**First reading.**', 1)[1].split('\n\n', 1)[0].lower()
                overview_descriptions = (
                    ('conceptual overview', 'subsequent numbered'),
                    ('the 0 series introduces the main ideas visually',
                     'later chapters develop the mathematics'),
                )
                self.assertTrue(
                    any(all(phrase in first_reading for phrase in description)
                        for description in overview_descriptions),
                    f'{name}: first-reading guidance must explain the overview/later-chapter progression',
                )
                self.assertNotIn('widgets', book.metadata)
                for cell in book.cells:
                    if cell.cell_type == 'code':
                        compile(cell.source, name, 'exec')
                        self.assertIsNone(cell.execution_count)
                        self.assertEqual(cell.outputs, [])

    def test_split_preserves_display_equations_references_and_views(self):
        combined = '\n'.join(c.source for name, b in self.books.items()
                             if name.startswith(('0_5', '0_6', '0_7')) for c in b.cells)
        def normalize(s):
            # Preserve the historical split inventory, translating only the
            # documented symbol migrations: period p -> d, modal A -> F,
            # opening fraction D -> D_open, second order n -> p.
            for old, new in ((r'\frac{2\pi}{p}', r'\frac{2\pi}{d}'),
                             (r'\frac{1}{p}', r'\frac{1}{d}'),
                             (r'\frac1p', r'\frac1d'),
                             (r'\frac{k_0}{p}', r'\frac{k_0}{d}'),
                             ('-p/2', '-d/2'), ('p/2', 'd/2'),
                             ('mx/p', 'mx/d'), ('m/p', 'm/d'),
                             ('A_m', 'F_m'), ('A_n', 'F_p'),
                             (r'\sum_n', r'\sum_p'),
                             (r'\varepsilon_{m-n}', r'\varepsilon_{m-p}'),
                             ('F_n', 'F_p'), ('Q_{mn}', 'Q_{mp}'),
                             (r'\delta_{mn}', r'\delta_{mp}')):
                s = s.replace(old, new)
            s = re.sub(r'([qm])D(?!_)', lambda match: match[1] + r'D_{\mathrm{open}}', s)
            s = re.sub(r'\bD\b', lambda _: r'D_{\mathrm{open}}', s)
            return re.sub(r'\s+', '', s)
        normalized = normalize(combined)
        # Keep the original inventory immutable. These exact equation updates
        # record the later distinction between mask transmission t_m and field
        # coefficients C_m, and the explicit exit-field name. In particular,
        # never replace t_m globally: it remains correct for the ideal mask.
        notation_updates = (
            (
                'grating-evolution-explanation',
                r'$$\boxed{C_m(-h-\Delta)=t_m e^{i\beta_m\Delta}.}$$',
                r'$$\boxed{C_m(-h-\Delta)=C_m(-h) e^{i\beta_m\Delta}.}$$',
            ),
            (
                'absorber-common-plane',
                r'$$\widetilde t_m=\frac{1}{p}\int_{-p/2}^{p/2}t_{\mathrm{ideal}}(x)e^{-i2\pi mx/p}\,dx'
                r'=D\,\operatorname{sinc}(mD).$$',
                r'$$t_m=\frac{1}{d}\int_{-d/2}^{d/2}t_{\mathrm{ideal}}(x)e^{-i2\pi mx/d}\,dx'
                r'=D_{\mathrm{open}}\,\operatorname{sinc}(mD_{\mathrm{open}}).$$',
            ),
            (
                'absorber-common-plane',
                r'$$U_{\mathrm{ideal}}(x,-h)=e^{ik_0h}t_{\mathrm{ideal}}(x),\qquad'
                r'C_m^{\mathrm{ideal}}(-h)=e^{ik_0h}\widetilde t_m.$$',
                r'$$U_{\mathrm{exit}}^{\mathrm{ideal}}(x)=e^{ik_0h}t_{\mathrm{ideal}}(x),\qquad'
                r'C_m^{\mathrm{ideal}}(-h)=e^{ik_0h}t_m.$$',
            ),
        )
        expected_updates = {
            (cell_id, normalize(old)): normalize(new)
            for cell_id, old, new in notation_updates
        }
        for component in self.inventory['components']:
            for formula in component['math']:
                if formula.startswith('$$'):
                    expected = expected_updates.get(
                        (component['cell_id'], normalize(formula)), normalize(formula))
                    self.assertIn(expected, normalized, component['cell_id'])
            for link in component['links']:
                if link.startswith('http'):
                    self.assertIn(link, combined)
            if component['cell_type'] == 'code' and component['cell_id'] != 'grating-setup':
                destination = self.books[component['destinations'][0]]
                moved = next(c for c in destination.cells if c.id == component['cell_id'])
                self.assertEqual(moved.source, component['code'])

    def test_navigation_targets_exist_and_sequence_is_ordered(self):
        index = nbformat.read(ROOT/'0_0_Foundations_of_Scalar_Optics.ipynb', 4)
        source = index.cells[0].source
        positions = [source.index(name) for name in self.books]
        self.assertEqual(positions, sorted(positions))
        for name, book in [*self.books.items(), ('0_0_Foundations_of_Scalar_Optics.ipynb', index)]:
            for cell in book.cells:
                if cell.cell_type != 'markdown':
                    continue
                for target in re.findall(r'\]\(([^)]+)\)', cell.source):
                    parsed = urlsplit(target)
                    if not parsed.scheme and parsed.path:
                        self.assertTrue((ROOT/unquote(parsed.path)).exists(), (name, target))
        names = list(self.books)
        for i, name in enumerate(names):
            text = '\n'.join(c.source for c in self.books[name].cells)
            if i:
                self.assertIn(names[i-1], text)
            if i < 6:
                self.assertIn(names[i+1], text)

if __name__ == '__main__':
    unittest.main()
