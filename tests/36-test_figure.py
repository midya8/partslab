import unittest,importlib.util
from pathlib import Path
from partslab.io import ROOT,read
class FigureTests(unittest.TestCase):
    def test_frozen_figure_exact_and_all_16_basis_images(self):
        spec=importlib.util.spec_from_file_location('render_audit','examples/render_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);svg=m.render(read(ROOT/'artifacts/models.json'),read(ROOT/'artifacts/report.json'));self.assertEqual(svg,Path('assets/parts-audit.svg').read_text());self.assertEqual(svg.count('PART '),16);self.assertIn('PCA and SVD win',svg);self.assertIn('CC BY 4.0',svg)
