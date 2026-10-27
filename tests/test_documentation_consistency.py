"""Guard the notation and prediction labels used in the public README."""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentationConsistency(unittest.TestCase):
    def test_exponents_match_manuscript_and_estimator(self):
        readme = (ROOT / "README.md").read_text()
        paper = (ROOT / "paper/ICLR2027_ScaleWithoutSupport.tex").read_text()
        estimator = (ROOT / "src/features/fractal_math.py").read_text()

        self.assertRegex(
            readme,
            r"Hurst exponent\s+\(Ĥ; rescaled-range analysis\).*?"
            r"Hölder exponent \(Ŝ; return-probability scaling\)",
        )
        self.assertIn(r"\textbf{Hurst:} by rescaled-range analysis", paper)
        self.assertIn(r"\textbf{H\"older:}", paper)
        self.assertIn(r"p_\varepsilon(\tau)\sim\tau^{-\Strue}", paper)
        self.assertIn("H, H_r2 = calc_H(increment)", estimator)
        self.assertIn("S, S_r2 = calc_S(trajectory)", estimator)

    def test_predictions_and_title_match_manuscript(self):
        readme = (ROOT / "README.md").read_text()
        paper = (ROOT / "paper/ICLR2027_ScaleWithoutSupport.tex").read_text()
        title = "Scale Without Support: Fractal Signatures Are Not Document Representations"

        self.assertTrue(readme.startswith("# " + title))
        self.assertIn("title     = {" + title + "}", readme)
        self.assertIn("four falsifiable predictions (P1–P4", readme)
        self.assertEqual(set(re.findall(r"\\textbf\{P([1-4])\}", paper)), set("1234"))

    def test_n_star_qualification_matches_manuscript(self):
        stats = json.loads((ROOT / "experiments/stats/t2_n_star_composition.json").read_text())
        paper = (ROOT / "paper/ICLR2027_ScaleWithoutSupport.tex").read_text()
        generator = (ROOT / "experiments/code/consolidate_t0_t1_t2.py").read_text()

        self.assertIn(r"point estimate under a validated floor for \Hest", paper)
        self.assertIn(r"\emph{lower} bound for \Sest", paper)
        for key, value in stats.items():
            self.assertIn("point estimate" if key.endswith("_H") else "lower bound", value["note"])
        self.assertIn("if est_key == \"S\" else", generator)


if __name__ == "__main__":
    unittest.main()
