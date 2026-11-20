import unittest
import pandas as pd

from src.parcel_ml.target_typo_check import check_target_typos
from src.parcel_ml.spelling_check import check_spelling_variants


class TestSaurabhiChecks(unittest.TestCase):

    def test_target_typos_detected(self):
        df = pd.DataFrame({
            "delivery_days": [3.5, 5.9, 8.2, 36.5, 59.0, 88.9]
        })

        result = check_target_typos(df)

        self.assertTrue(result["present"])
        self.assertEqual(result["count"], 3)

    def test_target_typos_not_present(self):
        df = pd.DataFrame({
            "delivery_days": [3.5, 5.9, 8.2, 10.5, 11.2]
        })

        result = check_target_typos(df)

        self.assertFalse(result["present"])
        self.assertIsNone(result["count"])

    def test_spelling_variants_detected(self):
        df = pd.DataFrame({
            "service": [
                "express", "Express", " EXPRESS ", "exp.",
                "economy", "Economy", "eco.",
                "standard", "Standard", "sta."
            ]
        })

        result = check_spelling_variants(df)

        self.assertTrue(result["present"])
        self.assertEqual(result["column"], "service")
        self.assertEqual(result["true_levels"], 3)

    def test_spelling_variants_not_present(self):
        df = pd.DataFrame({
            "service": ["express", "economy", "standard"]
        })

        result = check_spelling_variants(df)

        self.assertFalse(result["present"])
        self.assertIsNone(result["column"])
        self.assertIsNone(result["true_levels"])


if __name__ == "__main__":
    unittest.main()