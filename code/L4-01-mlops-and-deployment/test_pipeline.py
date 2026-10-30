"""Tests for pipeline.py. Run: python -m unittest -v test_pipeline

These are the kinds of checks a CI job runs on every change: they are fast,
need no network, and fail loudly if a refactor changes behavior.
"""
import tempfile
import unittest

import numpy as np

from pipeline import (Registry, auc, decide, featurize, fingerprint, make_data,
                      promotion_gate, psi, train, validate)

COLS = ["age", "income_k", "tenure_months", "logins_30d", "region"]


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.df = make_data(500, seed=0)

    def test_clean_data_passes(self):
        self.assertEqual(validate(self.df), [])

    def test_missing_column(self):
        self.assertIn("missing column 'age'", validate(self.df.drop(columns="age")))

    def test_unit_change_is_caught(self):
        bad = self.df.assign(income_k=self.df["income_k"] * 1000)
        self.assertTrue(any("income_k" in p and "outside" in p for p in validate(bad)))

    def test_unseen_category(self):
        bad = self.df.copy()
        bad.loc[0, "region"] = "Mars"
        self.assertTrue(any("unseen" in p for p in validate(bad)))

    def test_too_many_missing(self):
        bad = self.df.copy()
        bad.loc[:50, "income_k"] = np.nan          # about 10% missing, limit is 2%
        self.assertTrue(any("missing" in p for p in validate(bad)))

    def test_wrong_type(self):
        bad = self.df.assign(age=self.df["age"].astype(str))
        self.assertTrue(any("expected numbers" in p for p in validate(bad)))


class TestReproducibility(unittest.TestCase):
    def test_same_data_same_fingerprint(self):
        self.assertEqual(fingerprint(make_data(300, 1)), fingerprint(make_data(300, 1)))
        self.assertNotEqual(fingerprint(make_data(300, 1)), fingerprint(make_data(300, 2)))

    def test_fingerprint_ignores_column_order(self):
        df = make_data(300, 1)
        self.assertEqual(fingerprint(df), fingerprint(df[df.columns[::-1]]))

    def test_training_is_deterministic(self):
        df = make_data(2000, 3)
        a, b = train(df, COLS), train(df, COLS)
        np.testing.assert_array_equal(a[-1].coef_, b[-1].coef_)

    def test_model_beats_chance(self):
        df = make_data(4000, 4)
        self.assertGreater(auc(train(df.iloc[:3000], COLS), df.iloc[3000:], COLS), 0.65)

    def test_featurize_one_hot_is_complete(self):
        X = featurize(make_data(200, 5), ["region"])
        np.testing.assert_array_equal(X.sum(axis=1), np.ones(200))


class TestRegistry(unittest.TestCase):
    def test_promote_and_rollback(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            df = make_data(1000, 6)
            v1 = reg.register(train(df, COLS), COLS, fingerprint(df), {}, {})
            v2 = reg.register(train(df, COLS[:2]), COLS[:2], fingerprint(df), {}, {})
            reg.set_alias("champion", v1, "first")
            reg.set_alias("champion", v2, "better")
            self.assertEqual(reg.get("champion")[1]["version"], 2)
            previous = reg.history()[-1]["old"]          # rollback = move the alias back
            reg.set_alias("champion", previous, "rollback")
            self.assertEqual(reg.get("champion")[1]["version"], 1)
            self.assertIsNone(reg.get("challenger"))

    def test_tampered_artifact_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            df = make_data(500, 7)
            v = reg.register(train(df, COLS), COLS, fingerprint(df), {}, {})
            reg.set_alias("champion", v, "first")
            with open(f"{d}/model_v1.pkl", "ab") as f:
                f.write(b"x")
            with self.assertRaises(AssertionError):
                reg.get("champion")


class TestDriftAndDecisions(unittest.TestCase):
    def test_psi_near_zero_without_drift(self):
        rng = np.random.default_rng(0)
        self.assertLess(psi(rng.normal(size=20000), rng.normal(size=20000)), 0.01)

    def test_psi_large_with_drift(self):
        rng = np.random.default_rng(0)
        self.assertGreater(psi(rng.normal(size=20000), rng.normal(1.0, 1, 20000)), 0.25)

    def test_psi_categorical_and_symmetric(self):
        a = np.array(["x"] * 70 + ["y"] * 30)
        b = np.array(["x"] * 40 + ["y"] * 60)
        self.assertAlmostEqual(psi(a, b), psi(b, a))
        self.assertGreater(psi(a, b), 0.25)

    def test_gate(self):
        self.assertFalse(promotion_gate(0.65, None)[0])
        self.assertFalse(promotion_gate(0.75, 0.80)[0])
        self.assertTrue(promotion_gate(0.798, 0.80)[0])

    def test_decisions(self):
        healthy = dict(loss=0.40, control_loss=0.41, control_pred_psi=0.01)
        self.assertEqual(decide(0.0, 0.01, 0.01, canary=healthy)[0], "OK")
        worse = dict(loss=0.50, control_loss=0.41, control_pred_psi=0.01)
        self.assertEqual(decide(0.0, 0.01, 0.01, canary=worse)[0], "ROLLBACK")
        shifted = dict(loss=0.41, control_loss=0.41, control_pred_psi=0.01)
        self.assertEqual(decide(0.0, 0.01, 0.9, canary=shifted)[0], "ROLLBACK")
        self.assertEqual(decide(0.2, 0.01, 0.01)[0], "BLOCK_INPUTS")
        self.assertEqual(decide(0.0, 0.40, 0.01)[0], "ALERT_RETRAIN")   # drift: retrain, not rollback
        self.assertEqual(decide(0.0, 0.15, 0.01)[0], "WATCH")


if __name__ == "__main__":
    unittest.main()
