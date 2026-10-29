"""Тесты модели и артефактов: эталонные предсказания и метаданные."""
import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

КОРЕНЬ = Path(__file__).resolve().parent.parent
АРТЕФАКТЫ = КОРЕНЬ / "artifacts"


class ТестАртефактов(unittest.TestCase):
    def test_артефакты_существуют(self):
        for имя in ("model.joblib", "meta.json", "etalon.json"):
            self.assertTrue((АРТЕФАКТЫ / имя).exists(),
                            f"нет артефакта {имя}")

    def test_метаданные_полные(self):
        мета = json.loads((АРТЕФАКТЫ / "meta.json")
                          .read_text(encoding="utf-8"))
        for ключ in ("версия_модели", "версия_данных", "метрики", "порог",
                     "признаки", "цель"):
            self.assertIn(ключ, мета)
        self.assertEqual(мета["цель"], "ушёл")
        self.assertEqual(len(мета["признаки"]), 20)
        self.assertGreater(мета["порог"], 0.0)
        self.assertLess(мета["порог"], 1.0)

    def test_метрики_на_месте(self):
        мета = json.loads((АРТЕФАКТЫ / "meta.json")
                          .read_text(encoding="utf-8"))
        мет = мета["метрики"]
        self.assertGreater(мет["recall"], 0.4)
        self.assertGreater(мет["precision"], 0.5)
        self.assertGreater(мет["pr_auc"], 0.5)

    def test_эталонные_предсказания(self):
        модель = joblib.load(АРТЕФАКТЫ / "model.joblib")
        эталон = json.loads((АРТЕФАКТЫ / "etalon.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(len(эталон), 5)
        for пример in эталон:
            таблица = pd.DataFrame([пример["features"]])
            вероятность = float(модель.predict_proba(таблица)[:, 1][0])
            self.assertAlmostEqual(вероятность, пример["probability"],
                                   delta=0.01)
