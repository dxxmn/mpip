"""Тесты модели и артефактов: эталонные предсказания и метаданные."""
import hashlib
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from train import порог_под_бюджет

КОРЕНЬ = Path(__file__).resolve().parent.parent
АРТЕФАКТЫ = КОРЕНЬ / "artifacts"


class ТестАртефактов(unittest.TestCase):
    def test_артефакты_существуют(self):
        for имя in ("model.joblib", "meta.json", "etalon.json", "report.json"):
            self.assertTrue((АРТЕФАКТЫ / имя).exists(),
                            f"нет артефакта {имя}")

    def test_метаданные_полные(self):
        мета = json.loads((АРТЕФАКТЫ / "meta.json")
                          .read_text(encoding="utf-8"))
        for ключ in ("версия_модели", "версия_данных", "sha256_данных",
                     "метрики", "порог", "признаки", "цель", "версии"):
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

    def test_отчёт_совпадает_с_данными_и_мета(self):
        отчёт = json.loads((АРТЕФАКТЫ / "report.json")
                           .read_text(encoding="utf-8"))
        мета = json.loads((АРТЕФАКТЫ / "meta.json")
                          .read_text(encoding="utf-8"))
        хеш = hashlib.sha256((КОРЕНЬ / "data" / "telco_churn.csv")
                             .read_bytes()).hexdigest()
        self.assertEqual(отчёт["данные"]["sha256"], хеш)
        self.assertEqual(отчёт["данные"]["строк_исходно"], 7043)
        self.assertEqual(отчёт["данные"]["строк_после_чистки"], 6970)
        self.assertEqual(отчёт["сплит"]["обучение"]
                         + отчёт["сплит"]["валидация"], 6970)
        self.assertEqual(отчёт["метрики"], мета["метрики"])
        self.assertEqual(мета["версии"]["scikit-learn"], sklearn.__version__)
        self.assertEqual(len(отчёт["журнал"]), 4)
        self.assertLessEqual(отчёт["метрики"]["отмечено"],
                             int(0.2 * отчёт["сплит"]["валидация"]))

    def test_порог_не_превышает_бюджет_при_совпадениях(self):
        вероятности = np.array([0.9] * 50 + [0.1] * 50)
        порог = порог_под_бюджет(вероятности, бюджет=0.2)
        self.assertLessEqual(int((вероятности >= порог).sum()), 20)
