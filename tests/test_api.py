"""Тесты сервиса: успешный путь, ошибки, служебные эндпоинты."""
import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

КЛИЕНТ = TestClient(app)

КОРЕНЬ = Path(__file__).resolve().parent.parent
МЕТА = json.loads((КОРЕНЬ / "artifacts" / "meta.json").read_text(encoding="utf-8"))

# клиент из датасета: новый, помесячный контракт, DSL
ВАЛИДНЫЙ = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


class ТестЗдоровья(unittest.TestCase):
    def test_health_отвечает(self):
        ответ = КЛИЕНТ.get("/health")
        self.assertEqual(ответ.status_code, 200)
        тело = ответ.json()
        self.assertEqual(тело["status"], "ok")
        self.assertTrue(тело["model_loaded"])


class ТестПрогноза(unittest.TestCase):
    def test_валидный_запрос(self):
        ответ = КЛИЕНТ.post("/predict", json=ВАЛИДНЫЙ)
        self.assertEqual(ответ.status_code, 200)
        тело = ответ.json()
        self.assertIn("churn", тело)
        self.assertIn("probability", тело)
        self.assertGreaterEqual(тело["probability"], 0.0)
        self.assertLessEqual(тело["probability"], 1.0)
        self.assertAlmostEqual(тело["threshold"], МЕТА["порог"], places=4)
        self.assertEqual(тело["model_version"], МЕТА["версия_модели"])

    def test_неизвестный_тип_интернета(self):
        данные = dict(ВАЛИДНЫЙ, InternetService="Кабельный")
        ответ = КЛИЕНТ.post("/predict", json=данные)
        self.assertEqual(ответ.status_code, 422)

    def test_нет_обязательного_поля(self):
        данные = dict(ВАЛИДНЫЙ)
        данные.pop("tenure")
        ответ = КЛИЕНТ.post("/predict", json=данные)
        self.assertEqual(ответ.status_code, 422)

    def test_стаж_вне_диапазона(self):
        данные = dict(ВАЛИДНЫЙ, tenure=-1)
        ответ = КЛИЕНТ.post("/predict", json=данные)
        self.assertEqual(ответ.status_code, 422)

    def test_лишнее_поле_игнорируется(self):
        данные = dict(ВАЛИДНЫЙ, лишнее_поле="игнорируется")
        ответ = КЛИЕНТ.post("/predict", json=данные)
        self.assertEqual(ответ.status_code, 200)

    def test_флаг_нового_клиента_не_принимается(self):
        # клиент не может подменить вычисляемый признак
        данные = dict(ВАЛИДНЫЙ, **{"новый_клиент": 0})
        ответ = КЛИЕНТ.post("/predict", json=данные)
        self.assertEqual(ответ.status_code, 200)


class ТестСлужебных(unittest.TestCase):
    def test_model_info(self):
        ответ = КЛИЕНТ.get("/model-info")
        self.assertEqual(ответ.status_code, 200)
        тело = ответ.json()
        self.assertIn("model_version", тело)
        self.assertIn("metrics", тело)
        self.assertEqual(len(тело["features"]), 20)

    def test_docs_доступна(self):
        self.assertEqual(КЛИЕНТ.get("/docs").status_code, 200)
