"""Сервис прогноза оттока: /health, /predict, /model-info, /docs.

Запуск: python -m uvicorn app.main:app --port 8077
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (Здоровье, Ошибка, Прогноз, ПризнакиКлиента,
                         СведенияОМодели)

КОРЕНЬ = Path(__file__).resolve().parent.parent
АРТЕФАКТЫ = КОРЕНЬ / "artifacts"

# Запасное значение: в норме порог всегда берётся из meta.json
ПОРОГ_ПО_УМОЛЧАНИЮ = 0.53

СОСТОЯНИЕ = {}


def загрузить_артефакты():
    """Читает модель и метаданные при импорте, чтобы работали и uvicorn, и TestClient."""
    try:
        СОСТОЯНИЕ["модель"] = joblib.load(АРТЕФАКТЫ / "model.joblib")
        СОСТОЯНИЕ["мета"] = json.loads(
            (АРТЕФАКТЫ / "meta.json").read_text(encoding="utf-8"))
        СОСТОЯНИЕ["ошибка"] = None
    except Exception as ошибка:  # noqa: BLE001
        СОСТОЯНИЕ["модель"] = None
        СОСТОЯНИЕ["мета"] = {}
        СОСТОЯНИЕ["ошибка"] = str(ошибка)


@asynccontextmanager
async def жизненный_цикл(app: FastAPI):
    """Перезагружает артефакты при старте сервиса."""
    загрузить_артефакты()
    yield


# Загружаем сразу: тесты и запуск ведут себя одинаково
загрузить_артефакты()

app = FastAPI(
    title="Сервис прогноза оттока",
    version="1.0.0",
    description="Учебный сервис практики 4: модель как API с контрактом.",
    lifespan=жизненный_цикл,
)


@app.get("/health", response_model=Здоровье, tags=["служебные"])
def health():
    """Проверка работоспособности сервиса и загрузки модели."""
    мета = СОСТОЯНИЕ.get("мета") or {}
    return Здоровье(
        status="ok" if СОСТОЯНИЕ.get("модель") is not None else "degraded",
        model_version=мета.get("версия_модели", "unknown"),
        model_loaded=СОСТОЯНИЕ.get("модель") is not None,
    )


@app.post("/predict", response_model=Прогноз, tags=["прогноз"],
          responses={422: {"model": Ошибка}, 503: {"model": Ошибка}})
def predict(признаки: ПризнакиКлиента):
    """Возвращает вероятность ухода клиента и класс риска."""
    модель = СОСТОЯНИЕ.get("модель")
    if модель is None:
        raise HTTPException(status_code=503,
                            detail="модель не загружена")
    мета = СОСТОЯНИЕ.get("мета") or {}
    порог = float(мета.get("порог", ПОРОГ_ПО_УМОЛЧАНИЮ))

    # флаг вычисляем сами, иначе клиент пришлёт противоречивые данные
    строка = pd.DataFrame([{
        "gender": признаки.gender,
        "SeniorCitizen": признаки.SeniorCitizen,
        "Partner": признаки.Partner,
        "Dependents": признаки.Dependents,
        "tenure": признаки.tenure,
        "PhoneService": признаки.PhoneService,
        "MultipleLines": признаки.MultipleLines,
        "InternetService": признаки.InternetService,
        "OnlineSecurity": признаки.OnlineSecurity,
        "OnlineBackup": признаки.OnlineBackup,
        "DeviceProtection": признаки.DeviceProtection,
        "TechSupport": признаки.TechSupport,
        "StreamingTV": признаки.StreamingTV,
        "StreamingMovies": признаки.StreamingMovies,
        "Contract": признаки.Contract,
        "PaperlessBilling": признаки.PaperlessBilling,
        "PaymentMethod": признаки.PaymentMethod,
        "MonthlyCharges": признаки.MonthlyCharges,
        "TotalCharges": признаки.TotalCharges,
        "новый_клиент": int(признаки.tenure == 0),
    }])
    вероятность = float(модель.predict_proba(строка)[:, 1][0])
    return Прогноз(
        churn=int(вероятность >= порог),
        probability=round(вероятность, 4),
        threshold=порог,
        model_version=мета.get("версия_модели", "unknown"),
    )


@app.get("/model-info", response_model=СведенияОМодели, tags=["служебные"])
def model_info():
    """Сведения о модели: версии, метрики, порог."""
    мета = СОСТОЯНИЕ.get("мета")
    if not мета:
        raise HTTPException(status_code=503, detail="метаданные недоступны")
    return СведенияОМодели(
        model_version=мета.get("версия_модели", "unknown"),
        data_version=мета.get("версия_данных", "unknown"),
        trained_at=мета.get("обучена", "unknown"),
        features=мета.get("признаки", []),
        target=мета.get("цель", "ушёл"),
        threshold=float(мета.get("порог", ПОРОГ_ПО_УМОЛЧАНИЮ)),
        metrics=мета.get("метрики", {}),
    )
