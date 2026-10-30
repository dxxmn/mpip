"""Сервис прогноза оттока: /health, /predict, /model-info, /docs.

Запуск: python -m uvicorn app.main:app --port 8077
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status

from app.schemas import (Здоровье, Ошибка, Прогноз, ПризнакиКлиента,
                         СведенияОМодели)

КОРЕНЬ = Path(__file__).resolve().parent.parent
АРТЕФАКТЫ = КОРЕНЬ / "artifacts"

СОСТОЯНИЕ = {"модель": None, "мета": {}, "ошибка": None}


def загрузить_артефакты():
    """Читает модель и метаданные при импорте, чтобы работали и uvicorn, и TestClient."""
    try:
        модель = joblib.load(АРТЕФАКТЫ / "model.joblib")
        мета = json.loads(
            (АРТЕФАКТЫ / "meta.json").read_text(encoding="utf-8"))
        if "порог" not in мета:
            raise ValueError("в meta.json нет порога")
        СОСТОЯНИЕ.update(модель=модель, мета=мета, ошибка=None)
    except Exception as ошибка:  # noqa: BLE001
        СОСТОЯНИЕ.update(модель=None, мета={}, ошибка=str(ошибка))


@asynccontextmanager
async def жизненный_цикл(app: FastAPI):
    """Перезагружает артефакты при старте сервиса."""
    загрузить_артефакты()
    yield


# Загружаем сразу: тесты и запуск ведут себя одинаково
загрузить_артефакты()

app = FastAPI(
    title="Сервис прогноза оттока",
    version=(СОСТОЯНИЕ["мета"] or {}).get("версия_модели", "unknown"),
    description="Учебный сервис практики 4: модель как API с контрактом.",
    lifespan=жизненный_цикл,
)


def _мета_или_503():
    """Метаданные обязательны: без порога сервис не должен угадывать."""
    мета = СОСТОЯНИЕ.get("мета") or {}
    if not мета or "порог" not in мета:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="метаданные модели недоступны")
    return мета


@app.get("/health", response_model=Здоровье, tags=["служебные"],
         responses={503: {"model": Здоровье,
                          "description": "Модель или метаданные не загружены"}})
def health(response: Response):
    """Проверка работоспособности сервиса и загрузки модели."""
    мета = СОСТОЯНИЕ.get("мета") or {}
    загружена = СОСТОЯНИЕ.get("модель") is not None and "порог" in мета
    if not загружена:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return Здоровье(
        status="ok" if загружена else "degraded",
        model_version=мета.get("версия_модели", "unknown"),
        model_loaded=загружена,
    )


@app.post("/predict", response_model=Прогноз, tags=["прогноз"],
          responses={422: {"description": "Ошибка валидации запроса"},
                     503: {"model": Ошибка}})
def predict(признаки: ПризнакиКлиента):
    """Возвращает вероятность ухода клиента и класс риска."""
    модель = СОСТОЯНИЕ.get("модель")
    if модель is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="модель не загружена")
    мета = _мета_или_503()
    порог = float(мета["порог"])

    # Флаг вычисляем сами, иначе клиент пришлёт противоречивые данные.
    данные = признаки.model_dump()
    данные["новый_клиент"] = int(признаки.tenure == 0)
    # Колонки берём у модели: порядок и состав не расходятся с обучением.
    колонки = list(getattr(модель, "feature_names_in_", данные))
    строка = pd.DataFrame([данные])[колонки]

    вероятность = float(модель.predict_proba(строка)[:, 1][0])
    return Прогноз(
        churn=int(вероятность >= порог),
        probability=round(вероятность, 4),
        threshold=порог,
        model_version=мета.get("версия_модели", "unknown"),
    )


@app.get("/model-info", response_model=СведенияОМодели, tags=["служебные"],
         responses={503: {"model": Ошибка}})
def model_info():
    """Сведения о модели: версии, метрики, порог."""
    мета = _мета_или_503()
    return СведенияОМодели(
        model_version=мета.get("версия_модели", "unknown"),
        data_version=мета.get("версия_данных", "unknown"),
        trained_at=мета.get("обучена", "unknown"),
        features=мета.get("признаки", []),
        target=мета.get("цель", "ушёл"),
        threshold=float(мета["порог"]),
        metrics=мета.get("метрики", {}),
    )
