"""Обучение модели оттока, те же данные и порог, что в ноутбуке ПЗ3.

Запуск: python train.py
Результат: artifacts/model.joblib, meta.json, etalon.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

КОРЕНЬ = Path(__file__).resolve().parent
ДАННЫЕ = КОРЕНЬ / "data" / "telco_churn.csv"
АРТЕФАКТЫ = КОРЕНЬ / "artifacts"

ЧИСЛА = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
         "новый_клиент"]
КАТЕГОРИИ = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
             "InternetService", "OnlineSecurity", "OnlineBackup",
             "DeviceProtection", "TechSupport", "StreamingTV",
             "StreamingMovies", "Contract", "PaperlessBilling",
             "PaymentMethod"]
ЦЕЛЬ = "ушёл"
БЮДЖЕТ = 0.20                 # доля базы, которую смена успевает обзвонить
ВЕРСИЯ_МОДЕЛИ = "churn-logreg-1.0.0"
ЗЕРНО = 42


def загрузить_данные(путь=ДАННЫЕ):
    """Читает CSV и возвращает таблицу как есть."""
    if not путь.exists():
        raise SystemExit(
            f"нет файла данных: {путь}\n"
            "положите telco_churn.csv в папку data")
    return pd.read_csv(путь, keep_default_na=False)


def подготовить(таблица):
    """TotalCharges -> число; пустые (tenure = 0) -> 0 плюс флаг; Churn -> 0/1."""
    таблица = таблица.copy()
    таблица["TotalCharges"] = pd.to_numeric(
        таблица["TotalCharges"].replace(" ", pd.NA), errors="coerce")
    таблица["новый_клиент"] = (таблица["tenure"] == 0).astype(int)
    таблица["TotalCharges"] = таблица["TotalCharges"].fillna(0)

    таблица[ЦЕЛЬ] = (таблица["Churn"] == "Yes").astype(int)
    таблица = таблица.drop(columns=["customerID", "Churn"])
    return таблица


def собрать_модель():
    """Pipeline: предобработка и модель в одном объекте."""
    подготовка = ColumnTransformer([
        ("числа", Pipeline([
            ("пропуски", SimpleImputer(strategy="median")),
            ("масштаб", StandardScaler()),
        ]), ЧИСЛА),
        ("категории", Pipeline([
            ("пропуски", SimpleImputer(strategy="most_frequent")),
            ("коды", OneHotEncoder(handle_unknown="ignore")),
        ]), КАТЕГОРИИ),
    ])
    return Pipeline([
        ("подготовка", подготовка),
        ("модель", LogisticRegression(max_iter=1000, random_state=ЗЕРНО)),
    ])


def порог_под_бюджет(вероятности, бюджет=БЮДЖЕТ):
    """Порог по верхним k клиентам: k = бюджет * размер валидации."""
    k = int(np.floor(бюджет * len(вероятности)))
    if k <= 0:
        raise ValueError("бюджет слишком мал для такой валидации")
    return float(np.sort(вероятности)[::-1][k - 1])


def метрики(факт, вероятность, порог):
    """Метрики на выбранном пороге плюс качество ранжирования."""
    прогноз = (вероятность >= порог).astype(int)
    return {
        "precision": round(float(precision_score(факт, прогноз, zero_division=0)), 4),
        "recall": round(float(recall_score(факт, прогноз, zero_division=0)), 4),
        "f1": round(float(f1_score(факт, прогноз, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(факт, вероятность)), 4),
        "pr_auc": round(float(average_precision_score(факт, вероятность)), 4),
        "доля_отмеченных": round(float(прогноз.mean()) * 100, 2),
    }


def обучить(таблица):
    """Обучает модель и возвращает её вместе с порогом, метриками и сплитом."""
    X_обуч, X_вал, y_обуч, y_вал = train_test_split(
        таблица[ЧИСЛА + КАТЕГОРИИ], таблица[ЦЕЛЬ],
        test_size=0.2, random_state=ЗЕРНО, stratify=таблица[ЦЕЛЬ])
    модель = собрать_модель()
    модель.fit(X_обуч, y_обуч)
    вероятность = модель.predict_proba(X_вал)[:, 1]
    порог = порог_под_бюджет(вероятность)
    return модель, порог, метрики(y_вал.to_numpy(), вероятность, порог), (X_вал, y_вал)


def эталонные_предсказания(модель, X_вал, порог):
    """Пять примеров с ожидаемыми вероятностями для тестов."""
    примеры = []
    образец = X_вал.head(5).reset_index(drop=True)
    for i in range(len(образец)):
        строка = образец.iloc[[i]]
        вероятность = float(модель.predict_proba(строка)[:, 1][0])
        признаки = {}
        for ключ, значение in образец.iloc[i].items():
            if pd.isna(значение):
                признаки[ключ] = None
            elif isinstance(значение, (bool, np.bool_)):
                признаки[ключ] = int(значение)
            elif isinstance(значение, (int, np.integer)):
                признаки[ключ] = int(значение)
            elif isinstance(значение, (float, np.floating)):
                признаки[ключ] = float(значение)
            else:
                признаки[ключ] = str(значение)
        примеры.append({
            "features": признаки,
            "churn": int(вероятность >= порог),
            "probability": round(вероятность, 4),
        })
    return примеры


def main():
    сырые = загрузить_данные()
    таблица = подготовить(сырые)
    print(f"данные: {ДАННЫЕ.name}, строк {len(таблица)}")
    print(f"признаков: {len(ЧИСЛА)} числовых и {len(КАТЕГОРИИ)} категориальных")

    модель, порог, мет, (X_вал, _) = обучить(таблица)
    print(f"порог под бюджет {int(БЮДЖЕТ * 100)}%: {порог:.4f}")
    print("метрики на валидации:", мет)

    АРТЕФАКТЫ.mkdir(parents=True, exist_ok=True)
    joblib.dump(модель, АРТЕФАКТЫ / "model.joblib")
    (АРТЕФАКТЫ / "etalon.json").write_text(
        json.dumps(эталонные_предсказания(модель, X_вал, порог),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (АРТЕФАКТЫ / "meta.json").write_text(json.dumps({
        "версия_модели": ВЕРСИЯ_МОДЕЛИ,
        "версия_данных": f"{ДАННЫЕ.name}:{len(таблица)}",
        "обучена": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "признаки": ЧИСЛА + КАТЕГОРИИ,
        "цель": ЦЕЛЬ,
        "порог": round(порог, 6),
        "метрики": мет,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("артефакты:", sorted(p.name for p in АРТЕФАКТЫ.iterdir()))


if __name__ == "__main__":
    main()
