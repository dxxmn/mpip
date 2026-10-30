"""Обучение модели оттока, те же данные и порог, что в ноутбуке ПЗ3.

Строки с повторяющимся профилем удаляются до сплита.
Запуск: python train.py
Результат: artifacts/model.joblib, meta.json, etalon.json, report.json
"""
import hashlib
import json
import platform
import sys
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
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
ВЕРСИЯ_МОДЕЛИ = "churn-logreg-1.1.0"
ЗЕРНО = 42

# Пороги, которые попадают в таблицу отчёта и в карточку ПЗ3.
ПОРОГИ_ДЛЯ_ОТЧЁТА = (0.40, 0.50, 0.60)


def загрузить_данные(путь=ДАННЫЕ):
    """Читает CSV и возвращает таблицу как есть."""
    if not путь.exists():
        raise SystemExit(
            f"нет файла данных: {путь}\n"
            "положите telco_churn.csv в папку data")
    return pd.read_csv(путь, keep_default_na=False)


def _sha256(путь):
    """Хеш данных: по нему видно, на какой версии файла обучена модель."""
    return hashlib.sha256(Path(путь).read_bytes()).hexdigest()


def подготовить(таблица):
    """TotalCharges -> число; пустые (tenure = 0) -> 0 плюс флаг; Churn -> 0/1.

    Строки с одинаковым профилем признаков удаляются до сплита: одинаковые
    данные с разными исходами мешают честной оценке.
    """
    таблица = таблица.copy()
    таблица["TotalCharges"] = pd.to_numeric(
        таблица["TotalCharges"].replace(" ", pd.NA), errors="coerce")
    таблица["новый_клиент"] = (таблица["tenure"] == 0).astype(int)
    таблица["TotalCharges"] = таблица["TotalCharges"].fillna(0)

    таблица[ЦЕЛЬ] = (таблица["Churn"] == "Yes").astype(int)
    таблица = таблица.drop(columns=["customerID", "Churn"])

    признаки = [c for c in таблица.columns if c != ЦЕЛЬ]
    дубли = таблица.duplicated(subset=признаки, keep=False)
    таблица = таблица.loc[~дубли].reset_index(drop=True)
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
    """Порог по верхним k клиентам: k = бюджет * размер валидации.

    Порог округляется вверх до сотых. Если на границе есть одинаковые
    вероятности, порог поднимается, чтобы не отметить больше k клиентов:
    бюджет — это ограничение сверху, а не точное число звонков.
    """
    вероятности = np.asarray(вероятности, dtype=float)
    k = int(np.floor(бюджет * len(вероятности)))
    if k <= 0:
        raise ValueError("бюджет слишком мал для такой валидации")
    if k >= len(вероятности):
        return 0.0

    порог = float(np.sort(вероятности)[::-1][k - 1])
    порог = float(np.ceil(порог * 100) / 100)
    # Совпадения на границе могут дать больше k отмеченных.
    while (вероятности >= порог).sum() > k and порог < 1.0:
        порог = round(min(1.0, порог + 0.01), 2)
    return порог


def метрики_решения(факт, прогноз):
    """Метрики готового бинарного решения без качества ранжирования."""
    факт = np.asarray(факт, dtype=int)
    прогноз = np.asarray(прогноз, dtype=int)
    tn, fp, fn, tp = confusion_matrix(факт, прогноз, labels=[0, 1]).ravel()
    return {
        "precision": round(float(precision_score(факт, прогноз,
                                                 zero_division=0)), 6),
        "recall": round(float(recall_score(факт, прогноз, zero_division=0)), 6),
        "f1": round(float(f1_score(факт, прогноз, zero_division=0)), 6),
        "отмечено": int(прогноз.sum()),
        "доля_отмеченных": round(float(прогноз.mean()) * 100, 2),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def метрики(факт, вероятность, порог):
    """Метрики на выбранном пороге плюс качество ранжирования."""
    факт = np.asarray(факт, dtype=int)
    вероятность = np.asarray(вероятность, dtype=float)
    итог = метрики_решения(факт, (вероятность >= порог).astype(int))
    итог["roc_auc"] = round(float(roc_auc_score(факт, вероятность)), 4)
    итог["pr_auc"] = round(float(average_precision_score(факт, вероятность)), 4)
    return итог


def обучить(таблица):
    """Обучает модель и возвращает её, порог, вероятности и обе части сплита."""
    X_обуч, X_вал, y_обуч, y_вал = train_test_split(
        таблица[ЧИСЛА + КАТЕГОРИИ], таблица[ЦЕЛЬ],
        test_size=0.2, random_state=ЗЕРНО, stratify=таблица[ЦЕЛЬ])
    модель = собрать_модель()
    модель.fit(X_обуч, y_обуч)
    вероятность = модель.predict_proba(X_вал)[:, 1]
    порог = порог_под_бюджет(вероятность)
    return модель, порог, вероятность, X_вал, y_вал, y_обуч


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


def _строка_числа(значение, знаков=3):
    """Число в русской записи: 0,663 вместо 0.663, округление половины вверх."""
    квант = Decimal(1).scaleb(-знаков)
    число = Decimal(str(float(значение))).quantize(квант,
                                                  rounding=ROUND_HALF_UP)
    return str(число).replace(".", ",")


def собрать_отчёт(таблица, сырые, порог, вероятность, X_вал, y_вал,
                  метрики_порога):
    """Считает все числа карточки ПЗ3 и журнала экспериментов."""
    y = np.asarray(y_вал, dtype=int)
    n = len(y)
    вероятность = np.asarray(вероятность, dtype=float)

    # Базлайн 1: «все остаются».
    все_нули = np.zeros(n, dtype=int)
    базлайн_все = метрики_решения(y, все_нули)
    базлайн_все["accuracy"] = round(float((y == все_нули).mean()) * 100, 2)
    базлайн_все["pr_auc"] = None

    # Базлайн 2: правило «помесячный контракт -> риск».
    правило = (X_вал["Contract"] == "Month-to-month").to_numpy().astype(int)
    базлайн_правило = метрики_решения(y, правило)
    базлайн_правило["pr_auc"] = round(
        float(average_precision_score(y, правило)), 4)

    # Перебор порогов: фиксированные точки плюс выбранный порог.
    пороги = sorted(set([round(float(t), 2) for t in ПОРОГИ_ДЛЯ_ОТЧЁТА]
                        + [round(float(порог), 2)]))
    таблица_порогов = []
    for t in пороги:
        строка = {"порог": t}
        строка.update(метрики_решения(y, (вероятность >= t).astype(int)))
        таблица_порогов.append(строка)

    # Максимум F1 — для ответа «почему не взяли порог по F1».
    лучший_f1 = None
    for t in np.unique(вероятность):
        прогноз = (вероятность >= t).astype(int)
        f1 = float(f1_score(y, прогноз, zero_division=0))
        if лучший_f1 is None or f1 > лучший_f1["f1"]:
            лучший_f1 = {
                "f1": round(f1, 4),
                "порог": round(float(t), 4),
                "отмечено": int(прогноз.sum()),
                "доля_отмеченных": round(float(прогноз.mean()) * 100, 2),
            }

    # Сравнение при равном бюджете: правило не ранжирует, поэтому из его
    # флагов берём случайную долю, равную бюджету.
    отмечено_модели = int((вероятность >= порог).sum())
    помечено_правилом = int(правило.sum())
    if помечено_правилом == 0:
        raise ValueError("правило не отметило ни одного клиента")
    доля_правила = отмечено_модели / помечено_правилом
    попаданий_правила = базлайн_правило["TP"] * доля_правила
    попаданий_модели = метрики_порога["TP"]
    равный_бюджет = {
        "звонков": отмечено_модели,
        "доля_правила": round(float(доля_правила), 4),
        "попаданий_правила": round(float(попаданий_правила), 1),
        "precision_правила": базлайн_правило["precision"],
        "попаданий_модели": int(попаданий_модели),
        "precision_модели": метрики_порога["precision"],
        "прирост_попаданий_проц": round(
            float((попаданий_модели / попаданий_правила - 1) * 100), 1),
    }

    # Журнал: числа не печатаем руками, они берутся из расчёта.
    def строка_журнала(номер, что, мет, вывод):
        return {"№": номер, "что_пробовали": что, "метрики": мет,
                "вывод": вывод}

    ж_все = строка_журнала(
        1, "Базлайн «все остаются»",
        f"accuracy {_строка_числа(базлайн_все['accuracy'], 1)}%, recall 0",
        "планка бессмысленности: высокая accuracy при нулевой пользе")
    вывод_правила = (f"ловит много, но берёт "
                     f"{_строка_числа(базлайн_правило['доля_отмеченных'], 1)}% базы")
    вывод_правила += (" — не влезает в бюджет"
                      if базлайн_правило["доля_отмеченных"] > БЮДЖЕТ * 100
                      else " — влезает в бюджет")
    ж_правило = строка_журнала(
        2, "Правило «помесячный контракт → риск»",
        (f"precision {_строка_числа(базлайн_правило['precision'])}, "
         f"recall {_строка_числа(базлайн_правило['recall'])}, "
         f"отмечено {базлайн_правило['отмечено']}"),
        вывод_правила)
    строка_05 = таблица_порогов[пороги.index(0.50)]
    сравнение = ("лучше" if строка_05["precision"]
                 > базлайн_правило["precision"] else "не лучше")
    вывод_05 = (f"{сравнение} правила по precision, но "
                f"{_строка_числа(строка_05['доля_отмеченных'], 1)}% базы")
    вывод_05 += (" — больше бюджета" if строка_05["доля_отмеченных"] > БЮДЖЕТ * 100
                 else " — влезает в бюджет")
    ж_модель_05 = строка_журнала(
        3, "Логистическая регрессия, порог 0,5",
        (f"precision {_строка_числа(строка_05['precision'])}, "
         f"recall {_строка_числа(строка_05['recall'])}, "
         f"отмечено {строка_05['отмечено']}"),
        вывод_05)
    ж_модель_бюджет = строка_журнала(
        4, (f"Модель, порог {_строка_числа(порог, 2)} "
            f"(бюджет {int(БЮДЖЕТ * 100)}%)"),
        (f"precision {_строка_числа(метрики_порога['precision'])}, "
         f"recall {_строка_числа(метрики_порога['recall'])}, "
         f"отмечено {метрики_порога['отмечено']}"),
        (f"выбранный вариант: {попаданий_модели} попаданий против "
         f"~{round(float(попаданий_правила))} у правила"))

    отчёт = {
        "версия_модели": ВЕРСИЯ_МОДЕЛИ,
        "данные": {
            "файл": ДАННЫЕ.name,
            "sha256": _sha256(ДАННЫЕ),
            "строк_исходно": int(len(сырые)),
            "строк_после_чистки": int(len(таблица)),
            "удалено_дублей": int(len(сырые) - len(таблица)),
            "отток_доля_проц": round(float(таблица[ЦЕЛЬ].mean()) * 100, 2),
        },
        "сплит": {
            "seed": ЗЕРНО,
            "test_size": 0.2,
            "обучение": int(len(таблица) - n),
            "валидация": int(n),
            "отток_обучение_проц": None,
            "отток_валидация_проц": round(float(y.mean()) * 100, 2),
        },
        "порог": round(float(порог), 5),
        "метрики": метрики_порога,
        "таблица_порогов": таблица_порогов,
        "условия_достаточности": {
            "бюджет_проц": round(БЮДЖЕТ * 100),
            "recall_не_ниже": 0.45,
            "precision_не_ниже": 0.60,
            "выполнено": bool(метрики_порога["recall"] >= 0.45
                              and метрики_порога["precision"] >= 0.60),
        },
        "базлайны": {
            "все_остаются": базлайн_все,
            "правило_помесячный_контракт": базлайн_правило,
        },
        "лучший_f1": лучший_f1,
        "равный_бюджет": равный_бюджет,
        "журнал": [ж_все, ж_правило, ж_модель_05, ж_модель_бюджет],
        "версии": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    return отчёт


def main():
    сырые = загрузить_данные()
    таблица = подготовить(сырые)
    print(f"данные: {ДАННЫЕ.name}, строк {len(таблица)}")
    print(f"удалено строк с повторяющимся профилем: {len(сырые) - len(таблица)}")
    print(f"признаков: {len(ЧИСЛА)} числовых и {len(КАТЕГОРИИ)} категориальных")

    модель, порог, вероятность, X_вал, y_вал, y_обуч = обучить(таблица)
    мет = метрики(y_вал.to_numpy(), вероятность, порог)
    print(f"порог под бюджет {int(БЮДЖЕТ * 100)}%: {порог:.4f}")
    print("метрики на валидации:", мет)

    отчёт = собрать_отчёт(таблица, сырые, порог, вероятность, X_вал, y_вал,
                          мет)
    # Доля оттока в обучении — по тому же сплиту, что и модель.
    отчёт["сплит"]["отток_обучение_проц"] = round(
        float(y_обуч.mean()) * 100, 2)

    АРТЕФАКТЫ.mkdir(parents=True, exist_ok=True)
    joblib.dump(модель, АРТЕФАКТЫ / "model.joblib")
    (АРТЕФАКТЫ / "etalon.json").write_text(
        json.dumps(эталонные_предсказания(модель, X_вал, порог),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (АРТЕФАКТЫ / "report.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8")
    (АРТЕФАКТЫ / "meta.json").write_text(json.dumps({
        "версия_модели": ВЕРСИЯ_МОДЕЛИ,
        "версия_данных": f"{ДАННЫЕ.name}:{len(таблица)}",
        "sha256_данных": отчёт["данные"]["sha256"],
        "обучена": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "признаки": ЧИСЛА + КАТЕГОРИИ,
        "цель": ЦЕЛЬ,
        "порог": round(порог, 5),
        "метрики": мет,
        "версии": отчёт["версии"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("артефакты:", sorted(p.name for p in АРТЕФАКТЫ.iterdir()))


if __name__ == "__main__":
    sys.exit(main())
