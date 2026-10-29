# Сервис прогноза оттока

Проект команды БИВТ-23-АЭ-1 (группа № 3) по сквозному проекту «Система
прогнозирования оттока клиентов телеком-оператора».

Модель из практики 3, упакованная в API-сервис и контейнер: практика 4.

## Что внутри

| Файл | Назначение |
| --- | --- |
| `train.py` | обучение модели и упаковка артефактов |
| `app/main.py` | сервис на FastAPI: `/health`, `/predict`, `/model-info` |
| `app/schemas.py` | схема запроса и ответа, валидация входа |
| `tests/test_api.py` | тесты сервиса |
| `tests/test_model.py` | тесты модели и артефактов |
| `notebooks/ПЗ3_Задача_и_базлайн.ipynb` | постановка задачи, базлайн, модель, выбор порога |
| `data/telco_churn.csv` | данные: Telco Customer Churn, 7043 строки |
| `artifacts/` | модель и метаданные, создаются обучением |
| `Dockerfile` | образ сервиса |
| `requirements-dev.txt` | зависимости для ноутбука (matplotlib, ipykernel) |
| `Makefile` | команды одной строкой |

## Модель

Логистическая регрессия в `Pipeline`: заполнение пропусков, one-hot кодирование
и масштабирование учатся только на обучающей части.

- данные: Telco Customer Churn, 7043 клиента, отток 26,54%;
- сплит: 80/20 стратифицированный, `random_state = 42`;
- порог: 0,5285 — подобран под бюджет обзвона 20% базы (как в карточке ПЗ3);
- метрики на валидации: precision 0,662, recall 0,497, PR-AUC 0,634, ROC-AUC 0,842.

Флаг `новый_клиент` (`tenure == 0`) вычисляется сервисом, а не приходит в запросе:
так клиент не может прислать противоречивые данные.

## Как запустить локально

1. Создать окружение и установить зависимости:

       python -m venv .venv
       source .venv/bin/activate        # Windows: .venv\Scripts\activate
       python -m pip install -r requirements.txt

2. Обучить модель:

       python train.py

3. Запустить сервис:

       python -m uvicorn app.main:app --host 127.0.0.1 --port 8077

4. Проверить:

       curl http://127.0.0.1:8077/health

## Как проверить тестами

       python -m unittest discover -s tests -v

Тесты поднимают сервис через TestClient, поэтому отдельный запуск uvicorn
для тестов не нужен.

## Ноутбук практики 3

Расчёты лежат в `notebooks/ПЗ3_Задача_и_базлайн.ipynb`, открываются в VS Code
как обычный `.ipynb`. Сервису `matplotlib` и `ipykernel` не нужны, поэтому
они вынесены в отдельный файл.

    python -m pip install -r requirements.txt -r requirements-dev.txt

Или одной командой: `make установка-дев`.

Дальше в VS Code: открыть `notebooks/ПЗ3_Задача_и_базлайн.ipynb`, справа
сверху выбрать ядро (Python из этого же venv) и нажать «Run All».

В Colab достаточно открыть файл и загрузить `telco_churn.csv`, когда
ноутбук об этом попросит: он ищет данные в `../data/`, `data/`, рядом
с собой и в `/content/`. Все выводы и графики уже сохранены в файле.

## Пример запроса

    curl -X POST http://127.0.0.1:8077/predict \
      -H "Content-Type: application/json" \
      -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
           "tenure":1,"PhoneService":"No","MultipleLines":"No phone service",
           "InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes",
           "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
           "StreamingMovies":"No","Contract":"Month-to-month",
           "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
           "MonthlyCharges":29.85,"TotalCharges":29.85}'

Ответ: `{"churn": 1, "probability": 0.6146, "threshold": 0.528485, "model_version": "churn-logreg-1.0.0"}`.

## Как собрать образ

    docker build -t churn-service:1.0.0 .
    docker run -p 8077:8077 churn-service:1.0.0

## Что считается готовым результатом

* модель обучена, артефакты лежат в `artifacts/`;
* сервис отвечает на `/health`, `/predict`, `/model-info`, `/docs`;
* невалидный запрос возвращает 422;
* тесты проходят одной командой;
* образ собирается, контейнер отвечает теми же запросами;
* версия образа связана с версией модели.
