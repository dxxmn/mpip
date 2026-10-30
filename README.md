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
| `data/telco_churn.csv` | данные: Telco Customer Churn, 7043 строки (после чистки 6970) |
| `artifacts/` | модель, метаданные и `report.json` с числами карточки; создаются обучением |
| `Dockerfile` | образ сервиса |
| `requirements-dev.txt` | зависимости для тестов и ноутбука (httpx, pytest, matplotlib, ipykernel) |
| `Makefile` | команды одной строкой |

## Модель

Логистическая регрессия в `Pipeline`: заполнение пропусков, one-hot кодирование
и масштабирование учатся только на обучающей части. Строки с повторяющимся
профилем признаков удаляются до сплита (73 строки).

- данные: Telco Customer Churn, 7043 строки, после удаления дублей — 6970, отток 26,24%;
- сплит: 80/20 стратифицированный, `random_state = 42`, 5576 / 1394;
- порог: 0,55 — круглый порог под бюджет обзвона 20% базы (как в карточке ПЗ3);
- метрики на валидации: precision 0,663, recall 0,489, PR-AUC 0,634, ROC-AUC 0,836;
- `report.json` содержит те же числа, что в карточке, и журнал экспериментов.

Порог выбран на той же валидации, на которой считаются метрики. Для отчёта
это стандартный протокол ПЗ3, но precision и recall здесь оптимистичны: для
честной оценки на новых данных нужен отдельный набор для выбора порога.

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

       python -m pip install -r requirements-dev.txt
       python -m unittest discover -s tests -v

Тесты поднимают сервис через TestClient, поэтому отдельный запуск uvicorn
для тестов не нужен. Обучение и тесты одной командой: `make проверка`.

## Ноутбук практики 3

Расчёты лежат в `notebooks/ПЗ3_Задача_и_базлайн.ipynb`, открываются в VS Code
как обычный `.ipynb`. Сервису `matplotlib` и `ipykernel` не нужны, поэтому
они вынесены в `requirements-dev.txt`.

    python -m pip install -r requirements-dev.txt

Или одной командой: `make установка-дев`.

Дальше в VS Code: открыть `notebooks/ПЗ3_Задача_и_базлайн.ipynb`, справа
сверху выбрать ядро (Python из этого же venv) и нажать «Run All».

В Colab достаточно открыть файл и загрузить `telco_churn.csv`, когда
ноутбук об этом попросит: он ищет данные в `../data/`, `data/`, текущей
папке, `/content/` и `/content/data/`. Журнал экспериментов собирается
из расчёта, а не переписывается руками. Все выводы и графики уже сохранены.

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

Ответ: `{"churn": 1, "probability": 0.6055, "threshold": 0.55, "model_version": "churn-logreg-1.1.0"}`.

## Как собрать образ

    docker build -t churn-service:1.1.0 .
    docker run -p 8077:8077 churn-service:1.1.0

Образ собирается на `python:3.14-slim`: версии Python и библиотек совпадают
с теми, на которых обучены артефакты. Зависимости зафиксированы в
`requirements.txt`, поэтому `joblib`-файл не ломается при пересборке.

## Что считается готовым результатом

* модель обучена, артефакты лежат в `artifacts/`;
* сервис отвечает на `/health`, `/predict`, `/model-info`, `/docs`;
* `/health` возвращает 503, если модель или метаданные не загрузились;
* невалидный запрос возвращает 422;
* тесты проходят одной командой;
* образ собирается, контейнер отвечает теми же запросами;
* версия образа связана с версией модели.
