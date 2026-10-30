# Образ сервиса прогноза оттока
# Собирается одной командой: docker build -t churn-service:1.1.0 .
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Рабочая папка внутри контейнера
WORKDIR /app

# Сначала зависимости: слой кэшируется и не пересобирается
# при каждом изменении кода
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем код и артефакты модели
COPY app/ app/
COPY artifacts/ artifacts/

# Порт, который слушает сервис
EXPOSE 8077

# Проверка работоспособности средствами Docker
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; \
    urllib.request.urlopen('http://127.0.0.1:8077/health')"

# Команда запуска: та же, что локально
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8077"]
