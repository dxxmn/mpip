# Учебный проект практики 4: команды одной строкой
# Использование: make установка, make обучение, make тесты, make запуск

PYTHON ?= python
ОБРАЗ ?= churn-service
ВЕРСИЯ ?= 1.0.0

.PHONY: установка установка-дев обучение тесты запуск остановка образ контейнер откат

установка:
	$(PYTHON) -m pip install -r requirements.txt

установка-дев:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

обучение:
	$(PYTHON) train.py

тесты:
	$(PYTHON) -m unittest discover -s tests -v

запуск:
	$(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port 8077

образ:
	docker build -t $(ОБРАЗ):$(ВЕРСИЯ) .

контейнер:
	docker run --rm -p 8077:8077 $(ОБРАЗ):$(ВЕРСИЯ)

откат:
	@echo "Запускаем предыдущую версию $(ОБРАЗ):$(ВЕРСИЯ)"
	docker run --rm -p 8077:8077 $(ОБРАЗ):$(ВЕРСИЯ)
