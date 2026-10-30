"""Схемы запроса и ответа: контракт сервиса."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Пол = Literal["Female", "Male"]
ДаНет = Literal["Yes", "No"]
Линии = Literal["Yes", "No", "No phone service"]
Интернет = Literal["DSL", "Fiber optic", "No"]
ДопУслуга = Literal["Yes", "No", "No internet service"]
Контракт = Literal["Month-to-month", "One year", "Two year"]
Оплата = Literal["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"]


class ПризнакиКлиента(BaseModel):
    """Девятнадцать признаков; флаг «новый клиент» сервис считает сам."""

    gender: Пол = Field(..., description="пол")
    SeniorCitizen: int = Field(..., ge=0, le=1,
                               description="пенсионер: 1 — да, 0 — нет")
    Partner: ДаНет = Field(..., description="есть партнёр")
    Dependents: ДаНет = Field(..., description="есть иждивенцы")
    tenure: int = Field(..., ge=0, le=120,
                        description="сколько месяцев клиент с оператором")
    PhoneService: ДаНет = Field(..., description="подключена телефония")
    MultipleLines: Линии = Field(..., description="несколько линий")
    InternetService: Интернет = Field(..., description="тип интернета")
    OnlineSecurity: ДопУслуга = Field(..., description="онлайн-безопасность")
    OnlineBackup: ДопУслуга = Field(..., description="онлайн-бэкап")
    DeviceProtection: ДопУслуга = Field(..., description="защита устройства")
    TechSupport: ДопУслуга = Field(..., description="техподдержка")
    StreamingTV: ДопУслуга = Field(..., description="стриминговое ТВ")
    StreamingMovies: ДопУслуга = Field(..., description="стриминговые фильмы")
    Contract: Контракт = Field(..., description="тип договора")
    PaperlessBilling: ДаНет = Field(..., description="электронный счёт")
    PaymentMethod: Оплата = Field(..., description="способ оплаты")
    MonthlyCharges: float = Field(..., ge=0, le=1000,
                                  description="ежемесячный платёж")
    TotalCharges: float = Field(..., ge=0, le=100000,
                                description="сумма платежей за всё время")

    model_config = {
        "json_schema_extra": {
            "пример": {
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
        }
    }


class Прогноз(BaseModel):
    """Ответ сервиса."""

    churn: Literal[0, 1] = Field(..., description="1 — риск ухода, 0 — нет")
    probability: float = Field(..., ge=0, le=1,
                               description="вероятность ухода")
    threshold: float = Field(..., gt=0, le=1,
                             description="порог отсечения")
    model_version: str = Field(..., description="версия модели")


class Здоровье(BaseModel):
    """Ответ проверки работоспособности."""

    status: str
    model_version: str
    model_loaded: bool


class СведенияОМодели(BaseModel):
    """Служебная информация о модели."""

    model_version: str
    data_version: str
    trained_at: str
    features: list[str]
    target: str
    threshold: float
    metrics: dict[str, float | int]


class Ошибка(BaseModel):
    """Описание ошибки."""

    detail: str
    code: Optional[str] = None
