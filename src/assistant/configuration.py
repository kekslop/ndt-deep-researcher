import os
from dataclasses import dataclass, fields
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from enum import Enum


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"


@dataclass(kw_only=True)
class Configuration:
    # Маппинг переменных окружения
    ENV_MAPPING = {
        "llm_api_base": "OPENAI_API_BASE",
        "llm_api_key": "OPENAI_API_KEY",
        "local_llm": "OPENAI_MODEL",
    }

    # Дефолтные значения
    DEFAULTS = {
        "max_web_research_loops": 3,
        "local_llm": "llama-3-8b-instruct-8k",
        "llm_api_base": "https://llama3gpu.neuraldeep.tech/v1",
        "search_api": SearchAPI.TAVILY
    }

    max_web_research_loops: int = DEFAULTS["max_web_research_loops"]
    local_llm: str = DEFAULTS["local_llm"]
    llm_api_base: str = DEFAULTS["llm_api_base"]
    llm_api_key: str = ""
    search_api: SearchAPI = DEFAULTS["search_api"]

    @classmethod
    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """
        Создает экземпляр Configuration с приоритетом:
        1. Значения из configurable (UI)
        2. Переменные окружения
        3. Дефолтные значения
        """
        # Получаем значения из configurable
        configurable = config.get("configurable", {}) if config else {}

        print(f"Config from UI: {configurable}")  # Отладочный вывод

        values: dict[str, Any] = {}

        for field in fields(cls):
            if not field.init:
                continue

            field_name = field.name

            # 1. Сначала проверяем значение из UI (configurable)
            config_value = configurable.get(field_name)
            if config_value is not None and config_value != "":
                values[field_name] = config_value
                continue

            # 2. Затем проверяем значение из переменных окружения
            env_var_name = cls.ENV_MAPPING.get(field_name, field_name.upper())
            env_value = os.getenv(env_var_name)
            if env_value is not None and env_value != "":
                values[field_name] = env_value
                continue

            # 3. Наконец, используем дефолтное значение
            if field_name in cls.DEFAULTS:
                values[field_name] = cls.DEFAULTS[field_name]

        print(f"Final values: {values}")  # Отладочный вывод

        return cls(**values)

    def __post_init__(self):
        """Валидация после инициализации"""
        if not self.llm_api_base:
            raise ValueError("LLM API Base URL is required")
        if not self.llm_api_key:
            raise ValueError("LLM API Key is required")

        # Форматирование URL
        if not self.llm_api_base.startswith(('http://', 'https://')):
            self.llm_api_base = f"https://{self.llm_api_base}"
        self.llm_api_base = self.llm_api_base.rstrip('/')