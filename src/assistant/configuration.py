import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

from enum import Enum

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"

@dataclass(kw_only=True)
class Configuration:
    max_web_research_loops: int = 3
    local_llm: str = os.getenv("LOCAL_LLM", "llama-3-8b-instruct-8k")
    llm_api_base: str = os.getenv("LLM_API_BASE")
    llm_api_key: str = os.getenv("LLM_API_KEY")
    search_api: SearchAPI = SearchAPI.TAVILY


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})