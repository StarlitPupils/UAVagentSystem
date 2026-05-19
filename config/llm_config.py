import os
from dotenv import load_dotenv
from config.settings import config as global_config

load_dotenv()

class LLMConfig:
    PROVIDERS = {
        "deepseek": {
            "api_key_env": "DEEPSEEK_API_KEY",
            "model_env": "DEEPSEEK_MODEL",
            "default_model": "deepseek-reasoner",
            "base_url": "https://api.deepseek.com"
        }
    }

    @classmethod
    def get_provider_config(cls, provider: str):
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        info = cls.PROVIDERS[provider]
        api_key = os.getenv(info["api_key_env"])
        if not api_key:
            raise ValueError(f"Missing {info['api_key_env']} in .env")
        timeout = int(os.getenv("LLM_TIMEOUT", "90"))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        return {
            "provider": provider,
            "api_key": api_key,
            "base_url": info["base_url"],
            "model": os.getenv(info["model_env"], info["default_model"]),
            "timeout": timeout,
            "max_retries": max_retries
        }

    @classmethod
    def get_default_provider(cls):
        return os.getenv("LLM_PROVIDER", "deepseek")