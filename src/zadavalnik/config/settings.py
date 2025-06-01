from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    BOT_TOKEN: str
    TEST_USER_TGID: int
    OPENAI_API_KEY: str
    DATABASE_URL: str = "sqlite+aiosqlite:///./zadavainik_logs.db"
    # OPENAI_MODEL: str = "deepseek-chat-v3-0324"  # Не работает tool use
    # OPENAI_MODEL: str = "deepseek-r1"  # Не работает tool use

    # OPENAI_MODEL: str = "gemini-2.0-flash-001" # Быстро, недорого. Работает tool use. Для тестов подходит.
    
    OPENAI_MODEL: str = "o4-mini"   # Хорошо, но надо разбираться с tool use
    # OPENAI_MODEL: str = "grok-3-mini-beta"  # Хорошо, но дороговато

    OPENAI_API_URL: str = "https://bothub.chat/api/v2/openai/v1"
    MAX_TESTS_PER_DAY: int = 5 # Максимальное количество тестов в день на пользователя

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings()