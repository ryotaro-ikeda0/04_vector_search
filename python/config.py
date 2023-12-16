from pydantic import BaseSettings


class Settings(BaseSettings):
    # Azure OpenAI 検証環境 接続情報
    AZURE_OPENAI_API_INSTANCE_NAME: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_API_DEPLOYMENT_NAME: str
    AZURE_OPENAI_API_MODEL_NAME: str

    EMBEDDING_DEPLOYMENT_NAME: str
    EMBEDDING_MODEL_NAME: str

    QDRANT_URL: str

    PORT_PGVECTOR: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    class Config:
        env_file = '../.env'