from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Environment-controlled model and retrieval settings.

    Retrieval configuration is kept here so the RAG component remains
    portable and does not depend on machine-specific paths or constants.
    """

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    vector_db_path: str
    embedding_model: str
    rag_collection: str
    rag_top_k: int
    rag_relevance_threshold: float
    api_host: str
    api_port: int
    streamlit_host: str
    streamlit_port: int


def load_settings() -> Settings:
    """Load and validate application configuration."""

    load_dotenv()

    rag_top_k = int(os.getenv("RAG_TOP_K", "3"))
    rag_relevance_threshold = float(
        os.getenv("RAG_RELEVANCE_THRESHOLD", "0.35")
    )

    if rag_top_k < 1:
        raise ValueError("RAG_TOP_K must be at least 1")

    if not 0.0 <= rag_relevance_threshold <= 1.0:
        raise ValueError(
            "RAG_RELEVANCE_THRESHOLD must be between 0 and 1"
        )

    return Settings(
        llm_base_url=os.getenv(
            "LLM_BASE_URL",
            "http://localhost:11434/v1",
        ),
        llm_api_key=os.getenv(
            "LLM_API_KEY",
            "not-required",
        ),
        llm_model=os.getenv(
            "LLM_MODEL",
            "qwen2.5:3b",
        ),
        vector_db_path=os.getenv(
            "VECTOR_DB_PATH",
            ".data/vector_db",
        ),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        rag_collection=os.getenv(
            "RAG_COLLECTION",
            "customer-support",
        ),
        rag_top_k=rag_top_k,
        rag_relevance_threshold=rag_relevance_threshold,
        api_host=os.getenv(
            "API_HOST",
            "127.0.0.1",
        ),
        api_port=int(
            os.getenv("API_PORT", "8000")
        ),
        streamlit_host=os.getenv(
            "STREAMLIT_HOST",
            "127.0.0.1",
        ),
        streamlit_port=int(
            os.getenv("STREAMLIT_PORT", "8501")
        ),
    )