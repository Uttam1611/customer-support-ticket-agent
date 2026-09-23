from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.rag.document_loader import (
    load_support_documents,
    split_support_documents,
)
from src.rag.retriever import KnowledgeRetriever
from src.utils.errors import ComponentNotReadyError


class FakeEmbeddings(Embeddings):
    """Small deterministic embeddings implementation for fast unit tests."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.lower()

        if any(
            word in normalized
            for word in (
                "shipping",
                "delivery",
                "tracking",
                "business day",
            )
        ):
            return [1.0, 0.0, 0.0, 0.0]

        if any(
            word in normalized
            for word in (
                "payment",
                "charge",
                "charged",
                "card",
            )
        ):
            return [0.0, 1.0, 0.0, 0.0]

        if any(
            word in normalized
            for word in (
                "return",
                "refund",
                "replacement",
            )
        ):
            return [0.0, 0.0, 1.0, 0.0]

        if any(
            word in normalized
            for word in (
                "account",
                "password",
                "email",
                "login",
            )
        ):
            return [0.0, 0.0, 0.0, 1.0]

        return [0.0, 0.0, 0.0, 0.0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self._vector(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._vector(text)


def make_settings(
    vector_db_path: str,
    *,
    top_k: int = 3,
    threshold: float = 0.35,
) -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=vector_db_path,
        embedding_model="test-embeddings",
        rag_collection="test-support",
        rag_top_k=top_k,
        rag_relevance_threshold=threshold,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


def make_retriever(
    knowledge_dir: Path,
    vector_db_path: Path,
    monkeypatch,
    *,
    top_k: int = 3,
    threshold: float = 0.35,
) -> KnowledgeRetriever:
    settings = make_settings(
        str(vector_db_path),
        top_k=top_k,
        threshold=threshold,
    )

    monkeypatch.setattr(
        "src.rag.retriever.build_embeddings",
        lambda _: FakeEmbeddings(),
    )

    return KnowledgeRetriever(
        settings,
        knowledge_dir,
    )


def test_loader_preserves_source_names(
    knowledge_dir: Path,
) -> None:
    documents = load_support_documents(
        knowledge_dir
    )

    assert {
        item.metadata["source"]
        for item in documents
    } == {
        "accounts.md",
        "payments.md",
        "returns.md",
        "shipping.md",
    }


def test_splitter_keeps_source_metadata(
    knowledge_dir: Path,
) -> None:
    chunks = split_support_documents(
        load_support_documents(knowledge_dir)
    )

    assert chunks

    assert all(
        chunk.metadata.get(
            "source",
            "",
        ).endswith(".md")
        for chunk in chunks
    )


def test_stable_ids_are_deterministic(
    knowledge_dir: Path,
) -> None:
    settings = make_settings(
        ".data/test-vector-db"
    )

    retriever = KnowledgeRetriever(
        settings,
        knowledge_dir,
    )

    first = retriever._stable_id(
        "shipping.md",
        "standard delivery",
    )

    second = retriever._stable_id(
        "shipping.md",
        "standard delivery",
    )

    different = retriever._stable_id(
        "shipping.md",
        "express delivery",
    )

    assert first == second
    assert first != different


@pytest.mark.asyncio
async def test_search_before_initialization_is_controlled(
    knowledge_dir: Path,
    tmp_path: Path,
) -> None:
    settings = make_settings(
        str(tmp_path / "vectors")
    )

    retriever = KnowledgeRetriever(
        settings,
        knowledge_dir,
    )

    with pytest.raises(ComponentNotReadyError):
        await retriever.search("shipping")


@pytest.mark.asyncio
async def test_blank_query_returns_no_results(
    knowledge_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    retriever = make_retriever(
        knowledge_dir,
        tmp_path / "vectors",
        monkeypatch,
    )

    assert await retriever.search("") == []
    assert await retriever.search("   ") == []


@pytest.mark.asyncio
async def test_initialize_creates_persistent_store(
    knowledge_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    vector_path = tmp_path / "vectors"

    retriever = make_retriever(
        knowledge_dir,
        vector_path,
        monkeypatch,
    )

    await retriever.initialize()

    assert vector_path.exists()
    assert retriever._store is not None


@pytest.mark.asyncio
async def test_initialization_is_idempotent(
    knowledge_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    vector_path = tmp_path / "vectors"

    first = make_retriever(
        knowledge_dir,
        vector_path,
        monkeypatch,
    )

    await first.initialize()

    assert first._store is not None

    first_count = first._store._collection.count()

    second = make_retriever(
        knowledge_dir,
        vector_path,
        monkeypatch,
    )

    await second.initialize()

    assert second._store is not None

    second_count = second._store._collection.count()

    assert first_count == second_count


@pytest.mark.asyncio
async def test_relevant_query_returns_source_and_content(
    knowledge_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    retriever = make_retriever(
        knowledge_dir,
        tmp_path / "vectors",
        monkeypatch,
    )

    await retriever.initialize()

    results = await retriever.search(
        "How long does standard shipping take?"
    )

    assert results
    assert results[0]["source"] == "shipping.md"
    assert "business days" in results[0]["content"]


@pytest.mark.asyncio
async def test_irrelevant_query_returns_no_results(
    knowledge_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    retriever = make_retriever(
        knowledge_dir,
        tmp_path / "vectors",
        monkeypatch,
        threshold=0.35,
    )

    await retriever.initialize()

    results = await retriever.search(
        "quantum spacecraft maintenance procedures"
    )

    assert results == []


@pytest.mark.asyncio
async def test_explicit_limit_is_respected(
    knowledge_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    retriever = make_retriever(
        knowledge_dir,
        tmp_path / "vectors",
        monkeypatch,
        top_k=3,
    )

    await retriever.initialize()

    results = await retriever.search(
        "payment charge",
        limit=1,
    )

    assert len(results) <= 1


@pytest.mark.asyncio
async def test_sources_are_safe_filenames(
    knowledge_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    retriever = make_retriever(
        knowledge_dir,
        tmp_path / "vectors",
        monkeypatch,
    )

    await retriever.initialize()

    results = await retriever.search(
        "payment charge"
    )

    assert results

    for result in results:
        assert "/" not in result["source"]
        assert "\\" not in result["source"]
        assert result["source"].endswith(".md")