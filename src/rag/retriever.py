from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError


class KnowledgeRetriever:
    """Persistent Chroma-backed knowledge retriever."""

    MIN_RELEVANCE_SCORE = 0.25

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    @staticmethod
    def _document_id(source: str, content: str) -> str:
        """Create a deterministic ID for a knowledge-base chunk."""
        raw = f"{source}\n{content}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _safe_source(source: object) -> str:
        """Return only a safe source filename."""
        value = str(source or "").strip()
        if not value:
            return "unknown"

        return Path(value).name

    async def initialize(self) -> None:
        """Load, split, and persist the support knowledge base."""
        documents = split_support_documents(
            load_support_documents(self.documents_dir)
        )

        if not documents:
            raise ValueError("No knowledge-base chunks were produced")

        embeddings = build_embeddings(self.settings)

        vector_db_path = Path(self.settings.vector_db_path)

        # Chroma creates the directory itself. We do not create unrelated
        # directories during initialization.
        store = Chroma(
            collection_name=self.settings.rag_collection,
            embedding_function=embeddings,
            persist_directory=str(vector_db_path),
        )

        ids = [
            self._document_id(
                str(document.metadata.get("source", "")),
                document.page_content,
            )
            for document in documents
        ]

        existing = store.get(ids=ids)
        existing_ids = set(existing.get("ids", []))

        new_documents = []
        new_ids = []

        for document, document_id in zip(documents, ids):
            if document_id not in existing_ids:
                new_documents.append(document)
                new_ids.append(document_id)

        if new_documents:
            store.add_documents(
                documents=new_documents,
                ids=new_ids,
            )

        self._store = store

    async def search(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """Retrieve relevant knowledge-base chunks."""
        if self._store is None:
            raise ComponentNotReadyError(
                "Knowledge retriever has not been initialized"
            )

        clean_query = query.strip()
        if not clean_query:
            return []

        requested_limit = limit or self.settings.rag_top_k

        if requested_limit <= 0:
            return []

        results = self._store.similarity_search_with_relevance_scores(
            clean_query,
            k=requested_limit,
        )

        normalized: list[dict[str, str]] = []

        for document, score in results:
            if score < self.MIN_RELEVANCE_SCORE:
                continue

            content = document.page_content.strip()
            source = self._safe_source(
                document.metadata.get("source")
            )

            if not content:
                continue

            normalized.append(
                {
                    "content": content,
                    "source": source,
                }
            )

            if len(normalized) >= requested_limit:
                break

        return normalized