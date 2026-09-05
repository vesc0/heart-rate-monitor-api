import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from core.logger import get_logger
logger = get_logger(__name__)


class RagPipelineUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    source: str
    text: str
    tags: list[str]


@dataclass(frozen=True)
class RetrievedContext:
    id: str
    title: str
    source: str
    text: str
    score: float


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RagPipelineUnavailableError(
                "sentence-transformers is not installed."
            ) from exc

        self.model_name = model_name
        self.signature = f"sentence-transformers:{model_name}"
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RagPipelineUnavailableError(
                f"Could not load sentence-transformer model '{model_name}'."
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=float).tolist()


class HashingEmbeddingProvider:
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.signature = f"hashing-vectorizer:{dimensions}:word-1-2"
        self._vectorizer = HashingVectorizer(
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._vectorizer.transform(texts)
        return vectors.astype(np.float32).toarray().tolist()


class LocalHrvRagPipeline:
    def __init__(
        self,
        knowledge_path: Path,
        persist_dir: Path,
        collection_name: str = "hrv_medical_knowledge",
    ):
        self.knowledge_path = knowledge_path
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client: Any = None
        self._embedding_provider: Any = None
        self._documents: list[KnowledgeDocument] = []

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedContext]:
        self._ensure_initialized()
        query_embedding = self._embedding_provider.embed([query])[0]
        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
        ).points
        
        contexts = []
        for res in results:
            contexts.append(
                RetrievedContext(
                    id=str(res.id),
                    title=str(res.payload.get("title", "")),
                    source=str(res.payload.get("source", "")),
                    text=str(res.payload.get("text", "")),
                    score=float(res.score),
                )
            )
        return contexts

    def _ensure_initialized(self) -> None:
        if self._client is not None:
            return

        try:
            from qdrant_client import QdrantClient
        except Exception as exc:
            raise RagPipelineUnavailableError(
                "qdrant-client is not installed. Install backend dependencies first."
            ) from exc

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._documents = self._load_documents()
        self._embedding_provider = self._create_embedding_provider()
        self._client = QdrantClient(path=str(self.persist_dir))
        self._sync_collection()

    def _load_documents(self) -> list[KnowledgeDocument]:
        try:
            raw_documents = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RagPipelineUnavailableError(
                f"Knowledge base not found: {self.knowledge_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RagPipelineUnavailableError(
                f"Knowledge base is not valid JSON: {self.knowledge_path}"
            ) from exc

        documents = []
        for item in raw_documents:
            documents.append(
                KnowledgeDocument(
                    id=str(item.get("id", str(uuid.uuid4()))),
                    title=str(item["title"]),
                    source=str(item.get("source", "Local HRV knowledge base")),
                    text=str(item["text"]),
                    tags=[str(tag) for tag in item.get("tags", [])],
                )
            )

        if not documents:
            raise RagPipelineUnavailableError("Knowledge base contains no documents.")

        return documents

    def _create_embedding_provider(self) -> Any:
        backend = os.getenv("RAG_EMBEDDING_BACKEND", "auto").strip().lower()
        model_name = os.getenv(
            "RAG_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        if backend in {"auto", "sentence-transformers", "sentence_transformers"}:
            try:
                return SentenceTransformerEmbeddingProvider(model_name)
            except RagPipelineUnavailableError as exc:
                if backend != "auto":
                    raise
                logger.warning(
                    "Falling back to hashing embeddings because %s", exc
                )

        if backend in {"auto", "hashing", "hashing-vectorizer"}:
            dimensions = self._read_positive_int("RAG_HASHING_DIMENSIONS", 384)
            return HashingEmbeddingProvider(dimensions=dimensions)

        raise RagPipelineUnavailableError(
            "RAG_EMBEDDING_BACKEND must be 'auto', 'sentence-transformers', or 'hashing'."
        )

    def _sync_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        needs_rebuild = True
        
        if exists:
            count = self._client.count(collection_name=self.collection_name).count
            if count == len(self._documents):
                needs_rebuild = False

        if needs_rebuild:
            if exists:
                self._client.delete_collection(collection_name=self.collection_name)
            
            dummy_text = "test"
            vector_size = len(self._embedding_provider.embed([dummy_text])[0])

            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            
            texts = [self._document_text(document) for document in self._documents]
            embeddings = self._embedding_provider.embed(texts)
            
            points = []
            for i, document in enumerate(self._documents):
                try:
                    # Qdrant supports UUIDs and positive integers
                    # If document.id is numeric, use it, else parse/create UUID
                    if document.id.isdigit():
                        point_id = int(document.id)
                    else:
                        point_id = str(uuid.UUID(document.id))
                except ValueError:
                    point_id = str(uuid.uuid4())
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embeddings[i],
                        payload={
                            "title": document.title,
                            "source": document.source,
                            "tags": ",".join(document.tags),
                            "text": document.text,
                        }
                    )
                )
            
            self._client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(
                "Indexed %s HRV knowledge documents in Qdrant collection '%s'",
                len(self._documents),
                self.collection_name,
            )

    @staticmethod
    def _document_text(document: KnowledgeDocument) -> str:
        tags = " ".join(document.tags)
        return f"{document.title}\nTags: {tags}\n{document.text}"

    @staticmethod
    def _read_positive_int(name: str, default: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            return default
        return value if value > 0 else default
