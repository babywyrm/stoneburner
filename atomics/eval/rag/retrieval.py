"""RAG retrieval layer: document loading, chunking, and embedder protocol."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from atomics.eval.rag import RAGChunk

logger = logging.getLogger("atomics.eval.rag.retrieval")


@dataclass
class Document:
    content: str
    source: str
    metadata: dict[str, str] | None = None


@dataclass
class Chunk:
    content: str
    source: str
    chunk_index: int
    offset: int
    token_estimate: int


class Embedder(Protocol):
    """Protocol for text embedders."""

    dim: int
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")
    if not text:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


class MockEmbedder:
    """Deterministic embedder for tests."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.model_name = "mock"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self.dim for _ in texts]


class LocalSentenceTransformerEmbedder:
    """Local embedder using sentence-transformers."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "RAG indexing requires the [rag] extra: "
                'uv pip install "atomics[rag]"'
            ) from exc
        self.model_name = model
        self._model = SentenceTransformer(model)
        # Prefer the renamed API; fall back for older sentence-transformers.
        dim_fn = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self.dim = dim_fn()

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]


def load_documents(path: Path | str) -> list[Document]:
    """Load .txt, .md, .json, and .html files from a path."""
    p = Path(path)
    if p.is_file():
        return [_load_file(p)]
    if not p.is_dir():
        raise ValueError(f"Path is not a file or directory: {path}")

    docs: list[Document] = []
    for ext in ("*.txt", "*.md", "*.json", "*.html"):
        for file_path in p.rglob(ext):
            try:
                docs.append(_load_file(file_path))
            except (UnicodeDecodeError, OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping %s: %s", file_path, exc)
    return sorted(docs, key=lambda d: d.source)


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _load_file(path: Path) -> Document:
    content = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(content)
        if isinstance(data, dict) and "text" in data:
            content = _coerce_text(data["text"])
        elif isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, indent=2)
    return Document(
        content=content,
        source=path.name,
        metadata={"full_path": str(path.resolve())},
    )


def _require_sqlite_vec() -> Any:
    try:
        import sqlite_vec
    except ImportError as exc:
        raise ImportError(
            "RAG indexing requires the [rag] extra: "
            'uv pip install "atomics[rag]"'
        ) from exc
    return sqlite_vec


class RAGIndex:
    """sqlite-vec backed index for RAG retrieval."""

    def __init__(self, db_path: Path | str, embedder: Embedder | None = None) -> None:
        self.db_path = Path(db_path)
        self.embedder = embedder or MockEmbedder(dim=8)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        vec = _require_sqlite_vec()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.enable_load_extension(True)
            vec.load(conn)
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0("
                f"embedding FLOAT[{self.embedder.dim}], "
                f"+source TEXT, +chunk_index INTEGER, +offset INTEGER, +content TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT)"
            )

    def build(
        self,
        documents: list[Document],
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> int:
        """Chunk and embed documents, then store them. Returns chunk count."""
        vec = _require_sqlite_vec()
        chunks: list[Chunk] = []
        for doc in documents:
            chunk_texts = chunk_text(doc.content, chunk_size=chunk_size, overlap=overlap)
            for i, text in enumerate(chunk_texts):
                chunks.append(
                    Chunk(
                        content=text,
                        source=doc.source,
                        chunk_index=i,
                        offset=i * (chunk_size - overlap),
                        token_estimate=len(text.split()),
                    )
                )
        source_paths = {
            doc.source: (doc.metadata or {}).get("full_path", doc.source)
            for doc in documents
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.enable_load_extension(True)
            vec.load(conn)
            conn.execute("DELETE FROM chunks")
            if chunks:
                embeddings = self.embedder.embed([c.content for c in chunks])
                conn.executemany(
                    "INSERT INTO chunks (source, chunk_index, offset, content, embedding) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            c.source,
                            c.chunk_index,
                            c.offset,
                            c.content,
                            vec.serialize_float32(emb),
                        )
                        for c, emb in zip(chunks, embeddings, strict=True)
                    ],
                )
            self._set_meta(conn, "chunk_count", str(len(chunks)))
            self._set_meta(conn, "chunk_size", str(chunk_size))
            self._set_meta(conn, "overlap", str(overlap))
            self._set_meta(conn, "embedding_model", self.embedder.model_name)
            self._set_meta(conn, "source_paths", json.dumps(source_paths))
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[RAGChunk]:
        """Retrieve top-k chunks for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vec = _require_sqlite_vec()
        query_embedding = self.embedder.embed([query])[0]
        with sqlite3.connect(self.db_path) as conn:
            conn.enable_load_extension(True)
            vec.load(conn)
            rows = conn.execute(
                "SELECT source, chunk_index, offset, content, distance "
                "FROM chunks WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (vec.serialize_float32(query_embedding), top_k),
            ).fetchall()
        return [
            RAGChunk(content=row[3], label="retrieved", source=row[0])
            for row in rows
        ]

    def info(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT key, value FROM index_meta").fetchall()
        return {key: value for key, value in rows}

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
