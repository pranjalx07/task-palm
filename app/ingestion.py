from io import BytesIO
from collections.abc import Sequence

from openai import OpenAI
from pypdf import PdfReader

from app.config import Settings
from app.database import SessionLocal
from app.models import Document, DocumentChunk
from app.vector_store import VectorStore


ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def extract_text(filename: str, content: bytes) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == "txt":
        return content.decode("utf-8", errors="replace").strip()
    if suffix == "pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    raise ValueError("Only .pdf and .txt files are supported")


def fixed_chunks(text: str, size: int = 1000, overlap: int = 100) -> list[str]:
    if overlap >= size:
        raise ValueError("Chunk overlap must be smaller than chunk size")
    return [text[start : start + size] for start in range(0, len(text), size - overlap)]


def recursive_chunks(text: str, size: int = 1000) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks or [text]


def make_chunks(text: str, strategy: str) -> list[str]:
    if strategy == "fixed":
        return fixed_chunks(text)
    if strategy == "recursive":
        return recursive_chunks(text)
    raise ValueError("chunking_strategy must be 'fixed' or 'recursive'")


def embed_chunks(chunks: Sequence[str], settings: Settings) -> list[list[float]]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(model=settings.embedding_model, input=list(chunks))
    return [item.embedding for item in response.data]


def ingest_document(
    filename: str, content: bytes, strategy: str, settings: Settings, vector_store: VectorStore
) -> tuple[int, int]:
    text = extract_text(filename, content)
    if not text:
        raise ValueError("The document contains no extractable text")
    chunks = make_chunks(text, strategy)
    embeddings = embed_chunks(chunks, settings)
    with SessionLocal() as db:
        document = Document(filename=filename, chunking_strategy=strategy)
        db.add(document)
        db.flush()
        db.add_all(
            [DocumentChunk(document_id=document.id, chunk_index=index, text=chunk) for index, chunk in enumerate(chunks)]
        )
        db.commit()
        document_id = document.id
    vector_store.upsert_chunks(document_id, chunks, embeddings)
    return document_id, len(chunks)