from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import get_settings
from app.database import Base, engine
from app.ingestion import ingest_document
from app.memory import build_memory
from app.rag import chat
from app.schemas import ChatRequest, ChatResponse, UploadResponse
from app.vector_store import build_vector_store


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=get_settings().app_name, lifespan=lifespan)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    if not request.session_id.strip() or not request.message.strip():
        raise HTTPException(status_code=400, detail="session_id and message are required")
    settings = get_settings()
    try:
        answer = chat(
            request.session_id,
            request.message,
            settings,
            build_memory(settings),
            build_vector_store(settings),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Chat service failed") from exc
    return ChatResponse(session_id=request.session_id, answer=answer)


@app.post("/documents/upload", response_model=UploadResponse, tags=["documents"])
async def upload_document(
    file: UploadFile = File(...), chunking_strategy: str = "fixed"
) -> UploadResponse:
    filename = file.filename or ""
    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")
    if chunking_strategy not in {"fixed", "recursive"}:
        raise HTTPException(status_code=400, detail="chunking_strategy must be 'fixed' or 'recursive'")
    try:
        content = await file.read()
        document_id, chunk_count = ingest_document(
            filename,
            content,
            chunking_strategy,
            get_settings(),
            build_vector_store(get_settings()),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Document processing failed") from exc
    return UploadResponse(
        document_id=document_id,
        filename=filename,
        chunking_strategy=chunking_strategy,
        chunk_count=chunk_count,
    )