# Document RAG API

Small FastAPI backend for document ingestion, custom RAG chat, and interview booking.

## Requirements

- Python 3.12+
- Redis running at `REDIS_URL`
- Qdrant running at `QDRANT_URL`
- OpenAI API key for embeddings, chat, and structured booking extraction

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Keep secrets out of `.env.example` and source control.

Start the API:

```powershell
python -m uvicorn app.main:app --reload
```

## APIs

Upload a PDF or TXT document:

```powershell
curl -X POST "http://127.0.0.1:8000/documents/upload?chunking_strategy=recursive" -F "file=@sample.txt"
```

Chat with a document:

```powershell
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"session_id":"abc123","message":"What does the document say about X?"}'
```

Use the same `session_id` for multi-turn memory. Booking requests are detected by the LLM; missing name, email, date, or time are requested one at a time, and complete bookings are stored in SQLite.

## Storage

- SQLite stores document metadata, chunks, and confirmed bookings.
- Qdrant stores chunk embeddings and text payloads.
- Redis stores chat history and in-progress booking fields.
