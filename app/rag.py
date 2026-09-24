from collections.abc import Sequence

from openai import OpenAI

from app.config import Settings
from app.booking import booking_reply
from app.ingestion import embed_chunks
from app.memory import ConversationMemory
from app.vector_store import VectorStore


def build_context(results: Sequence[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[Document chunk {index + 1}]\n{result['text']}" for index, result in enumerate(results)
    )


def retrieve_context(question: str, settings: Settings, vector_store: VectorStore) -> str:
    query_embedding = embed_chunks([question], settings)[0]
    results = vector_store.search(query_embedding, limit=5)
    return build_context(results)


def answer_question(
    question: str,
    history: Sequence[dict[str, str]],
    context: str,
    settings: Settings,
) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    prompt = (
        "Answer the user's question using the document context when relevant. "
        "If the answer is not in the context, say that you do not know.\n\n"
        f"Document context:\n{context or '[No relevant document context found]'}"
    )
    messages = [{"role": "system", "content": prompt}, *history, {"role": "user", "content": question}]
    response = OpenAI(api_key=settings.openai_api_key).chat.completions.create(
        model=settings.llm_model,
        messages=messages,
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The LLM returned an empty response")
    return answer


def chat(
    session_id: str,
    question: str,
    settings: Settings,
    memory: ConversationMemory,
    vector_store: VectorStore,
) -> str:
    history = memory.get_messages(session_id)
    booking_answer = booking_reply(session_id, question, history, settings, memory)
    if booking_answer is not None:
        return booking_answer
    context = retrieve_context(question, settings, vector_store)
    answer = answer_question(question, history, context, settings)
    memory.append(session_id, "user", question)
    memory.append(session_id, "assistant", answer)
    return answer