import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

from app.config import Settings
from app.database import SessionLocal
from app.memory import ConversationMemory
from app.models import Booking


class BookingFields(BaseModel):
    booking_intent: bool = False
    name: str | None = Field(default=None)
    email: str | None = Field(default=None)
    date: str | None = Field(default=None)
    time: str | None = Field(default=None)


class CompleteBooking(BaseModel):
    name: str
    email: str
    date: str
    time: str

    @model_validator(mode="after")
    def validate_email(self) -> "CompleteBooking":
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", self.email):
            raise ValueError("email must be valid")
        return self


def extract_booking_fields(
    message: str, history: list[dict[str, str]], settings: Settings
) -> BookingFields:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    prompt = (
        "Identify whether the user wants to book an interview and extract only details explicitly provided. "
        "Never invent values. Return booking_intent=true for booking requests, even when fields are missing."
    )
    response = OpenAI(api_key=settings.openai_api_key).beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": prompt},
            *history,
            {"role": "user", "content": message},
        ],
        response_format=BookingFields,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("The LLM returned invalid booking data")
    return parsed


def merge_booking_fields(
    previous: dict[str, str], extracted: BookingFields
) -> dict[str, str]:
    values = dict(previous)
    for field in ("name", "email", "date", "time"):
        value = getattr(extracted, field)
        if value:
            values[field] = value.strip()
    return values


def missing_booking_field(fields: dict[str, str]) -> str | None:
    for field in ("name", "email", "date", "time"):
        if not fields.get(field):
            return field
    return None


def save_booking(fields: dict[str, str]) -> CompleteBooking:
    booking = CompleteBooking(**fields)
    with SessionLocal() as db:
        db.add(Booking(**booking.model_dump()))
        db.commit()
    return booking


def booking_reply(
    session_id: str,
    message: str,
    history: list[dict[str, str]],
    settings: Settings,
    memory: ConversationMemory,
) -> str | None:
    extracted = extract_booking_fields(message, history, settings)
    if not extracted.booking_intent:
        return None
    fields = merge_booking_fields(memory.get_booking(session_id), extracted)
    missing = missing_booking_field(fields)
    if missing:
        memory.set_booking(session_id, fields)
        question = {
            "name": "What is your name?",
            "email": "What email should I use?",
            "date": "What date would you prefer?",
            "time": "What time would you prefer?",
        }[missing]
        memory.append(session_id, "user", message)
        memory.append(session_id, "assistant", question)
        return question
    try:
        booking = save_booking(fields)
    except ValueError:
        question = "Please provide a valid email address."
        memory.set_booking(session_id, fields)
        memory.append(session_id, "user", message)
        memory.append(session_id, "assistant", question)
        return question
    memory.clear_booking(session_id)
    confirmation = (
        f"Your interview is booked for {booking.date} at {booking.time}. "
        f"Confirmation will be sent to {booking.email}."
    )
    memory.append(session_id, "user", message)
    memory.append(session_id, "assistant", confirmation)
    return confirmation