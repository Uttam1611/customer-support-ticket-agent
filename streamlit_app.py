from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

st.set_page_config(
    page_title="Customer Support",
    page_icon="🎧",
)

st.title("Customer Support")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"demo-{uuid.uuid4().hex[:8]}"

if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}

if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                st.caption(
                    "Sources: "
                    + ", ".join(sorted(set(str(item) for item in sources)))
                )

            ticket_id = message.get("ticket_id")
            if ticket_id:
                st.success(f"Ticket created: {ticket_id}")

            if "message_id" in message and message["message_id"]:
                if st.button(
                    "🔊",
                    key=f"speaker_{message['message_id']}",
                    help="Play this response",
                ):
                    play_response(message)


def play_response(message: dict) -> None:
    message_id = message.get("message_id")
    if not message_id:
        return

    if message_id in st.session_state.audio_cache:
        st.audio(st.session_state.audio_cache[message_id], format="audio/mpeg")
        return

    try:
        with st.spinner("Generating audio..."):
            response = httpx.post(
                f"{API_BASE_URL}/voice/synthesize",
                json={
                    "message_id": str(message_id),
                    "text": message.get("content", ""),
                },
                timeout=httpx.Timeout(60.0, connect=5.0),
            )

        if response.status_code >= 400:
            st.error("Could not generate speech for this response.")
            return

        audio = response.content
        if not audio:
            st.error("The speech service returned empty audio.")
            return

        st.session_state.audio_cache[message_id] = audio
        st.audio(audio, format="audio/mpeg")
    except httpx.RequestError:
        st.error("The speech service is unavailable right now.")


def send_message(prompt: str) -> None:
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        return

    user_message = {
        "role": "user",
        "content": cleaned_prompt,
    }
    st.session_state.messages.append(user_message)

    try:
        with st.spinner("Contacting support backend..."):
            response = httpx.post(
                f"{API_BASE_URL}/chat",
                json={
                    "session_id": str(st.session_state.session_id).strip() or "demo-session",
                    "message": cleaned_prompt,
                },
                timeout=httpx.Timeout(60.0, connect=5.0),
            )

        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", "The backend rejected the request.")
            except ValueError:
                detail = "The backend returned an invalid response."
            st.error(f"Backend error: {detail}")
            return

        payload = response.json()

        assistant_message = {
            "role": "assistant",
            "content": payload.get("response", ""),
            "sources": payload.get("sources", []),
            "ticket_id": payload.get("ticket_id"),
            "message_id": str(uuid.uuid4()),
        }

        if not assistant_message["content"]:
            st.error("The backend returned an empty response.")
            return

        st.session_state.messages.append(assistant_message)

    except httpx.RequestError:
        st.error("The support backend is unavailable. Please check that FastAPI is running.")
    except (TypeError, ValueError, KeyError):
        st.error("The backend returned an invalid response.")


st.sidebar.title("Session")
current_session = st.sidebar.text_input(
    "Session ID",
    value=st.session_state.session_id,
    help="Use the same ID to continue a conversation across multiple turns.",
)
if current_session.strip():
    st.session_state.session_id = current_session.strip()
else:
    st.session_state.session_id = f"demo-{uuid.uuid4().hex[:8]}"
    st.sidebar.warning("Session ID was blank, so a new one was generated.")

st.sidebar.caption(f"Active session: {st.session_state.session_id}")

for message in st.session_state.messages:
    render_message(message)

st.divider()

st.subheader("Voice input")

recording = st.audio_input("Record your message")
if recording is not None:
    if st.button("Transcribe recording", key="transcribe_voice"):
        try:
            with st.spinner("Transcribing audio..."):
                response = httpx.post(
                    f"{API_BASE_URL}/voice/transcribe",
                    files={
                        "audio": (
                            "recording.wav",
                            recording.getvalue(),
                            recording.type or "audio/wav",
                        )
                    },
                    timeout=httpx.Timeout(60.0, connect=5.0),
                )

            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", "Transcription failed.")
                except ValueError:
                    detail = "Transcription failed."
                st.error(detail)
            else:
                st.session_state.voice_transcript = response.json().get("transcript", "")
        except httpx.RequestError:
            st.error("The transcription service is unavailable.")

st.session_state.voice_transcript = st.text_area(
    "Editable transcript",
    value=st.session_state.voice_transcript,
    placeholder="Your transcription will appear here...",
)

if st.button("Submit transcript", key="submit_transcript"):
    transcript = st.session_state.voice_transcript.strip()
    if not transcript:
        st.warning("Please provide a transcript first.")
    else:
        send_message(transcript)
        st.session_state.voice_transcript = ""
        st.rerun()

st.divider()

prompt = st.chat_input("Type your support question...")
if prompt:
    send_message(prompt)
    st.rerun()