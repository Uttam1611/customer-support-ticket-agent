# Customer Support Ticket Agent

A full-stack customer support ticket agent that combines persistent RAG-based knowledge retrieval, session-aware conversational ticket creation, FastAPI REST APIs, and a Streamlit chat interface. The system also supports microphone-based speech-to-text (STT) and optional text-to-speech (TTS) playback while keeping the existing text chat workflow unchanged.

## Features

- **Persistent RAG knowledge retrieval**
  - Chroma vector database
  - Stable document IDs
  - Configurable top-k retrieval
  - Relevance threshold for unknown/irrelevant questions
  - Source attribution in responses

- **Conversational support agent**
  - LangGraph-based workflow
  - Knowledge-base retrieval before answering
  - Routing between answering a question and creating a support ticket
  - Multi-turn session state

- **Support ticket workflow**
  - Customer name
  - Customer email
  - Issue description
  - Category
  - Ticket creation after required fields are collected
  - Duplicate ticket protection
  - Ticket retrieval through the API

- **REST API**
  - Health check
  - Text chat
  - Ticket retrieval
  - Speech-to-text
  - Text-to-speech

- **Streamlit interface**
  - Session-aware chat
  - Chat history
  - Retrieved sources
  - Ticket ID display
  - Microphone input
  - Editable STT transcript
  - Speaker button for agent responses

- **Testing**
  - RAG retrieval and relevance policy
  - Unknown/irrelevant queries
  - Multi-turn ticket workflow
  - Ticket retrieval
  - Missing-field validation
  - Duplicate protection
  - Voice service behavior
  - Graceful failure handling

## Architecture

```text
                    ┌─────────────────────┐
                    │  Streamlit Client   │
                    │                     │
                    │ Text / Microphone   │
                    │ Chat / Audio        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    │                     │
                    │ /chat               │
                    │ /tickets/{id}       │
                    │ /voice/transcribe   │
                    │ /voice/synthesize   │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Support Pipeline│        │  Voice Pipeline │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 ▼                          ▼
        ┌─────────────────┐        ┌─────────────────┐
        │   LangGraph     │        │ STT / TTS       │
        │ Support Workflow│        │ Adapters        │
        └────────┬────────┘        └─────────────────┘
                 │
       ┌─────────┼──────────┐
       ▼         ▼          ▼
    Retrieve   Decide    Ticket Flow
       │         │          │
       ▼         ▼          ▼
   Chroma DB    LLM     Ticket Storage
   / RAG                 / Tool
```

## Technology Stack

- Python 3.11+
- FastAPI
- Uvicorn
- Streamlit
- LangChain
- LangGraph
- Chroma
- Pydantic
- SQLAlchemy / PostgreSQL components where configured by the project
- Local or OpenAI-compatible LLM endpoint
- Google Speech Recognition for STT
- Edge TTS for TTS
- Pytest

## Project Structure

```text
customer_support_ticket_agent/
│
├── src/
│   ├── api/
│   │   └── server.py
│   ├── llm/
│   │   ├── client.py
│   │   └── workflow.py
│   ├── rag/
│   │   └── retriever.py
│   ├── sessions/
│   │   └── store.py
│   ├── voice/
│   │   ├── contracts.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── stt.py
│   │   └── tts.py
│   ├── pipeline.py
│   ├── config.py
│   └── ...
│
├── knowledge_base/
│   └── *.md
│
├── tests/
│   └── ...
│
├── streamlit_app.py
├── requirements.txt
├── .env.example
└── README.md
```

## Prerequisites

- Python 3.11 or newer
- `pip`
- A configured LLM endpoint compatible with the project's LangChain/OpenAI-compatible client
- Internet access for Google speech recognition and Edge TTS when using those services
- Windows, Linux, or macOS

## Installation

Clone or extract the project and enter the project directory:

```bash
cd customer_support_ticket_agent
```

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
copy .env.example .env
```

For Linux/macOS:

```bash
cp .env.example .env
```

Configure the required LLM settings in `.env`.

## Running the Application

### Start the FastAPI backend

The backend can be started with:

```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8001
```

The backend health endpoint is:

```text
http://127.0.0.1:8001/health
```

### Start the Streamlit frontend

Open a second terminal with the virtual environment activated.

Windows PowerShell:

```powershell
$env:API_BASE_URL="http://127.0.0.1:8001"
streamlit run streamlit_app.py
```

The Streamlit application will normally be available at:

```text
http://localhost:8501
```

## API Endpoints

### Health

```http
GET /health
```

Example:

```bash
curl http://127.0.0.1:8001/health
```

Expected response when the application is ready:

```json
{
  "status": "ready"
}
```

### Chat

```http
POST /chat
```

Example request:

```json
{
  "session_id": "demo-session",
  "message": "How long does standard shipping take?"
}
```

Example PowerShell request:

```powershell
$body = @{
    session_id = "demo-session"
    message = "How long does standard shipping take?"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8001/chat" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The response contains the agent response and may include retrieved source information and a ticket ID when a ticket is created.

### Retrieve a Ticket

```http
GET /tickets/{ticket_id}
```

Example:

```bash
curl http://127.0.0.1:8001/tickets/TICKET_ID
```

Replace `TICKET_ID` with the ID returned after ticket creation.

### Speech-to-Text

```http
POST /voice/transcribe
```

The endpoint accepts an uploaded WAV audio file and returns a transcription response.

Example response structure:

```json
{
  "success": true,
  "transcript": "I have a problem with my payment",
  "processing_time_ms": 1234
}
```

The transcript is displayed in the Streamlit interface before it is submitted to the existing text agent, allowing the user to correct it.

### Text-to-Speech

```http
POST /voice/synthesize
```

The endpoint accepts the displayed agent response text and returns playable audio.

The Streamlit interface generates audio only when the user selects the speaker control. Audio is associated with the corresponding agent message.

## RAG Workflow

The RAG component loads the knowledge base into a persistent Chroma collection.

The retrieval process:

```text
User Query
    ↓
Retriever
    ↓
Chroma Similarity Search
    ↓
Relevance Filtering
    ↓
Top-k Relevant Documents
    ↓
LangGraph Agent
    ↓
Grounded Answer + Sources
```

The retriever uses deterministic document identifiers so that repeated initialization does not create duplicate documents.

Queries with insufficient relevant evidence are handled using the configured relevance threshold rather than blindly generating an answer from unrelated knowledge-base content.

## Agent Workflow

The conversational workflow is implemented with LangGraph.

```text
User Message
     ↓
Retrieve
     ↓
Decide
     ├───────────────┐
     │               │
     ▼               ▼
   Answer          Ticket
     │               │
     │        Collect Missing Fields
     │               │
     │               ▼
     │        Validate Ticket Data
     │               │
     │               ▼
     │          Create Ticket
     │
     └───────► Response
```

The ticket workflow collects:

1. Customer name
2. Customer email
3. Issue description
4. Category

Supported categories include:

```text
order
payment
account
technical
other
```

The system maintains state using a client-supplied `session_id`, allowing information to be collected over multiple messages.

## Voice Workflow

Voice functionality is integrated into the existing application rather than creating a separate voice agent.

### Speech-to-Text

```text
Microphone
    ↓
Audio Recording
    ↓
/voice/transcribe
    ↓
Editable Transcript
    ↓
User Confirmation
    ↓
Existing /chat Endpoint
    ↓
Existing Agent Workflow
```

The transcript must be explicitly submitted after editing. Voice input therefore follows the same RAG, session, routing, and ticket workflow as typed input.

### Text-to-Speech

```text
Completed Agent Response
          ↓
     Speaker Button
          ↓
   /voice/synthesize
          ↓
      Audio Bytes
          ↓
       Playback
```

TTS is not autoplayed. A synthesis failure does not remove or invalidate the original text response.

## Testing

Run the complete test suite with:

```bash
pytest -q
```

The test suite covers the core behavior required by the implementation, including:

- RAG retrieval
- Relevant versus irrelevant queries
- Stable retrieval IDs
- Session state
- Multi-turn ticket creation
- Missing ticket fields
- Ticket validation
- Duplicate ticket protection
- Ticket retrieval
- Voice STT success and failure cases
- Voice TTS success and failure cases
- API/pipeline error handling

## Example User Flow

### Knowledge-base question

```text
User:
How long does standard shipping take?

Agent:
Standard delivery usually takes three to five business days.

Sources:
shipping.md
```

### Ticket creation

```text
User:
I need to report a technical issue.

Agent:
I can create a ticket. What is your name?

User:
Uttam Singh

Agent:
What email address should I associate with the ticket?

User:
example@email.com

Agent:
Please describe the issue.

User:
The application is failing when I submit a request.

Agent:
What category does this issue belong to?

User:
technical

Agent:
Ticket created successfully.
Ticket ID: ...
```

## Configuration

Configuration is loaded through environment variables.

Important settings include:

```text
LLM_BASE_URL
LLM_API_KEY
LLM_MODEL
VECTOR_DB_PATH
EMBEDDING_MODEL
RAG_COLLECTION
RAG_TOP_K
RAG_RELEVANCE_THRESHOLD
API_HOST
API_PORT
STREAMLIT_HOST
STREAMLIT_PORT
```

Do not commit real API keys or secrets to the repository.

Use `.env.example` to document required configuration without exposing credentials.

## Error Handling

The API distinguishes between service readiness and processing failures.

Examples include:

- Backend not ready
- Invalid or blank session IDs
- Invalid or blank messages
- Empty voice recordings
- Unsupported audio formats
- Speech recognition failures
- TTS failures
- Invalid model routing decisions
- Insufficient RAG evidence

Voice failures are recoverable and do not replace the existing text-agent response with an error state.

## Git Workflow

The project was developed incrementally using Git commits for major implementation stages.

Example:

```bash
git log --oneline
```

The final repository should contain the completed source code, tests, configuration examples, knowledge base, and documentation.

## Submission Checklist

Before submission:

- [ ] Application starts successfully
- [ ] `/health` returns a ready status
- [ ] Text chat works
- [ ] RAG answers include relevant sources
- [ ] Ticket creation works across multiple turns
- [ ] Ticket ID is returned after creation
- [ ] Ticket retrieval works
- [ ] Microphone transcription works
- [ ] Transcript can be edited before submission
- [ ] TTS speaker control works
- [ ] TTS does not autoplay
- [ ] Voice failures do not break text chat
- [ ] `pytest -q` passes
- [ ] README is included
- [ ] `.env.example` is included
- [ ] No secrets are included
- [ ] Final project ZIP is created
- [ ] Demonstration video is included
- [ ] Google Drive sharing is set to anyone with the link as Viewer

## Demonstration

The demonstration should show the complete working flow:

1. Start the application.
2. Ask a knowledge-base question and show source attribution.
3. Create a support ticket through multiple turns.
4. Show the generated ticket ID.
5. Demonstrate microphone transcription.
6. Edit and submit the transcript.
7. Demonstrate speaker-button TTS playback.
8. Run the test suite and show the result.

