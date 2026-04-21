# GovAssist

GovAssist is a multilingual government-service assistant for India with:

- RAG-based retrieval from official service documents
- structured JSON answers for frontend rendering
- speech-to-text and text-to-speech support
- multilingual query handling
- safety filtering
- follow-up memory

It currently works best for:

- passport
- aadhaar
- pf / epfo
- state portal expansion through `data/source_catalog.json`

## Core Features

- Service-aware retrieval
- Topic-aware retrieval for `fees`, `documents`, `processing_time`, `eligibility`, and `steps`
- State-aware filtering
- Translation to English for retrieval, then translation back to the user language
- Speech input through microphone or uploaded audio
- Source chunk visibility for debugging
- Fallback extraction when the LLM response is weak or unavailable

## Response Schema

Every `/chat` response is normalized to this exact structure:

```json
{
  "query": "",
  "intent": "",
  "language": "",
  "answer": {
    "title": "",
    "steps": [],
    "documents": [],
    "fees": "",
    "processing_time": "",
    "eligibility": [],
    "official_links": []
  },
  "follow_up_questions": [],
  "safety": {
    "is_safe": true,
    "message": ""
  },
  "source_chunks": []
}
```

## Project Structure

```text
backend/
  main.py
  llm/
  memory/
  multilingual/
  rag/
  safety/
  voice/
frontend/
  src/
data/
  processed_chunks/
  source_catalog.json
scripts/
  setup_chunks.py
  gov_data_pipeline.py
requirements.txt
```

## Requirements

### Python packages

Install with:

```powershell
python -m venv .venv
.venv\Scripts\activate
.venv\Scripts\python -m pip install -r requirements.txt
```

Important packages include:

- `fastapi`
- `uvicorn`
- `sentence-transformers`
- `transformers`
- `torch`
- `langdetect`
- `requests`
- `accelerate`
- `openai-whisper`
- `soundfile`
- `pyttsx3`
- `scipy`

### FFmpeg

Speech-to-text needs `ffmpeg`.

If `ffmpeg.exe` is already installed, either:

1. Add its `bin` folder to your system `PATH`
2. Or set `FFMPEG_PATH` before starting the backend

Example:

```powershell
$env:FFMPEG_PATH="C:\Users\harir\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
```

You can also point it to the `bin` folder instead of the file:

```powershell
$env:FFMPEG_PATH="C:\Users\harir\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"
```

To verify:

```powershell
ffmpeg -version
```

## LLM API Setup

The backend is API-first for generation and translation workflows.

Set these environment variables before starting the backend.

### Example: Groq

```powershell
$env:OPENAI_API_KEY="your_groq_key"
$env:OPENAI_BASE_URL="https://api.groq.com/openai/v1"
$env:OPENAI_MODEL="llama-3.1-8b-instant"
```

### Example: OpenAI-compatible provider

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Optional local model fallback:

```powershell
$env:ENABLE_LOCAL_LLM_FALLBACK="true"
```

## How To Run The Project

### 1. Activate the virtual environment

From the project root:

```powershell
.venv\Scripts\activate
```

If the venv does not exist yet:

```powershell
python -m venv .venv
.venv\Scripts\activate
.venv\Scripts\python -m pip install -r requirements.txt
```

### 2. Start the backend

Use the venv Python explicitly on Windows:

```powershell
.venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
```

Backend URLs:

- Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 3. Start the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL:

- [http://localhost:5173](http://localhost:5173)

If needed, point the frontend to a different backend:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
```

## Speech Features

### Speech-to-text

Endpoint:

- `POST /speech-to-text`

It accepts uploaded audio and returns:

```json
{
  "text": "",
  "language": "",
  "error": ""
}
```

The frontend mic button:

- records browser audio
- sends it to `/speech-to-text`
- fills the query box with the transcript
- sends transcript language to `/chat`

### Text-to-speech

Endpoint:

- `POST /text-to-speech`

The project tries:

1. backend TTS
2. browser speech fallback in the frontend if backend TTS is unavailable

## Data Setup

Put chunked service data in:

```text
data/processed_chunks/
```

Supported formats:

- `.txt`
- `.md`
- `.json`

Preferred metadata-rich chunk format:

```json
{
  "title": "Passport Application - Required Documents",
  "text": "...",
  "service": "passport",
  "state": "national",
  "topic": "documents",
  "source_url": "https://www.passportindia.gov.in/...",
  "source_name": "passport.json"
}
```

### Bootstrap your uploaded chunk files

```powershell
.venv\Scripts\python scripts\setup_chunks.py
```

### Build fresh chunks from the source catalog

```powershell
.venv\Scripts\python scripts\gov_data_pipeline.py
```

## Backend Flow

`POST /chat`

1. Detect or receive the query language
2. Run safety filtering
3. Translate the query to English for retrieval
4. Detect service, state, and topic
5. Retrieve topic-matched chunks
6. Build the LLM prompt
7. Generate structured JSON
8. Normalize the response
9. Translate the final answer back to the user language if needed
10. Store a compact English memory summary for follow-ups

## Notes

- The LLM is instructed to draft answers in English first, and the backend handles localization.
- If retrieval confidence is weak, the backend returns a safer low-confidence response instead of forcing a bad answer.
- If the model is unavailable or returns invalid JSON, the backend still returns schema-shaped output.
- If speech-to-text fails, the frontend now shows the actual backend error message.
