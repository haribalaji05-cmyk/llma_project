# GovAssist Core Intelligence Backend

This project is now scoped only to the `AI / Core Intelligence Engineer` responsibilities:

- RAG and document retrieval
- LLM integration with a strict JSON prompt contract
- Multilingual detection and translation flow
- Safety filtering
- Short-term follow-up memory

## Active Backend Flow

`POST /chat`

1. Detect user language
2. Run safety evaluation
3. Translate the query to English for retrieval
4. Retrieve top source chunks from `data/processed_chunks`
5. Build the strict master prompt
6. Generate JSON from the LLM
7. Normalize the output to the required schema
8. Translate user-facing answer fields back to the detected language
9. Store the turn in memory for follow-up handling

## Required Response Schema

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
  rag/
    embedder.py
    retriever.py
    vector_store.py
  llm/
    generator.py
    prompt.py
  multilingual/
    detect.py
    translate.py
  safety/
    filter.py
  memory/
    chat_memory.py
data/
  raw_docs/
  processed_chunks/
requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

## API LLM Mode

The backend now supports an API-first LLM setup.

Set these environment variables before starting the server:

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Optional:

```powershell
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

If you want to re-enable local Hugging Face fallback:

```powershell
$env:ENABLE_LOCAL_LLM_FALLBACK="true"
```

Without `OPENAI_API_KEY`, the backend will use the rule-based fallback unless local fallback is explicitly enabled.

## Better Data Setup

The retriever now supports metadata-rich chunk files shaped like:

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

You can bootstrap the uploaded chunk files into this format with:

```powershell
python scripts/setup_chunks.py
```

You can also build fresh metadata chunks from the source catalog in `data/source_catalog.json` with:

```powershell
python scripts/gov_data_pipeline.py
```

The retriever now:

- tags chunks by `service`
- detects state names in the query
- prefers state-matched chunks when available
- avoids loading duplicate `.txt` copies when `.json` exists

## Data Expectations

- Put cleaned and chunked government-service text inside `data/processed_chunks/`
- Supported bootstrap formats: `.txt`, `.md`, `.json`
- JSON files should contain a list of chunk strings

## Notes

- The prompt in `backend/llm/prompt.py` follows the master prompt contract you provided.
- `backend/llm/generator.py` includes safe JSON parsing and schema normalization.
- Unsafe queries return the same schema with `safety.is_safe = false`.
- If the model is unavailable or returns invalid JSON, the backend still returns valid schema-shaped output.
