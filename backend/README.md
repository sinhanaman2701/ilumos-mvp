# iLumos backend

The backend owns the MongoDB Atlas connection. Original documents are stored in MongoDB GridFS; application metadata is stored in MongoDB collections.

## Local setup

1. Copy `.env.example` to `.env` and fill in the Atlas connection string.
2. Install `requirements.txt` in a Python virtual environment.
3. Start the API with:

```bash
uvicorn app.main:app --app-dir backend --reload --port 8000
```

The health endpoint is available at `http://localhost:8000/api/health`.

The `POST /api/chats` endpoint accepts one or more `claim_charts`, optional `supporting_documents`, and an optional `prompt`. It creates one chat/run, stores all original files in GridFS, deterministically extracts XLSX/PDF chart structure and image assets, sends the labeled draft to the configured Ollama evaluator, validates the approved rows, and runs the LangGraph workflow in the background. Use `GET /api/chats/{chat_id}/workspace` to reopen the latest persisted table, messages, and supporting-file references.

Messages are persisted through `POST /api/chats/{chat_id}/messages` and can reference uploaded supporting-document IDs with `file_ids`. The frontend uses the workspace endpoint when a recent chart is selected so the UI does not rely on in-memory state.

The active Send workflow uses Ollama Cloud’s direct OpenAI-compatible endpoint. Set `OLLAMA_BASE_URL=https://ollama.com/v1`, `OLLAMA_MODEL=nemotron-3-nano:30b-cloud`, and a valid `OLLAMA_API_KEY` in `backend/.env`. The prompts request JSON and disable reasoning output; the backend also validates and safely extracts evaluator-shaped JSON when a cloud model adds a preamble, fence, thinking block, or list wrapper. No local Ollama installation or daemon is required. The evaluator does not interpret images; image assets are preserved in GridFS and exposed through `/api/assets/{asset_id}`.

The frontend uses `NEXT_PUBLIC_API_BASE_URL` from `.env.local`; copy the repository `.env.local.example` to `.env.local` when the API is running on a different address.
