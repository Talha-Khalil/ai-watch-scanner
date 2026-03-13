# Watch Scanner

Upload a watch image → get identification + projected price (from ~28k watch records).

## Setup on macOS

### 1. Prerequisites

- **Python 3.13+**  
  Install via [Homebrew](https://brew.sh): `brew install python@3.13`
- **uv** (recommended)  
  `curl -LsSf https://astral.sh/uv/install.sh | sh`  
  Or: `brew install uv`

### 2. Clone and install

```bash
cd watch-scanner
uv sync
```

This creates a venv and installs deps from `pyproject.toml` / `uv.lock`.

### 3. Environment

Create a `.env` in the project root:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional
VECTORSTORE_DIR=./vectorstore
EMBED_MODEL=all-MiniLM-L6-v2
OPENAI_VISION_MODEL=gpt-4o-mini
OPENAI_TEXT_MODEL=gpt-4o-mini
TOP_K=5
MAX_IMAGE_PX=1024
PORT=8000
```

Get an API key from [OpenAI](https://platform.openai.com/api-keys).

### 4. Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

API: **http://localhost:8000**  
- `POST /scan` — upload image (JPEG/PNG/WebP, max 6 MB)  
- `GET /health` — health + watches indexed count  
- Docs: **http://localhost:8000/docs**
