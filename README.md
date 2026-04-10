# ai_medical_chatbot

AI-powered medical image assistant — upload an image and ask a question; the server calls vision-capable models (via the Groq API) and returns diagnostic/explanatory text.

## How it works

- A FastAPI backend (`app.py`) serves a single-page frontend and exposes a POST endpoint `/upload_and_query`.
- The frontend (`templates/index.html`) lets users upload an image and submit a text query.
- The server base64-encodes and validates the uploaded image, then sends it along with the query to Groq vision-capable models. Responses are returned to the browser and rendered.

## Project structure

- `ai_medical_chatbot/app.py` - FastAPI application; main web server and API endpoint.
- `ai_medical_chatbot/main.py` - standalone helper for testing the image->Groq flow from the command line.
- `ai_medical_chatbot/README.md` - this file.
- `templates/index.html` - frontend UI for image upload and query submission.
- `package.json` - Node package manifest present in the repo root (not required for the Python FastAPI server).

## Prerequisites

- Python 3.9+ (or compatible)
- A valid Groq API key saved as `GROQ_API_KEY` in your environment or a `.env` file.

## Quick setup

1. Create a `.env` file in the repo root (or set the environment variable):

```powershell
Set-Content .env "GROQ_API_KEY=your_api_key_here"
```

2. Install Python dependencies:

```bash
python -m pip install fastapi uvicorn jinja2 python-multipart pillow requests python-dotenv
```

3. Run the server (option A — run the module directly):

```powershell
python ai_medical_chatbot/app.py
```

Or (option B — use uvicorn):

```bash
uvicorn ai_medical_chatbot.app:app --reload --port 8000
```

4. Open your browser at `http://localhost:8000`, upload an image, and submit a question.

## API

- `GET /` — serves the frontend UI.
- `POST /upload_and_query` — accepts a multipart form with fields:
	- `image` (file)
	- `query` (string)

Response: JSON containing model outputs (keys for the models the app requests).

## Notes

- The application expects responses in the `choices[0].message.content` shape from the Groq API; if the API updates response schemas your client code may need adjustments.
- `package.json` exists but is not required for running the FastAPI server; remove or adapt it if you plan a Node-based backend.
- If you want, I can add a `requirements.txt` and a `.env.example` file.

## Next steps (suggested)

- Add `requirements.txt` for reproducible installs.
- Add `.env.example` (without real keys) to document environment variables.
- Add basic tests for `process_image()` in `main.py`.
