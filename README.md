# AI Medical Chatbot

An educational FastAPI application for asking questions about uploaded medical
images. It uses the Groq API for image-capable model responses.

## Setup

1. Install Python 3.9 or later.
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `GROQ_API_KEY` to a valid Groq API
   key. Keep `.env` private; it is excluded from Git.

## Run

From the repository root:

```powershell
python -m uvicorn app:app --reload --port 8000
```

Open `http://127.0.0.1:8000` in a browser.

The image analysis endpoint is `POST /upload_and_query`, with multipart form
fields `image` and `query`.

## Disclaimer

This application is for educational purposes only and is not a substitute for
professional medical advice, diagnosis, or treatment.
