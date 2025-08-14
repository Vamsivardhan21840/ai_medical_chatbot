from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import base64, requests, io, os, logging
from PIL import Image
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    try:
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")

        # base64 encode
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        # validate image
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        def make_api_request(model: str):
            return requests.post(
                GROQ_API_URL,
                json={"model": model, "messages": messages, "max_tokens": 1000},
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )

        # Use Groq’s current vision models but keep old response keys for the UI
        models = [
            ("llama", "meta-llama/llama-4-scout-17b-16e-instruct"),
            ("llava", "meta-llama/llama-4-maverick-17b-128e-instruct"),
        ]

        # Always initialize keys so frontend never gets undefined
        responses = {"llama": "", "llava": ""}

        for label, model_name in models:
            resp = make_api_request(model_name)
            if resp.status_code == 200:
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                logger.info(f"Processed response from {label} ({model_name}): {answer[:120]}...")
                responses[label] = answer or ""  # ensure string
            else:
                logger.error(f"Error from {label} ({model_name}): {resp.status_code} - {resp.text}")
                responses[label] = (
                    f"Error from {label} ({model_name}): {resp.status_code} - "
                    f"{(resp.text or 'No details').strip()}"
                )

        return JSONResponse(status_code=200, content=responses)

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
