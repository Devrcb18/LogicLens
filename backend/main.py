from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json
import uvicorn
from backend.utils import process_handwritten_images
from backend.agent_logic import verify

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

@app.post("/analyze")
async def analyze_math(file: UploadFile = File(None), text: str = Form(None)):
    try:
        if file:
            temp_path = os.path.join(BASE_DIR, f"temp_{file.filename}")
            with open(temp_path, "wb") as buffer:#temmp file
                shutil.copyfileobj(file.file, buffer)
            
            # img to latex
            ocr_data = process_handwritten_images(temp_path)
            
            # cleaning temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            # text ip fallback
            ocr_data = {"steps": [text], "ocr_confidence": 1.0}

        # langchain agent invoking
        result = verify(ocr_data)

        # result is already a dict
        return result

    except Exception as e:
        return {"error": str(e), "is_correct": False}
    
@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_file = os.path.join(FRONTEND_DIR, "index.html")
    with open(html_file, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
