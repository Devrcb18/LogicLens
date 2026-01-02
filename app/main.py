from fastapi import FastAPI, UploadFile, File, HTTPException
from agent.agent_logic import analyze_image
from app.app_schemas import LogicLensResult
app = FastAPI()
VALID_DATA_TYPES = ['image/png','image/jpeg']
@app.post('/processing',response_model=LogicLensResult, summary='analyzing hand written maths solution')
async def process_image(image: UploadFile=File(...)):
    if image.content_type not in VALID_DATA_TYPES:
        raise HTTPException(status_code=400,detail='Invalid Data Type')
    img = await image.read()
    if not img:
        raise HTTPException(status_code=400, detail='image missing')
    try:
        result = analyze_image(img)
    except Exception:
        raise HTTPException(status_code=500,detail='processing failed')
    return result
        
