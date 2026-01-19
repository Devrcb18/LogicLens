import base64
import os
import json
import re
from groq import Groq

def encode_image(image_path):
    """Encodes a local image to base64 string for the API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_handwritten_images(image_path):
   
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    base64_image = encode_image(image_path)
    prompt = """
    Analyze the math problem in this image. 
    1. Extract every individual step of the solution.
    2. Convert each step into a valid LaTeX string.
    3. Return ONLY a JSON object with this exact structure:
    {"steps": ["step1", "step2"], "ocr_confidence": 0.95}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        raw_content = chat_completion.choices[0].message.content
        print(f"DEBUG: Raw AI Response: {raw_content}")

        clean_json = re.sub(r'```json|```', '', raw_content).strip()
        data = json.loads(clean_json)
        
        return {
            "steps": data.get("steps", []),
            "ocr_confidence": data.get("ocr_confidence", 0.0)
        }

    except Exception as e:
        print(f"Vision Error: {e}")
        return {"steps": ["Error parsing steps"], "ocr_confidence": 0.0}
