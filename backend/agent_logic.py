from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage,HumanMessage
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['groq_api_key'] = os.getenv('GROQ_API_KEY')
# llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0.34)
system_message_content = """
You are the 'LogicLens' Math Debugger.

INPUT: You will receive a list of LaTeX steps and an OCR confidence score.
MISSION:
- Audit every transition. Identify EXACTLY which step (index) violates math rules.
- Don't just solve it; explain the logic gap (e.g., "You forgot to flip the sign").
- Provide a 'Tutor Suggestion' to help the student learn.

Strictly Return ONLY a valid JSON object with this exact structure:
{
  "is_correct": bool,
  "error_step_index": int,
  "explanation": "Why it's wrong",
  "suggestion": "How to think about it next time",
  "solution": "correct LaTeX solution",
  "total_confidence": float
}
"""
def verify(ocr_data):
    steps_text = "\n".join(ocr_data.get("steps", []))
    messages = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=steps_text)
    ]
    response = llm.invoke(messages, response_format={"type": "json_object"})
    content = response.content

    # Try to parse the entire content as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # If that fails, try to extract JSON from $\boxed{...}$
    match = re.search(r'\$\\boxed\{(.*)\}\$', content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Fallback: return error
    return {"error": "Failed to parse AI response", "is_correct": False}
    
