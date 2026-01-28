from langchain_groq import ChatGroq
from langchain.messages import SystemMessage,HumanMessage
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
os.environ['groq_api_key'] = os.getenv('GROQ_API_KEY')
llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
system_message_content = """
You are the 'LogicLens' Advanced Math Solver.

CRITICAL INSTRUCTIONS:
1. The OCR data may contain syntax errors like unmatched brackets () or {}. Fix these in your 'solution' field.
2. The 'solution' field MUST be a string with each step on a new line.
3. Each line should be a complete LaTeX mathematical expression.
4. Steps must be separated by actual newline characters (\n).
5. Use proper LaTeX commands for fractions, exponents, etc.
6. Include all intermediate steps in the derivation.
7. Handle all types of problems: algebra, calculus (derivatives, integrals), trigonometry, complex numbers, etc.
8. Example format:
y = (2x^{3} - 3x^{2} - x - 1)^{10}
\\frac{dy}{dx} = 10(2x^{3} - 3x^{2} - x - 1)^{9} \\cdot (6x^{2} - 6x - 1)
= 60x^{2}(2x^{3} - 3x^{2} - x - 1)^{9} - 60x(2x^{3} - 3x^{2} - x - 1)^{9} - 10(2x^{3} - 3x^{2} - x - 1)^{9}

INPUT: You will receive a list of LaTeX steps and an OCR confidence score.
MISSION:
- If the steps are correct, provide the complete solution with all steps.
- If incorrect, identify the error and provide the corrected full solution.
- Always provide a complete, step-by-step solution.

Strictly Return ONLY a valid JSON object with this exact structure:
{
  "is_correct": bool,
  "error_step_index": int,
  "explanation": "Why it's wrong or confirmation if correct",
  "suggestion": "How to think about it next time",
  "solution": "LaTeX solution with steps separated by newlines",
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
        data = json.loads(content)
        # Override total_confidence with ocr_confidence
        data["total_confidence"] = ocr_data.get("ocr_confidence", 0.9)
        return data
    except json.JSONDecodeError:
        pass

    # If that fails, try to extract JSON from $\boxed{...}$
    match = re.search(r'\$\\boxed\{(.*)\}\$', content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            data["total_confidence"] = ocr_data.get("ocr_confidence", 0.9)
            return data
        except json.JSONDecodeError:
            pass

    # Fallback: return error
    return {"error": "Failed to parse AI response", "is_correct": False, "total_confidence": ocr_data.get("ocr_confidence", 0.9)}
    
