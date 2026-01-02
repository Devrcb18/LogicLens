from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage,HumanMessage
import os
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

JSON OUTPUT(STRICTLY FOLLOW):
{
  "is_correct": bool,
  "error_step_index": int,
  "explanation": "Why it's wrong",
  "suggestion": "How to think about it next time",
  "solution": "correct LaTeX solution",
  "total_confidence": float
}
"""
agent=create_agent(
    model=llm,
    tools=[],
    system_prompt=SystemMessage(
        content=system_message_content)
)
def verify(ocr_data):
    steps_text = "\n".join(ocr_data.get("steps", []))
    response = agent.invoke({"messages": [HumanMessage(content=steps_text)]})
    return(response['messages'][-1].content)
    