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
You are the 'LogicLens' Math Auditor. Your goal is to find MISTAKES in student work, not just provide a solution.

AUDIT PROCESS:
1. Receive the list of math steps (OCR results).
2. For EACH transition (Line N to Line N+1), use the `verify_math_step` tool.
3. If the tool confirms a 'Logical Match', move to the next step.
4. If the tool detects an error:
   - Identify the EXACT index of the failing step.
   - Explain WHY it is wrong (e.g., "You subtracted 2 from the left but added 2 to the right").
   - Suggest a corrective hint instead of just the answer.

RESPONSE FORMAT (Strict JSON):
{
  "is_correct": boolean,
  "error_step_index": integer (or -1 if correct),
  "confidence_score": float (0.0-1.0),
  "explanation": "Tutor-style feedback pointing out the specific logic error",
  "suggestion": "How the student can fix their thinking",
  "corrected_solution_latex": "The full correct LaTeX block"
}
"""
agent=create_agent(
    model=llm,
    tools=[],
    system_prompt=SystemMessage(
        content=system_message_content)
)

# print(agent.get_graph().draw_mermaid())
def verify(ocr_text):
    response = agent.invoke({"messages": [HumanMessage(content=ocr_text)]})
    return(response['messages'][-1].content)
    
