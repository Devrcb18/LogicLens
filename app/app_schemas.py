from pydantic import BaseModel, Field

class logic_lens_result(BaseModel):
    is_correct: bool = Field(..., description = 'whether the answer is correct or not')
    error_step_index: int = Field(..., description = 'index of the error in given solution (-1 if correct)')
    confidence_score: float = Field(..., gt=0, lt=1,description = 'how much correct the solution is')
    explanation: str = Field(..., description = 'feedback on your mistake')
    suggestion: str = Field(..., description ='how the student can improve himself')
    corrected_solution_latex: str = Field(...,description = 'correct solution in LaTex format')

result = {'is_correct':True,'error_step_index':2,'confidence_score':0.8,'explanation':'there','suggestion':'good try','corrected_solution_latex':'xyz'}    
validated_data=logic_lens_result(**result)
print(validated_data)
