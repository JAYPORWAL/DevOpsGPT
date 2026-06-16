from pydantic import BaseModel, Field
from typing import List

class InterviewEvaluation(BaseModel):
    score: int = Field(description="Numerical rating of the answer from 0 to 10")
    strengths: List[str] = Field(description="List of strong points, correct technical elements, or concepts mentioned")
    weaknesses: List[str] = Field(description="List of weak points, incorrect statements, or omitted details")
    improved_answer: str = Field(description="An optimized, comprehensive version of the answer that the candidate should use")
