from pydantic import BaseModel, Field
from typing import List, Literal

class AnalyzeAgentInput(BaseModel):
    problem: str = Field(..., description="The problem statement to analyze.")
    code: str = Field(..., description="The code solution to analyze.")

class AnalyzeAgentOutput(BaseModel):
    result_source: Literal["RAG", "built-in"] # "RAG" or "built-in"
    current_complexity: str
    current_approach: str
    optimal_pattern: str
    edge_cases: List[str]