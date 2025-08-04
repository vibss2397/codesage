from pydantic import BaseModel, Field
from typing import List

from schemas.analyze_agent_schema import AnalyzeAgentOutput


class QaAgentInput(BaseModel):
    """
    QAAgentInput defines the input schema for the QAAgent.
    It includes fields necessary for the QA analysis process.
    """
    question: str = Field(..., description="The leetcode problem statement.")
    code_solution: str = Field(..., description="The code solution to the problem.")
    analyze_results: AnalyzeAgentOutput = Field(..., description="Analysis results from the Analyze Agent.")


class CodeQuality(BaseModel):
    naming: str = Field(..., description="Analysis of variable naming conventions.")
    structure: str = Field(..., description="Analysis of code structure and readability.")
    docs: str = Field(..., description="Documentation and comments analysis.")
    errors: str = Field(..., description="Error handling analysis.")

class Improvement(BaseModel):
    type: str = Field(..., description="Type of improvement (e.g., performance, style, typing).")
    action: str = Field(..., description="Action to take for the improvement.")
    impact: str = Field(..., description="Impact of the improvement (e.g., performance gain, readability).")

class QaAgentOutput(BaseModel):
    """
    QAAgentOutput defines the output schema for the QAAgent.
    It includes fields for the analysis results in JSON format.
    """
    problem_assumptions: List[str] = Field(..., description="List of problem constraints and assumptions.")
    test_cases: List[dict] = Field(..., description="List of test cases with input, expected output, and category.")
    code_quality: CodeQuality = Field(..., description="Analysis of code quality including naming, structure, documentation, and error handling.")
    improvements: List[Improvement] = Field(..., description="List of suggested improvements with type, action, and impact.")