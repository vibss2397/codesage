from pydantic import BaseModel
from typing import List, Literal

from schemas.analyze_agent_schema import AnalyzeAgentOutput
from schemas.qa_agent_schema import QaAgentOutput

class SynthesisAgentInput(BaseModel):
    """
    SynthesisAgentInput defines the input schema for the SynthesisAgent.
    It includes fields necessary for the synthesis process.
    """
    problem: str
    code: str
    analyze_agent_response: AnalyzeAgentOutput
    qa_agent_response: QaAgentOutput
    execution_analysis: dict


class SynthesisAgentOutput(BaseModel):
    priority_focus: Literal["correctness", "algorithmic", "quality"]
    priority_reasoning: str
    socratic_hints: List[str]
    positive_feedback: str
    hint: str