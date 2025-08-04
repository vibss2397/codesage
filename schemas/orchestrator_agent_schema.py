from pydantic import BaseModel, Field
from typing import List, Literal

class OrchestratorAgentInput(BaseModel):
    """
    OrchestratorAgentInput defines the input schema for the Orchestrator Agent.
    It includes fields necessary for coordinating between different agents.
    """
    problem: str = Field(..., description="The problem statement to be solved.")
    code_solution: str = Field(..., description="The code solution to the problem.")

class OrchestratorAgentOutput(BaseModel):
    """
    OrchestratorAgentOutput defines the output schema for the Orchestrator Agent.
    It includes fields for the final synthesized output after coordinating between agents.
    """
    problem_area: Literal["correctness", "algorithmic", "quality"] = Field(
        ..., description="Priority focus for synthesis."
    )
    hint: str