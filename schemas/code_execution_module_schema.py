from pydantic import BaseModel, Field
from typing import List

class CodeExecutionModuleInput(BaseModel):
    """
    CodeExecutionModuleInput defines the input schema for the code execution module.
    It includes fields necessary for executing code and analyzing results.
    """
    code: str = Field(..., description="The code to be executed.")
    test_cases: List[dict] = Field(..., description="List of test cases with input and expected output.")

class codeExecutionResult(BaseModel):
    """
    CodeExecutionResult defines the result schema for code execution.
    It includes fields for the actual output, expected output, and pass/fail status.
    """
    name: str = Field(..., description="Name of the test case.")
    input: dict = Field(..., description="Input for the test case.")
    actual_output: any = Field(..., description="Actual output from executing the code.")
    expected_output: any = Field(..., description="Expected output for the test case.")
    passed: bool = Field(..., description="Whether the test case passed or failed.")
    error: str = Field(None, description="Error message if any occurred during execution.")
    test_category: str = Field("unknown", description="Category of the test case.")


class CodeExecutionModuleOutput(BaseModel):
    """
    CodeExecutionModuleOutput defines the output schema for the code execution module.
    It includes fields for the results of code execution against test cases.
    """
    results: List[codeExecutionResult] = Field(..., description="List of results for each test case, including actual output and pass/fail status.")


class ExecutionResultsAnalysis(BaseModel):
    """
    ExecutionResultsAnalysis defines the analysis of execution results.
    It includes fields for overall pass/fail status and any errors encountered.
    """
    total_tests: int = Field(..., description="Total number of test cases executed.")
    passed: int = Field(..., description="Number of test cases that passed.")
    failed: int = Field(..., description="Number of test cases that failed.")
    pass_rate: str = Field(..., description="Pass rate as a percentage.")
    failed_tests: dict = Field(..., description="Dictionary of failed tests with reasons for failure.")