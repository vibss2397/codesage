import os
from dotenv import load_dotenv
from agents.orchestrator_agent import OrchestratorAgent
from schemas.orchestrator_agent_schema import OrchestratorAgentInput, OrchestratorAgentOutput

# Load environment variables from .env file
load_dotenv()

problem1 = """
    Find the length of the longest substring without repeating characters.
    Given a string s, find the length of the longest substring without repeating characters.
    """
    
code1 = """
def lengthOfLongestSubstring(s):
    max_len = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            substring = s[i:j+1]
            if len(set(substring)) == len(substring):
                max_len = max(max_len, len(substring))
            else:
                break
    return max_len
"""

def test_orchestrator_agent_execution(orchestrator_agent):
    """
    Test the execution of the OrchestratorAgent with a sample problem.
    """
    # Given
    input_data = OrchestratorAgentInput(
        problem=problem1,
        code_solution=code1
    )
    
    # When
    result = orchestrator_agent.execute(input_data)
    
    # Then
    assert result is not None
    assert isinstance(result, OrchestratorAgentOutput)
    assert result.problem_area is not None
    assert result.hint is not None
    print(result)

if __name__ == "__main__":
    # This block allows running the test directly for debugging
    agent = OrchestratorAgent(api_key=os.getenv("GEMINI_API_KEY"))
    input_data = OrchestratorAgentInput(
        problem=problem1,
        code_solution=code1
    )
    result = agent.execute(input_data)
    print("Execution Result:", result)
