from typing import Dict, Any
from prompts.qa_code_agent_prompt import SYSTEM_PROMPT as system_prompt
from agents.base_agent import BaseAgent
from schemas.qa_agent_schema import QaAgentInput, QaAgentOutput

class QaAgent(BaseAgent):
    def __init__(self, api_key, task_id=None, db=None):
        super().__init__(api_key, task_id, db)
        self.system_prompt = system_prompt

    def execute(self, input: QaAgentInput) -> QaAgentOutput:
        """
        Main business logic method for analyzing a solution.
        
        Args:
            question (str): The leetcode problem statement.
            code_solution (str): The code solution to the problem.
        
        Returns:
            Dict[str, Any]: The analysis results in JSON format.
        """
        self.update_status("Analyzing code solution for QA stuff")
        
        # Prepare the user prompt
        user_prompt = f"""
        QUESTION: {input.question}

        CODE SOLUTION: {input.code_solution}

        ANALYSIS CONTEXT:
            - Edge case categories: {input.analyze_results.edge_cases}
            - Current complexity: {input.analyze_results.current_complexity}
            - Optimal pattern: {input.analyze_results.optimal_pattern}

        Please provide a comprehensive QA analysis based on the above context.
        """
        
        # Call the model with the user prompt
        response_text = self.call_model(user_prompt)
        
        # Parse the JSON response
        return QaAgentOutput(
            **self.parse_json_response(response_text)
        )