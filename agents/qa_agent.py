from google import genai
from google.genai import types
from typing import Dict, Any, List
import json
from prompts.qa_code_agent_prompt import SYSTEM_PROMPT as system_prompt

class QAAgent:
    """
    Agent meant to be a QA engineer.
    """
    
    def __init__(self, api_key: str):
        self.system_prompt = system_prompt
        
        # 1. Initialize the low-level client
        self.client = genai.Client(api_key=api_key)

    def qa_analysis(self, question: str, code_solution: str) -> Dict[str, Any]:
        """
        Perform QA analysis on the given question and code solution.
        
        Args:
            question (str): The leetcode problem statement.
            code_solution (str): The code solution to the problem.
        
        Returns:
            Dict[str, Any]: The analysis results in JSON format.
        """
        # Prepare the input data
        input_data = {
            "question": question,
            "code_solution": code_solution
        }
        
        # Call the GenAI API with the system prompt and input data
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt),
            contents=json.dumps(input_data, indent=4),
        )
        
        # Parse the response
        try:
            res = response.text
        except Exception as e:
            print(f"Error in response: {e}")
            return {}
        
        for separators in ("```json", "```"):
            res = res.replace(separators, "")
            
        try:
            return json.loads(res)
        except:
            print("Failed to parse JSON response")
            return {"error": "Invalid JSON response from the model"}
        
