
from google import genai
from google.genai import types
import json

class BaseAgent:
    def __init__(self, api_key, task_id=None, db=None):
        self.client = genai.Client(api_key=api_key)
        self.task_id = task_id
        self.db = db
        self.system_prompt = ""  # Subclasses override
    
    def update_status(self, message, progress=None):
        """Update task status in shared DB"""
        if self.db and self.task_id:
            self.db.update_task_status(self.task_id, message, progress)
        print(f"Status Update: {message} (Progress: {progress})")

    def call_model(self, user_prompt, **kwargs):
        """Make API call with system prompt"""
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt
            ),
            contents=user_prompt,
        )
        return response.text
    
    def parse_json_response(self, response_text):
        """Parse JSON from model response"""
        # Handle ```json blocks, cleanup, etc.
        cleaned = response_text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw_response": response_text}
    
    def execute(self, input):
        """
        Execute the agent's task with the provided input.
        
        :param input: Input data for the agent.
        :return: Result of the execution.
        """
        pass  # Subclasses implement this method