from agents.qa_agent import QAAgent
import json
from dotenv import load_dotenv
load_dotenv()
import os


if __name__ == "__main__":
    API_KEY = os.getenv('GEMINI_API_KEY')
    # Test problem and code
    problem = "Check if a string is a valid palindrome"
    code = """
    def is_palindrome(s):
        clean = ''.join(c.lower() for c in s if c.isalnum())
        for i in range(len(clean)):
            for j in range(len(clean)):
                if clean[i] != clean[len(clean)-1-i]:
                    return False
        return True
    """
    agent = QAAgent(API_KEY)
    print("🤖 QA AGENT TEST")
    print(f"Problem: {problem}")
    print(f"Code length: {len(code)} characters")
    print("\nAgent will perform QA analysis on this solution...")
    result = agent.qa_analysis(problem, code)
    print(json.dumps(result, indent=2))