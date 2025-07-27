from rag_store import CodingPatternRag
from agents.analyze_agent import AnalyzeAgent
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
    
    print("🤖 ANALYZE AGENT TEST")
    print(f"Problem: {problem}")
    print(f"Code length: {len(code)} characters")
    print("\nAgent will use RAG tools to analyze this solution...")
    
    # Note: Requires initialized RAG system
    rag = CodingPatternRag()
    rag.add_patterns()
    agent = AnalyzeAgent(rag, API_KEY)
    result = agent.analyze_solution(problem, code)
    print(json.dumps(result, indent=2))