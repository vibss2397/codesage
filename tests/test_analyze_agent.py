from agents.analyze_agent import AnalyzeAgent
from agents.qa_agent import QAAgent
import json
from dotenv import load_dotenv
load_dotenv()
import os


    # Test Case 2: Sliding Window (distinctive keywords)
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
agent = AnalyzeAgent(api_key=os.getenv("GEMINI_API_KEY"))
result = agent.analyze_solution(problem1, code1)
print(result)

print('running qa agent')
qa_agent = QAAgent(api_key=os.getenv("GEMINI_API_KEY"))
result_qa = qa_agent.analyze_solution(
    question=problem1,
    code_solution=code1,
    analyze_results=result
)
print(json.dumps(result_qa, indent=2))