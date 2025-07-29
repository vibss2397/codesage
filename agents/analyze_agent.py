from google import genai
from google.genai import types
from typing import Dict, Any, List
import json
from prompts.analyze_agent_prompt import SYSTEM_PROMPT as system_prompt

class AnalyzeAgent:
    """
    Analyze Agent built with the gemini SDK.
    """
    
    def __init__(self, rag_system, api_key: str):
        self.rag = rag_system
        self.system_prompt = system_prompt
        
        # 1. Initialize the low-level client
        self.client = genai.Client(api_key=api_key)

        # 2. Define the tool schema explicitly
        search_tool_schema = types.Tool(
            function_declarations=[
                {
                    "name": "search_patterns",
                    "description": "Search the coding patterns database for relevant algorithmic patterns and optimizations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for finding relevant coding patterns based on the problem description."
                            }
                        },
                        "required": ["query"],
                    }
                }
            ]
        )

        # 3. Create the configuration object for the model
        self.config = types.GenerateContentConfig(tools=[search_tool_schema])

    def search_patterns(self, query: str) -> Dict[str, Any]:
       """
       This is the actual Python function that executes the RAG search.
       """
       print(f"🤖 Executing RAG search for: '{query}'")
       results = self.rag.search_similar(query, k=5, diverse=True)
       
       formatted_results = [
           {
               "pattern": result["pattern"],
               "chunk_type": result["chunk_type"],
               "description": result["text"],
               "relevance": f"{result['similarity_score']:.3f}"
           } for result in results
       ]
       
       return {"results": formatted_results}
    
    def analyze_solution(self, problem_description: str, user_code: str) -> Dict[str, Any]:
        """
        Analyzes the solution using a manual, multi-turn chat.
        """
        user_prompt = f"""
        Analyze this coding solution:

        PROBLEM:
        {problem_description}

        USER'S CODE:
        {user_code}
        Please provide a complete technical analysis following your defined output format.
        """
        # 4. Manually create the chat session with history
        chat = self.client.chats.create(
            model="gemini-1.5-flash",
            config=self.config,
            history=[
            types.Content(parts=[types.Part(text=self.system_prompt)], role='user'),
            types.Content(parts=[types.Part(text="OK.")], role='model'),
            ]
        )
        # 5. Send the user prompt
        response = chat.send_message(user_prompt)
    
        # 6. Manually check for and execute function calls
        if response.function_calls:
            tool_execution_results: List[types.Part] = []
            for function_call in response.function_calls:
                if function_call.name == "search_patterns":
                    tool_result = self.search_patterns(**function_call.args)
                    # Append the result for the model
                    tool_execution_results.append(
                        types.Part(function_response=types.FunctionResponse(name="search_patterns", response=tool_result))
                    )
            
            # 7. Send the tool results back to the model to get the final answer
            final_response = chat.send_message(tool_execution_results)
            analysis_text = final_response.text
        else:
            # If no function was called, the first response is the final one
            analysis_text = response.text
            
        # 8. Parse the final JSON from the response text
        try:
            analysis_text = analysis_text.replace("```json", "").replace("```", "").strip()
            return json.loads(analysis_text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse final JSON from the model.", "raw_response": analysis_text}