from agents.base_agent import BaseAgent
from prompts.analyze_agent_prompt import SYSTEM_PROMPT as ANALYZE_AGENT_PROMPT
from modules.rag_store import CodingPatternRag  # Assuming this is defined elsewhere
from schemas.analyze_agent_schema import AnalyzeAgentInput, AnalyzeAgentOutput

class AnalyzeAgent(BaseAgent):
    def __init__(self, api_key, task_id=None, db=None, debug_mode=False):
        super().__init__(api_key, task_id, db)
        self.rag = CodingPatternRag(api_key)
        self.rag.add_patterns()
        self.system_prompt = ANALYZE_AGENT_PROMPT  # From prompts/
        self.debug_mode = debug_mode  # Optional debug mode
    
    def execute(self, input: AnalyzeAgentInput):
        """Main business logic method"""
        # 1. Use inherited status update
        self.update_status("Searching pattern database")
        
        # 2. Do RAG search directly (no tool calling)
        pattern_results = self.rag.search_similar(input.problem, k=3)
        pattern_results_llm = self.rag.convert_to_query(pattern_results)
        
        # 3. Use inherited model call
        self.update_status("Analyzing code complexity")
        user_prompt = f"""
        PROBLEM: {input.problem}
        CODE: {input.code}
        PATTERNS: {pattern_results_llm}
        """
        response_text = self.call_model(user_prompt)
        
        # 4. Use inherited JSON parsing
        return AnalyzeAgentOutput(**self.parse_json_response(response_text))