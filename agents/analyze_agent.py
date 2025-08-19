from agents.base_agent import BaseAgent
from prompts.analyze_agent_prompt import SYSTEM_PROMPT as ANALYZE_AGENT_PROMPT
from modules.coding_problems_knowledge_base import PROBLEM_KNOWLEDGE_BASE
from modules.rag_store import CodingProblemRagStore, build_rag_index
from schemas.analyze_agent_schema import AnalyzeAgentInput, AnalyzeAgentOutput

class AnalyzeAgent(BaseAgent):
    def __init__(self, api_key, task_id=None, db=None, debug_mode=False):
        super().__init__(api_key, task_id, db)
        index, index_to_key_map = build_rag_index(api_key, PROBLEM_KNOWLEDGE_BASE)
        self.rag = CodingProblemRagStore(api_key, PROBLEM_KNOWLEDGE_BASE, index, index_to_key_map)
        self.system_prompt = ANALYZE_AGENT_PROMPT  # From prompts/
        self.debug_mode = debug_mode  # Optional debug mode
    
    def execute(self, input: AnalyzeAgentInput):
        """Main business logic method"""
        # 1. Use inherited status update
        self.update_status("Searching pattern database")
        
        # 2. Do RAG search directly (no tool calling)
        pattern_results = self.rag.search(input.problem, k=3)
        
        # 3. Use inherited model call
        self.update_status("Analyzing code complexity")
        user_prompt = f"""
        PROBLEM: {input.problem}
        CODE: {input.code}
        PATTERNS: {pattern_results}
        """
        response_text = self.call_model(user_prompt)
        
        # 4. Use inherited JSON parsing
        return AnalyzeAgentOutput(**self.parse_json_response(response_text))