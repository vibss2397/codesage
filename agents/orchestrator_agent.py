from agents.base_agent import BaseAgent
from agents.analyze_agent import AnalyzeAgent
from agents.qa_agent import QaAgent
from agents.synthesis_agent import SynthesisAgent

from modules.code_executor import CodeExecutor
from modules.execution_results_analyzer import ExecutionAnalyzer

from schemas import (
    OrchestratorAgentInput,
    OrchestratorAgentOutput,
    AnalyzeAgentInput,
    AnalyzeAgentOutput,
    QaAgentInput,
    QaAgentOutput,
    SynthesisAgentInput,
    SynthesisAgentOutput
)

class OrchestratorAgent(BaseAgent):
    def __init__(self, api_key, task_id=None, db=None):
        super().__init__(api_key, task_id, db)
        self.system_prompt = "You are the orchestrator agent responsible for coordinating the workflow of various agents."
        self.code_executor = CodeExecutor()
        self.execution_analyzer = ExecutionAnalyzer()
        self.api_key = api_key
        self.task_id = task_id
        self.db = db
    
    def execute(self, input: OrchestratorAgentInput) -> OrchestratorAgentOutput:
        """
        Main business logic method for orchestrating the agents.
        
        Args:
            input_data (dict): Input data to be processed by the agents.
        
        Returns:
            dict: The final output after processing by all agents.
        """
        self.update_status("Orchestrating agents to synthesize the solution")
        
        # Step 1: Analyze the code solution
        self.update_status("Spawning Analyze Agent")
        agent = AnalyzeAgent(self.api_key, self.task_id, self.db)
        problem_analysis = agent.execute(
            AnalyzeAgentInput(
                problem=input.problem,
                code=input.code_solution
            )
        )

        # Step 2: Perform QA analysis
        self.update_status("Spawning QA Agent")
        agent = QaAgent(self.api_key, self.task_id, self.db)
        qa_results = agent.execute(
            QaAgentInput(
                question=input.problem,
                code_solution=input.code_solution,
                analyze_results=problem_analysis
            )
        )

        # Step 3: Execute the code solution
        self.update_status("Executing code solution")
        execution_results = self.code_executor.run_test_cases(input.code_solution, qa_results.test_cases)

        # Step 4: Analyze execution results
        self.update_status("Analyzing execution results")
        execution_analysis = self.execution_analyzer.analyze_results(execution_results)

        # Step 5: Synthesize the final output
        self.update_status("Spawning Synthesis Agent")
        agent = SynthesisAgent(self.api_key, self.task_id, self.db)
        synthesis_output = agent.execute(
            SynthesisAgentInput(
                problem=input.problem,
                code=input.code_solution,
                analyze_agent_response=problem_analysis,
                qa_agent_response=qa_results,
                execution_analysis=execution_analysis
            )
        )
        self.update_status("Orchestration complete")
        return OrchestratorAgentOutput(
            problem_area=synthesis_output.priority_focus,
            hint=synthesis_output.hint
        )