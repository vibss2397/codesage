from agents.base_agent import BaseAgent
from schemas.synthesis_agent_schema import SynthesisAgentInput, SynthesisAgentOutput
from prompts.synthesis_agent_prompt import SYSTEM_PROMPT as system_prompt

class SynthesisAgent(BaseAgent):
    """
    SynthesisAgent is responsible for synthesizing information from various sources.
    It extends the BaseAgent class to provide specific functionalities for synthesis tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize any additional attributes or methods specific to synthesis
        self.system_prompt = system_prompt

    def execute(self, input: SynthesisAgentInput) -> SynthesisAgentOutput:
        """
        Synthesize information from the provided data sources.
        
        :param data_sources: List of data sources to synthesize information from.
        :return: Synthesized information.
        """
        # Implement synthesis logic here
        prompt = f"""
        Problem: 
        {input.problem}

        Code: 
        {input.code}

        Analyze Agent Response:
        {input.analyze_agent_response}

        QA Agent Response:
        {input.qa_agent_response}

        Execution Analysis:
        {input.execution_analysis}

        Based on the above information, synthesize the most relevant insights and provide coaching guidance.
        """
        # Call the base agent's execute method with the synthesized prompt
        response_text = self.call_model(prompt)

        return SynthesisAgentOutput(
            **self.parse_json_response(response_text)
        )
