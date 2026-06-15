
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.anthropic import AnthropicModel

import json

from dotenv import load_dotenv
import os
from collections import deque

from graphdb import MGraphManager

from pydantic_models import CommandCenterResponse

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

model = AnthropicModel(
    'claude-haiku-4-5', 
    provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY)
)   


class StateManagementAgent:
    """
    State Management Agent to maintain the state of the system.
    """
    
    def __init__(self, graphManager: MGraphManager):
        """
        Initialize the StateManagementAgent with a knowledge graph manager.
        """
        self.graphManager = graphManager
        system_prompt = """
        """

        self.agent = Agent(
            model,
            system_prompt=system_prompt
        )

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_entity_name)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_predicate)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.insert_predicate)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.remove_predicate)

    def run_agent(self, uav_messages, satellite_image, risk_image):
        """
        Run the state management agent to update the knowledge graph.
        """
        pass

class StrategyAgent:
    """
    Strategy Agent to decide what actions to take.
    """
    def __init__(self, graphManager: MGraphManager):
        """
        Initialize the StrategyAgent with a knowledge graph manager.
        """
        system_prompt = """
        You are the Strategy Agent for an urban disaster response system. Your role is to interpret the current state of the simulation and issue precise commands to the UAV fleet (Reconnaissance, Fire Extinguisher, Medical Rescue) to contain the disaster and save lives.

        You will be given two tools to access the Knowledge Graph:
            * Query by Entity Name: Allows you search the KG for the relationships involving a specific entity.
            * Query by Predicate: Allows you search the KG for all edges of a specific predicate.

        Your reasoning process must follow these steps:

            1. Assess the Threat:
            - Identify the location and intensity of the fire (Hotspots) from the current graph data.
            - Determine if there are visible survivors in need of extraction.

            2. Evaluate Asset Status:
            - Check the current position and fuel/water levels of each UAV.
            - Identify which UAVs are "Available" (not busy) and "Loaded" (carrying payload).

            3. Formulate the Mission:
            - **Firefighting**: The Extinguisher UAV is priority. Its target should be the "Center of Mass" of all hot spots, or the single hottest point if clearly defined. The target coordinates must be strictly between [0, 99] for both X and Y.
            - **Search**: The Recon UAV should be directed to search the areas surrounding the fire, prioritizing the densest urban grids, to locate survivors or assess structural damage.
            - **Rescue**: If survivors are detected, the Rescue UAV must be deployed to their coordinates immediately.

            4. Conflict Resolution & Command Generation:
            - **No Fire/No Survivors**: If there is no active fire and no detected survivors, issue RECALL commands for all UAVs to return to base (0,0) to conserve resources.
            - **Busy Assets**: If the only available UAV is currently performing a task (e.g., the Extinguisher is en route to a hotspot), do NOT re-route it unless there is an immediate, greater threat (e.g., a new hotspot appearing). Instead, wait for it to complete its mission or recall it explicitly.
            - **Coordinates**: Ensure all target coordinates are valid floats within the simulation boundaries (0.0 to 99.0).

            Output Format:
            - Provide the commands in the specified JSON format, ensuring the `commands` list is empty if no action is required.
        """

        self.graphManager = graphManager
        self.agent = Agent(
            model,
            output_type=CommandCenterResponse,
            system_prompt=system_prompt
        )

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_entity_name)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_predicate)

    def run_agent(self) -> CommandCenterResponse:
        """
        Run the strategy agent to decide what actions to take.
        """
        return self.agent.run_sync().output

class AssistantAgent:
    """
    Assistant agent to provide an interface between humans the system as a whole.
    """
    def __init__(self, graphManager: MGraphManager):
        """
        Initialize the AssistantAgent with a knowledge graph manager.
        """
        self.graphManager = graphManager

        system_prompt = """
        
        """

        self.agent = Agent(
            model,
            system_prompt=system_prompt
        )
        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_entity_name)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_predicate)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.insert_predicate)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.remove_predicate)

    def run_agent(self, query) -> str:
        """
        Run the assistant agent to answer a query.
        """
        return self.agent.run_sync().output

class CommandCenterAgent:

    def __init__(self):
        """
        Initialize the CommandCenterAgent with a knowledge graph manager.
        """
        self.graphManager = MGraphManager()
        self.state_management_agent = StateManagementAgent(self.graphManager)
        self.strategy_agent = StrategyAgent(self.graphManager)
        self.assistant_agent = AssistantAgent(self.graphManager)

    def update(self, uav_messages, satellite_image, risk_image):
        """
        Update the knowledge graph with new information from UAVs and satellites.
        """
        sat_bytes = BinaryContent(satellite_image)
        risk_bytes = BinaryContent(risk_image)

        # update the KG State with the state management agent
        self.state_management_agent.run_agent(uav_messages, sat_bytes, risk_bytes)

        # generate a new set of commands for the UAVs
        return self.strategy_agent.run_agent()
        
    def query(self, query):
        """
        Query the knowledge graph for information.
        """
        return self.assistant_agent.run_agent(query)
