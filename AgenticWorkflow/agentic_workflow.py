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

# Maximum characters allowed in the text portion of any agent prompt.
# The two images are passed as BinaryContent and do not count toward this limit.
MAX_TEXT_CHARS = 19_500


class StrategyAgent:
    """
    State Management Agent to maintain the state of the system.
    """
    
    def __init__(self, graphManager: MGraphManager):
        """
        Initialize the StateManagementAgent with a knowledge graph manager.
        """
        self.graphManager = graphManager
        self.memory = deque(maxlen=5)

        system_prompt = """
        You are an expert in developing a high-level strategy for monitoring and responding to a wildfire event.

        You will be given tools for maintaining a Knowledge Graph.
        You will be given a Satellite Image of the area, as well as a risk image.
        Red means danger, Green means safe.
        You will be given a history of your previous plans.

        You have three primary goals, in order of priority:
        Areas marked safe in the risk map can be ignored.
        Focus attention on areas that are at risk of fire or are currently on fire.
        
        In the areas at risk your priorities are:
        
        1. GOAL_HUMAN_LIFE: Preserve Human Life that could be impacted by fire
        2. GOAL_PROPERTY: Preserve Property that could be impacted by fire
        3. GOAL_EXTINGUISH: Extinguish Fires as fast as possible.


        You will track the status of these goals using a Knowledge Graph.
        Each goal is a top level entity of its own.
        The status of each goal is a predicate that will be updated as the simulation progresses.
        You will track a collection of locations of interest as child nodes for each goal.
        
        When given new information, you will update the status of the goals.

        You must consider:
        1. The current status of the goals.
        2. The new information from UAVs and satellites.
        3. The risk map.

        Your task is to generate a priority list of locations of interest to focus on.
        Your output should be of the following form:

        - GOAL_HUMAN_LIFE; LOCATION 55,55; PRIORITY 1
        - GOAL_HUMAN_LIFE; LOCATION 80,80; PRIORITY 2
        - GOAL_EXTINGUISH; LOCATION 30, 35; PRIORITY 3

        Areas that are not at risk of fire, or currently burning should not be included in the priority list.
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
        )(graphManager.insert_triplet_list)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.insert_predicate)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.remove_predicate)

    def _build_text_prompt(self, uav_messages) -> str:
        """
        Assemble the text portion of the StrategyAgent prompt and truncate it
        to MAX_TEXT_CHARS characters so the combined context stays within limits.
        The two satellite/risk images are passed separately as BinaryContent and
        are NOT counted against this limit.
        """
        history = list(reversed(self.memory))  # most-recent first
        text = (
            f"Here are the latest UAV messages: {uav_messages}\n\n"
            f"Here is a history of your previous responses:\n{history}"
        )
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
            text += "\n\n[... context truncated to fit token limit ...]"
        return text

    def run_agent(self, uav_messages, satellite_image, risk_image):
        """
        Run the state management agent to update the knowledge graph.
        """
        text_part = self._build_text_prompt(uav_messages)
        prompt = [
            text_part,
            satellite_image,
            risk_image
        ]

        response = self.agent.run_sync(prompt).output
        self.memory.append({
            "prompt": "Plan",
            "response": response
        })

        print(f"State Management Agent: {response}")

        return response

class DispatchAgent:
    """
    Strategy Agent to decide what actions to take.
    """
    def __init__(self, graphManager: MGraphManager):
        """
        Initialize the StrategyAgent with a knowledge graph manager.
        """

        self.memory = deque(maxlen=5)

        system_prompt = """
        You are an expert in translating high level fire control and monitoring stratgies into commands for a fleet of UAVs
        
        You will be given:
        
        ## Priority List
        A priority list will be provided by the strategy agent. You must follow this list to determine where to send UAVs.
        The priority list should be in the following format:

        - PRIORITY 1; LOCATION 55,55
        - PRIORITY 2; LOCATION 80,80
        - PRIORITY 3; LOCATION 30,35

        You will be given the following tools:

        ## History
        A list of your previous commands sent to the simulation.

        ## UAV Messages
        The current state and location of all UAVs in the simulation.
        Recon UAVs should be deployed to priority locations to obtain information.
        Recon UAVs should be spread out for maximum coverage.
        Extinguisher UAVs should be deployed to locations that are currently on fire.
        Resource UAVs should be deployed to locations that are at risk and have humans nearby to extract them.
        

        ## Images
        You will be given an image of the current simulation view, as well as a risk mask.

        ## Current Plan
        The current high-level plan generated by the strategy agent.


        Your job is to generate a dispatch plan that:
        1. Follows the priority list as closely as possible.
        2. Makes the most use of the current UAVs.
        3. Minimizes travel time between locations.
        """

        self.graphManager = graphManager
        self.agent = Agent(
            model,
            output_type=CommandCenterResponse,
            system_prompt=system_prompt
        )

    def run_agent(self, uav_messages, plan, satellite_image, risk_image) -> CommandCenterResponse:
        """
        Run the dispatch agent to translate the strategy plan into UAV commands.

        Args:
            uav_messages: Telemetry reports from all active UAVs.
            plan: Priority list string produced by the StrategyAgent.
            satellite_image: BinaryContent PNG of the current simulation view.
            risk_image: BinaryContent PNG of the UNet risk mask.
        """
        history = list(reversed(self.memory))  # most-recent first
        text = (
            f"Here are the latest UAV messages: {uav_messages}\n\n"
            f"Here is the current plan from the strategy agent:\n{plan}\n\n"
            f"Here is a history of your previous responses:\n{history}"
        )
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
            text += "\n\n[... context truncated to fit token limit ...]"

        prompt = [
            text,
            satellite_image,
            risk_image
        ]

        response = self.agent.run_sync(prompt).output
        self.memory.append({
            "prompt": "dispatch",
            "response": str(response)
        })

        print(f"Dispatch Agent Response: {response}")

        return response

class AssistantAgent:
    """
    Assistant agent to provide an interface between humans the system as a whole.
    """
    def __init__(self, graphManager: MGraphManager):
        """
        Initialize the AssistantAgent with a knowledge graph manager.
        """
        self.memory = deque(maxlen=5)
        self.graphManager = graphManager

        system_prompt = """
        You are a helpful assistant that can answer questions about a wildfire event.
        You have access to a knowledge graph that contains information about the event.
        Use the tools provided to answer the query.
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
        history = list(reversed(self.memory))  # most-recent first
        text = (
            f"Here is the query: {query}\n\n"
            f"Here is a history of your previous responses:\n{history}"
        )
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
            text += "\n\n[... context truncated to fit token limit ...]"

        response = self.agent.run_sync(text).output
        self.memory.append({
            "prompt": query,
            "response": str(response)
        })
        return response

class CommandCenterAgent:

    def __init__(self):
        """
        Initialize the CommandCenterAgent with a knowledge graph manager.
        """
        self.graphManager = MGraphManager()
        self.strategy_agent = StrategyAgent(self.graphManager)
        self.dispatch_agent = DispatchAgent(self.graphManager)
        self.assistant_agent = AssistantAgent(self.graphManager)

        self.current_plan = None
        self.plan_age = 0
        self.plan_update_interval = 2

    def update(self, uav_messages, satellite_image, risk_image):
        """
        Update the knowledge graph with new information from UAVs and satellites.
        """
        sat_bytes = BinaryContent(data=satellite_image, media_type='image/png')
        risk_bytes = BinaryContent(data=risk_image, media_type='image/png')

        # develop a plan with the strategy agent
        if self.current_plan is None or self.plan_age >= self.plan_update_interval:
            self.current_plan = self.strategy_agent.run_agent(uav_messages, sat_bytes, risk_bytes)
            self.plan_age = 0

        # generate a new set of commands for the UAVs
        response = self.dispatch_agent.run_agent(uav_messages, self.current_plan, sat_bytes, risk_bytes)
        self.plan_age += 1

        return [cmd.model_dump() for cmd in response.commands]
        
    def query(self, query):
        """
        Query the knowledge graph for information.
        """
        return self.assistant_agent.run_agent(query)
