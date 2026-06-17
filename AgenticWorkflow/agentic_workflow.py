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

        The top left corner of all images is the ORIGIN (0,0).

        You have three primary goals, in order of priority:
        
        1. GOAL_HUMAN_LIFE: Preserve Human Life that could be impacted by fire
        2. GOAL_PROPERTY: Preserve Urban and Housing areas.
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

        The satellite image is color coded as follows:

        WildFireState.EMPTY: (0, 0, 0),             # Black
        WildFireState.GRASSLAND: (124, 252, 0),     # Lawn Green (Light)
        WildFireState.SHRUB: (34, 139, 34),         # Forest Green (Medium)
        WildFireState.TREE: (0, 100, 0),            # Dark Green (Dark)
        WildFireState.FIRE: (255, 69, 0),           # Orange-Red
        WildFireState.BURNING: (255, 0, 0),         # Red
        WildFireState.HOUSING: (255, 140, 0),       # Dark Orange
        WildFireState.URBAN: (255, 255, 255),       # White
        WildFireState.ASH: (128, 128, 128),         # Grey
        WildFireState.WATER: (0, 0, 255)            # Blue

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

        ## UAVs
        You have three types of UAV:
        RECON: These have large detector ranges. They are good for tracking fire. They CANNOT Extinguish Fire. They CANNOT rescue humans.
        EXTINGUISH: These have small detector ranges. They CAN Extinguish Fire. They CANNOT rescue humans.
        RESCUE: These have small detector ranges. They CANNOT Extinguish Fire. They CAN ONLY rescue humans.

        RESCUE UAVs should ONLY be used to RECUSE Humans. You should always redirect them to the humans in most danger.


        ## UAV STATES
        UAVs in CRUISING State are working autonomously. They will continue to perform their tasks until they are given new commands.
        UAVs in TRAVELLING State are travelling to a waypoint.
        CRUISING UAVs can be given new waypoints to redirect them to a new location on the map where they can be more effective.

        ## UAV Behaviour
        A UAV will remain close to the last waypoint it was given. If a UAV is given a new waypoint, it will travel to that waypoint and then remain close to it.
        A UAV will continue to perform its task until it is given new commands.
        A UAV can be given a new waypoint to redirect it to a new location on the map where it can be more effective.

        ## UAV Messages
        The current state and location of all UAVs in the simulation.
        Recon UAVs should be deployed to priority locations to obtain information.
        Recon UAVs should be spread out for maximum coverage.
        Extinguisher UAVs should be deployed to locations that are currently on fire.
        Resource UAVs should be deployed to locations that are at risk and have humans nearby to extract them.
        

        ## Images
        You will be given an image of the current simulation view, as well as a risk mask.
        The top left corner of all images is the ORIGIN (0,0).

        The satellite image is color coded as follows:

        WildFireState.EMPTY: (0, 0, 0),             # Black
        WildFireState.GRASSLAND: (124, 252, 0),     # Lawn Green (Light)
        WildFireState.SHRUB: (34, 139, 34),         # Forest Green (Medium)
        WildFireState.TREE: (0, 100, 0),            # Dark Green (Dark)
        WildFireState.FIRE: (255, 69, 0),           # Orange-Red
        WildFireState.BURNING: (255, 0, 0),         # Red
        WildFireState.HOUSING: (255, 140, 0),       # Dark Orange
        WildFireState.URBAN: (255, 255, 255),       # White
        WildFireState.ASH: (128, 128, 128),         # Grey
        WildFireState.WATER: (0, 0, 255)            # Blue

        ## Current Plan
        The current high-level plan generated by the strategy agent.

        ## Task
        Your task is to:
        1. Issue a new command to *EVERY* UAV, even if they are already at their target location or Travelling.
        2. You must use the plan to determine how to prioritize the UAVs.
        3. You MUST use RESCUE UAVs to save endangered humans AND NOTHING ELSE.
        4. You MUST use EXTINGUISH UAVs to extinguish fires AND NOTHING ELSE.
        5. You MUST use RECON UAVs to obtain information AND NOTHING ELSE.
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
            f"Generate commands for EVERY available UAV in the fleet. This include UAVs that are travelling or at home base.\n\n"
            f"The number of UAVs is {len(uav_messages)}. Generate a command for each one. Do not miss any.\n\n"
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
        You have tools to acce the knowledge graph that contains information about the current goals and historical
        priorities.
        You have access to a plan that represents what the high-level strategy agent has decided to do. 
        You have access to a history of your previous interactions.

        Your task is to provide a clear and concise answer to operator queries and update the knowledge base
        as needed based on the operators requests.
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

    def run_agent(self, query:str, current_plan:str) -> str:
        """
        Run the assistant agent to answer a query.
        """
        history = list(reversed(self.memory))  # most-recent first
        text = (
            f"Here is the query: {query}\n\n"
            f"Current High Level Plan: {current_plan}\n\n"
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
        self.plan_update_interval = 5

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
        return self.assistant_agent.run_agent(query, self.current_plan)
