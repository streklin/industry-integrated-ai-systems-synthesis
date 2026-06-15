import random
import math
import os
import sys

# Ensure DispatchAgent directory is in sys.path so modules can import DecisionTransformer
current_dir = os.path.dirname(os.path.abspath(__file__))
dispatch_agent_dir = os.path.join(current_dir, "DispatchAgent")
if dispatch_agent_dir not in sys.path:
    sys.path.insert(0, dispatch_agent_dir)

from WildFireCA.WildFireCA import WildfireCA, WildFireState
from HumanAgents.HumanAgent import HumanAgent, HumanAgentState
from UAVAgents.UAVBase import ReconUAV, FireControlUAV, UAVState
from UAVAgents.UAVSimulatorModule import UAVSimulator
from DispatchAgent import DispatchActionMessageValidator

class WildfireUAVEnv:
    def __init__(self, num_recon_uavs: int, num_extinguish_uavs: int, width: int = 100, height: int = 100, num_humans: int = 15, seed: int = None):
        self.num_recon_uavs = num_recon_uavs
        self.num_extinguish_uavs = num_extinguish_uavs
        self.width = width
        self.height = height
        self.num_humans = num_humans
        self.initial_seed = seed
        
        self.ca = None
        self.human_agents = []
        self.uav_simulator = None
        self.validator = DispatchActionMessageValidator()
        self.step_count = 0
        
    def reset(self, seed=None):
        self.step_count = 0
        current_seed = seed if seed is not None else (self.initial_seed if self.initial_seed is not None else random.randint(0, 100000))
        
        # Initialize CA
        self.ca = WildfireCA(width=self.width, height=self.height, seed=current_seed)
        self.ca.regenerate(seed=current_seed)
        
        # Spawn human agents
        self.human_agents = []
        attempts = 0
        while len(self.human_agents) < self.num_humans and attempts < 1000:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            cell = self.ca.grid[x][y]
            
            if cell.state not in [WildFireState.WATER, WildFireState.BURNING, WildFireState.FIRE]:
                activity = random.choice([HumanAgentState.HIKING, HumanAgentState.CAMPING])
                agent = HumanAgent(x=x, y=y, max_x=self.width - 1, max_y=self.height - 1, activity_type=activity)
                self.human_agents.append(agent)
            attempts += 1
            
        # Ignite fire
        burnable_cells = []
        for x in range(self.width):
            for y in range(self.height):
                cell = self.ca.grid[x][y]
                if cell.state in [WildFireState.GRASSLAND, WildFireState.SHRUB, WildFireState.TREE, WildFireState.HOUSING]:
                    burnable_cells.append(cell)
                    
        if burnable_cells:
            selected_cell = random.choice(burnable_cells)
            selected_cell.ignite()
        else:
            self.ca.grid[self.width // 2][self.height // 2].ignite()
            
        # Initialize UAVs and Simulator
        uavs = []
        uav_id = 0
        
        # Spawn Recon UAVs
        for _ in range(self.num_recon_uavs):
            uav = ReconUAV(uav_id=uav_id, width=self.width, height=self.height)
            uav.x = 20.0
            uav.y = 20.0
            uav.state = UAVState.HANGER
            uav.detection_range = 5.0
            uav.velocity = 1.0
            uavs.append(uav)
            uav_id += 1
            
        # Spawn Extinguish UAVs
        for _ in range(self.num_extinguish_uavs):
            uav = FireControlUAV(uav_id=uav_id, width=self.width, height=self.height)
            uav.x = 20.0
            uav.y = 20.0
            uav.state = UAVState.HANGER
            uav.detection_range = 3.0
            uav.velocity = 0.5
            uavs.append(uav)
            uav_id += 1
            
        self.uav_simulator = UAVSimulator(uavs)
        
        # Get status updates from all UAVs as the initial observation
        status_messages = []
        for uav in self.uav_simulator.uavs:
            status_messages.append(uav.latest_messages)
            
        return status_messages

    def step(self, action: list[str]) -> tuple[list[list[str]], float, bool, dict]:
        """
        Runs one step of environment simulation.
        
        Args:
            action: Command from the DispatchAgent (e.g. ["DEPLOY", "1", "45", "50", "TRANSMIT"])
            
        Returns:
            - next_state: list of UAV status messages
            - reward: float
            - done: bool
            - info: dict
        """
        self.step_count += 1
        reward = 0.0
        invalid_command_used = False
        
        # 1. Parse and apply Action if present and well-formed
        if action and len(action) > 0:
            if self.validator.is_valid(action):
                cmd_type = action[0]
                uav_id = int(action[1])
                
                # Find the UAV by ID
                target_uav = None
                for uav in self.uav_simulator.uavs:
                    if uav.uav_id == uav_id:
                        target_uav = uav
                        break
                        
                if target_uav is not None:
                    if cmd_type == "DEPLOY":
                        target_x = float(action[2])
                        target_y = float(action[3])
                        target_uav.set_waypoint(target_x, target_y)
                        target_uav.set_acceleration(1.0)
                    elif cmd_type == "RECALL":
                        target_uav.go_home()
            else:
                # Penalty for poorly formed message
                reward -= 2.0
                invalid_command_used = True
                
        # 2. Update Humans
        new_casualties = 0
        for agent in self.human_agents:
            agent.update()
            # If not already casualty, check if on burning cell
            if agent.activity_type != HumanAgentState.CASUALTY:
                if self.ca.grid[agent.x][agent.y].state == WildFireState.BURNING:
                    agent.mark_casualty()
                    new_casualties += 1
                    
        # Apply casualty penalty
        reward -= new_casualties * 10.0
        
        # 3. Update UAV simulator
        self.uav_simulator.update(delta_time=1.0, ca_grid=self.ca.grid, humans=self.human_agents)
        
        # 4. Periodically update Wildfire CA (every 10 environment steps)
        initial_burning = sum(1 for x in range(self.width) for y in range(self.height) if self.ca.grid[x][y].state == WildFireState.BURNING)
        if self.step_count % 10 == 0:
            self.ca.update()
        final_burning = sum(1 for x in range(self.width) for y in range(self.height) if self.ca.grid[x][y].state == WildFireState.BURNING)
        
        # Shape reward based on fire propagation
        fire_increase = final_burning - initial_burning
        if fire_increase > 0:
            reward -= fire_increase * 0.5
        elif fire_increase < 0:
            # Positive reward if fires were extinguished
            reward += abs(fire_increase) * 1.0
            
        # 5. Gather status messages for next state
        status_messages = []
        for uav in self.uav_simulator.uavs:
            status_messages.append(uav.latest_messages)
            
        # 6. Check termination (fire goes out)
        done = (final_burning == 0)
        
        # Calculate alive/casualty counts for info dictionary
        casualties = sum(1 for a in self.human_agents if a.activity_type == HumanAgentState.CASUALTY)
        alive = len(self.human_agents) - casualties
        
        info = {
            "step": self.step_count,
            "burning_cells": final_burning,
            "humans_alive": alive,
            "casualties": casualties,
            "invalid_command": invalid_command_used
        }
        
        return status_messages, reward, done, info
