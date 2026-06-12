import math
import numpy as np
from enum import Enum
from WildFireCA.WildFireCA import CACell, WildFireState
from HumanAgents.HumanAgent import HumanAgent

class UAVState(Enum):
    """
    Represents the overall state of the UAV.
    HANGER: UAV is in the hanger and ready to deployed.
    CRUISING: UAV is cruising at a constant velocity
    LANDING: UAV is returning to the landing field.
    LOST: UAV is out of contact range of the control center.
    TRAVELLING: UAV is travelling to a provided waypoint.
    """
    HANGER = 1
    CRUISING = 2
    LANDING = 3
    LOST = 4
    TRAVELLING = 5
    RETURNING = 6

class UAVType(Enum):
    """
    Represents the type of UAV.
    BASE: A base UAV that is used for control and communication.
    RECON: A reconnaissance UAV that is used to detect fires and people.
    EXTINGUISH: An extinguish UAV that is used to extinguish fires.
    """
    BASE = 1
    RECON = 2
    EXTINGUISH = 3

class UAVBase:
    def __init__(self, turn_rate:float = 0.16, max_velocity:float = 1.0, width:float = 100, height:float = 100):
        """
        The UAVBase class is the base class for all UAV agents.
        It is responsible for:
            - Maintaining the state of the UAV
            - Executing the control inputs
        
        For simplicity (and compatibility with the rest of the simulation) we model the UAV as a point mass in
        2D Space.

        Args:
            width: The width of the work area.
            height: the hieght of the work area.

        If a drone leaves it works area (less than 0 or greater than width/height) it will be considered lost.
        """
        self.velocity = 0.0
        self.x = 0
        self.y = 0
        self.bank_angle = 0.0

        self.acceleration = 0.0
        self.max_velocity = max_velocity

        self.state = UAVState.HANGER

        self.width = width
        self.height = height

        self.waypoint_x = 0
        self.waypoint_y = 0

        self.home_base = (20, 20)

        self.fuel = 100.0

        # determine how large a region the UAV can detect fires and people in.
        # for extinguish operations, any fires in this nbd are extinguished.
        # not a perfect representation but good enough for representation purposes.
        self.detection_range = 1.0

        self.turn_rate = turn_rate
        self.uav_type:UAVType = UAVType.BASE
        self.latest_messages = [self.x, self.y, self.fuel, "NO FIRE", "NO HUMAN"]
    
    def go_home(self):
        """
        Set the waypoint of the UAV to its home base.
        """
        self.set_waypoint(self.home_base[0], self.home_base[1])
        self.state = UAVState.RETURNING

    def _get_detected_cells(self, ca_grid:list[list[CACell]]) -> list[tuple[int, int]]:
        """
        Get the cells that are in the detection range of the UAV.
        
        Args:
            ca_grid: The cellular automata grid.
        Returns:
            A list of tuples representing the coordinates of the detected cells.
        """
        detected_cells = []
        for i in range(int(self.x - self.detection_range), int(self.x + self.detection_range)):
            for j in range(int(self.y - self.detection_range), int(self.y + self.detection_range)):
                if 0 <= i < self.width and 0 <= j < self.height:
                    detected_cells.append((i, j))
        return detected_cells

    def _get_detected_humans(self, humans: list[HumanAgent]) -> list[HumanAgent]:
        """
        Get the humans that are in the detection range of the UAV.
        
        Args:
            humans: The list of human agents.
        Returns:
            A list of human agents that are in the detection range of the UAV.
        """
        detected_humans = []
        for human in humans:
            if self.detection_range >= math.hypot(human.x - self.x, human.y - self.y):
                detected_humans.append(human)
        return detected_humans

    def update(self, delta_time: float, ca_grid:list[list[CACell]], humans: list[HumanAgent]) -> list[str]:
        """
        Update the state of the UAV.
        
        Args:
            delta_time: The time step to update the UAV by.
        Returns:
            a list of messages to be given to dispatch. defaults to an empty array.
        """
        if self.state == UAVState.HANGER:
            self.velocity = 0.0
            self.acceleration = 0.0
            self.x = self.home_base[0]
            self.y = self.home_base[1]
            messages = [self.x, self.y, self.fuel, "NO FIRE", "NO HUMAN"]
            self.latest_messages = messages
            return messages

        self.velocity = min(self.velocity + self.acceleration * delta_time, self.max_velocity)
        
        self.x += self.velocity * delta_time * np.cos(self.bank_angle)
        self.y += self.velocity * delta_time * np.sin(self.bank_angle)

        self.fuel -= 1

        if self.fuel <= 0:
            self.go_home()

        self._update_travelling(delta_time=delta_time)

        messages = []
        messages.append(self.x)
        messages.append(self.y)
        messages.append(self.fuel)

        detected_cells = self._get_detected_cells(ca_grid)
        detected_humans = self._get_detected_humans(humans)

        has_fire = False
        has_human = len(detected_humans) > 0
        
        # check if there is fire or a human in the detection range
        for cell in detected_cells:
            if ca_grid[cell[0]][cell[1]].state == WildFireState.BURNING:
                has_fire = True
                continue

        if has_fire:
            messages.append("FIRE")
        else:
            messages.append("NO FIRE")
        
        if has_human:
            messages.append("HUMAN")
        else:
            messages.append("NO HUMAN")
        
        self.latest_messages = messages
        return messages
        
    def set_acceleration(self, acceleration: float):
        """
        Set the acceleration of the UAV.
        
        Args:
            acceleration: The acceleration to set the UAV to.
        """
        self.acceleration = acceleration


    def set_waypoint(self, waypoint_x: float, waypoint_y: float):
        """
        Set the waypoint of the UAV.
        
        Args:
            waypoint_x: The x coordinate of the waypoint.
            waypoint_y: The y coordinate of the waypoint.
        """
        # new waypoints cannot be set on UAV's that are returning for refuel/replenishment
        if self.state == UAVState.RETURNING:
            return

        self.waypoint_x = waypoint_x
        self.waypoint_y = waypoint_y
        self.state = UAVState.TRAVELLING

    def _update_travelling(self, delta_time: float):
        """
        Sets the bank angle of the UAV to turn towards the waypoint.
        """
        dx = self.waypoint_x - self.x
        dy = self.waypoint_y - self.y
        dist = math.hypot(dx, dy)
        if dist < self.velocity * delta_time:
            self.state = UAVState.CRUISING
            # If we returned home base, refuel and reset state
            if self.waypoint_x == self.home_base[0] and self.waypoint_y == self.home_base[1]:
                self.fuel = 100.0
                self.state = UAVState.HANGER
            return

        # update the angular velocity to turn towards the waypoint
        angle = math.atan2(dy, dx)
        
        # Calculate shortest angular difference wrapped to [-pi, pi]
        diff = (angle - self.bank_angle + math.pi) % (2 * math.pi) - math.pi
        
        if diff > self.turn_rate:
            self.bank_angle += self.turn_rate
        elif diff < -self.turn_rate:
            self.bank_angle -= self.turn_rate
        else:
            self.bank_angle = angle
            
        # Keep bank_angle wrapped to [-pi, pi]
        self.bank_angle = (self.bank_angle + math.pi) % (2 * math.pi) - math.pi
        

class ReconUAV(UAVBase):
    """
    Represents small, cheap, fast UAV's that can be deployed to monitor the fire front.
    They report back to the control center with an update:
    - Current Position
    - Sees Fire - this is true if there is fire in the ReconUAV's detection range.
    - Sees People - this is true if there is a human agent in the ReconUAV's detection range.
    """
    def __init__(self, turn_rate:float = math.pi / 10, max_velocity:float = 2.0, width:float = 100, height:float = 100):
        super().__init__(turn_rate, max_velocity, width, height)
        self.uav_type = UAVType.RECON

    def update(self, delta_time: float, ca_grid: list[list[CACell]], humans: list[HumanAgent]) -> list[str]:
        """
        Update the state of the UAV.
        
        Args:
            delta_time: The time step to update the UAV by.
        Returns:
            a list of messages to be given to dispatch. defaults to an empty array.
        """
        messages = super().update(delta_time, ca_grid, humans)
        if self.state == UAVState.TRAVELLING or self.state == UAVState.RETURNING or self.state == UAVState.HANGER:
            return messages

        # if we don't see fire or a human we return to the last known waypoint
        # otherwise, we update the waypoint to the position of the human or fire
        if "FIRE" not in messages and "HUMAN" not in messages:
            self.state = UAVState.TRAVELLING # should head back to the last set waypoint - NOT GO HOME
            return messages

        # keep around where we are
        if "FIRE" in messages or "HUMAN" in messages:
            self.set_waypoint(messages[0], messages[1])
        
        return messages
        

class FireControlUAV(UAVBase):
    """
    Represents the large, expensive UAV's that are deployed to fight fires.
    They report back to the control center with an update:
    - Current Position
    
    Once the agent reaches target location it will search for fire.
    If it sees fire, it will deploy its suppression system to supress the fire.
    Once deployed, the Agent will return to the runway.
    """
    def __init__(self, turn_rate:float = math.pi / 10, max_velocity:float = 1.0, width:float = 100, height:float = 100):
        super().__init__(turn_rate, max_velocity, width, height)
        self.uav_type = UAVType.EXTINGUISH

    def update(self, delta_time: float, ca_grid: list[list[CACell]], humans: list[HumanAgent]) -> list[str]:
        """
        Update the state of the UAV.
        """
        messages = super().update(delta_time, ca_grid, humans)
        if self.state == UAVState.TRAVELLING or self.state == UAVState.RETURNING or self.state == UAVState.HANGER:
            return messages

        # if we don't see fire, we return to the last known waypoint
        # otherwise, we update the waypoint to the position of the fire
        if "FIRE" not in messages:
            self.state = UAVState.TRAVELLING # should head back to the last set waypoint - NOT GO HOME
            return messages

        print(f"UAV EXTINGUISHING FIRE at ({messages[0]}, {messages[1]})")

        # extinguish all visible fire and return home
        detected_cells = self._get_detected_cells(ca_grid)

        for cell in detected_cells:
            ca_grid[cell[0]][cell[1]].extinguish()
            
        return messages