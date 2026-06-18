import math
import numpy as np
import random
from enum import Enum
from WildFireCA.WildFireCA import CACell, WildFireState
from HumanAgents.HumanAgent import HumanAgent, HumanAgentState

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
    RESCUE: A rescue UAV that is used to rescue humans.
    """
    BASE = 1
    RECON = 2
    EXTINGUISH = 3
    RESCUE = 4

class UAVBase:
    def __init__(self, uav_id: int = 0, turn_rate:float = 0.16, max_velocity:float = 1.0, width:float = 100, height:float = 100):
        """
        The UAVBase class is the base class for all UAV agents.
        It is responsible for:
            - Maintaining the state of the UAV
            - Executing the control inputs
        
        For simplicity (and compatibility with the rest of the simulation) we model the UAV as a point mass in
        2D Space.

        Args:
            uav_id: Unique identifier for the UAV.
            width: The width of the work area.
            height: the height of the work area.

        If a drone leaves its work area (less than 0 or greater than width/height) it will be considered lost.
        """
        self.uav_id = uav_id
        self.recalled = False

        self.velocity = 0.0
        self.x = 0.0
        self.y = 0.0
        self.bank_angle = 0.0

        self.acceleration = 0.0
        self.max_velocity = max_velocity

        self.state = UAVState.HANGER

        self.width = width
        self.height = height

        self.waypoint_x = 0.0
        self.waypoint_y = 0.0

        self.home_base = (20.0, 20.0)

        self.fuel = 500.0

        # determine how large a region the UAV can detect fires and people in.
        # for extinguish operations, any fires in this nbd are extinguished.
        self.detection_range = 1.0

        self.turn_rate = turn_rate
        self.uav_type:UAVType = UAVType.BASE
        self.latest_messages = [self.x, self.y, self.fuel, self.state, None, None, False, False]
        
        # RL / SVM specific attributes
        self.svm_model = None
        self.exploration_rate = 0.0
    
    def set_svm_model(self, model):
        """
        Load a trained SVM classifier policy model.
        """
        self.svm_model = model

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
        for i in range(int(self.x - self.detection_range), int(self.x + self.detection_range) + 1):
            for j in range(int(self.y - self.detection_range), int(self.y + self.detection_range) + 1):
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

    def get_grid_crop_features(self, ca_grid: list[list[CACell]], humans: list[HumanAgent]) -> np.ndarray:
        """
        Extract local grid crop around the UAV as binary feature vectors.
        Returns a 1D vector of shape (2 * (2 * R + 1)^2,) where R = int(detection_range).
        """
        R = int(self.detection_range)
        size = 2 * R + 1
        fire_channel = np.zeros((size, size))
        human_channel = np.zeros((size, size))
        
        cx, cy = int(self.x), int(self.y)
        
        active_humans = {}
        for h in humans:
            if h.activity_type not in (HumanAgentState.CASUALTY, HumanAgentState.RESCUED):
                active_humans[(h.x, h.y)] = True
                
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                gx, gy = cx + dx, cy + dy
                row = dx + R
                col = dy + R
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    if ca_grid[gx][gy].state == WildFireState.BURNING:
                        fire_channel[row, col] = 1.0
                    if (gx, gy) in active_humans:
                        human_channel[row, col] = 1.0
                        
        return np.concatenate([fire_channel.flatten(), human_channel.flatten()])

    def _get_boundary_override_action(self) -> int | None:
        """
        Safety layer applied before any policy action is returned.
        If the UAV is within WALL_MARGIN cells of any boundary, compute the
        steering action that turns it back toward the interior of the grid.
        Returns the corrective action (0/1/2) if triggered, or None if the
        UAV is safely away from all walls.
        """
        WALL_MARGIN = 5  # cells
        near_left   = self.x < WALL_MARGIN
        near_right  = self.x > self.width  - 1 - WALL_MARGIN
        near_top    = self.y < WALL_MARGIN
        near_bottom = self.y > self.height - 1 - WALL_MARGIN

        if not (near_left or near_right or near_top or near_bottom):
            return None  # well inside the grid — no correction needed

        # Steer toward a safe inset point well away from the nearest wall
        safe_x = max(WALL_MARGIN * 2, min(self.width  - 1 - WALL_MARGIN * 2, self.width  / 2.0))
        safe_y = max(WALL_MARGIN * 2, min(self.height - 1 - WALL_MARGIN * 2, self.height / 2.0))

        dx = safe_x - self.x
        dy = safe_y - self.y
        target_angle = math.atan2(dy, dx)
        diff = (target_angle - self.bank_angle + math.pi) % (2 * math.pi) - math.pi

        if diff > self.turn_rate / 2:
            return 0  # Turn Left
        elif diff < -self.turn_rate / 2:
            return 1  # Turn Right
        else:
            return 2  # Go Straight

    def select_rl_action(self, ca_grid: list[list[CACell]], humans: list[HumanAgent]) -> int:
        """
        Select action (0: turn left, 1: turn right, 2: go straight) based on policy or exploration.
        Boundary repulsion is always applied first as a safety override.
        """
        # Safety override: wall repulsion takes priority over all policies
        wall_action = self._get_boundary_override_action()
        if wall_action is not None:
            return wall_action

        if random.random() < self.exploration_rate:
            return random.choice([0, 1, 2])

        if self.svm_model is not None:
            features = self.get_grid_crop_features(ca_grid, humans)
            try:
                action = int(self.svm_model.predict([features])[0])
                return action
            except Exception:
                return 2
        else:
            return self.get_heuristic_action(ca_grid, humans)

    def apply_rl_action(self, action: int):
        """
        Apply the discrete steering action to update the bank angle.
        """
        if action == 0:  # Turn Left
            self.bank_angle += self.turn_rate
        elif action == 1:  # Turn Right
            self.bank_angle -= self.turn_rate
        # Action 2: Go Straight (do nothing)
            
        # Keep bank_angle wrapped to [-pi, pi]
        self.bank_angle = (self.bank_angle + math.pi) % (2 * math.pi) - math.pi

    def get_heuristic_action(self, ca_grid: list[list[CACell]], humans: list[HumanAgent]) -> int:
        """
        Simple deterministic fallback policy to turn towards target of interest.
        Used for fallback and expert demonstrations.
        """
        target_x, target_y = None, None
        
        if self.uav_type in (UAVType.RECON, UAVType.EXTINGUISH):
            # Try to find nearest fire in detection range first
            min_dist = float('inf')
            detected = self._get_detected_cells(ca_grid)
            for cell in detected:
                if ca_grid[cell[0]][cell[1]].state == WildFireState.BURNING:
                    dist = math.hypot(cell[0] - self.x, cell[1] - self.y)
                    if dist < min_dist:
                        min_dist = dist
                        target_x, target_y = cell[0] + 0.5, cell[1] + 0.5
            
            # If no fire in detection range, search grid for nearest fire
            if target_x is None:
                min_dist = float('inf')
                for x in range(self.width):
                    for y in range(self.height):
                        if ca_grid[x][y].state == WildFireState.BURNING:
                            dist = math.hypot(x - self.x, y - self.y)
                            if dist < min_dist:
                                min_dist = dist
                                target_x, target_y = x + 0.5, y + 0.5
                                
        elif self.uav_type == UAVType.RESCUE:
            # Target is the nearest active human
            min_dist = float('inf')
            for human in humans:
                if human.activity_type not in (HumanAgentState.CASUALTY, HumanAgentState.RESCUED):
                    dist = math.hypot(human.x - self.x, human.y - self.y)
                    if dist < min_dist:
                        min_dist = dist
                        target_x, target_y = human.x, human.y

        # Fallback: no primary target found → steer toward grid centre so the
        # UAV drifts back into a useful patrol area.
        if target_x is None:
            target_x = self.width / 2.0
            target_y = self.height / 2.0

        # Note: boundary repulsion is handled upstream in select_rl_action
        # via _get_boundary_override_action() so we don't duplicate it here.
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        
        diff = (target_angle - self.bank_angle + math.pi) % (2 * math.pi) - math.pi
        
        if diff > self.turn_rate / 2:
            return 0  # Turn Left
        elif diff < -self.turn_rate / 2:
            return 1  # Turn Right
        else:
            return 2  # Go Straight

    def update(self, delta_time: float, ca_grid:list[list[CACell]], humans: list[HumanAgent]) -> list[str]:
        """
        Update the state of the UAV.
        """
        if self.state == UAVState.HANGER:
            self.velocity = 0.0
            self.acceleration = 0.0
            self.x = self.home_base[0]
            self.y = self.home_base[1]
            self.recalled = False
            messages = [self.x, self.y, self.fuel, self.uav_type, None, None, False, False]
            self.latest_messages = [messages]
            return messages

        # Run RL action selection only when cruising at destination
        if self.state == UAVState.CRUISING:
            action = self.select_rl_action(ca_grid, humans)
            self.apply_rl_action(action)

        self.velocity = min(self.velocity + self.acceleration * delta_time, self.max_velocity)
        
        self.x += self.velocity * delta_time * np.cos(self.bank_angle)
        self.y += self.velocity * delta_time * np.sin(self.bank_angle)

        # Clamp boundaries to keep drone in workspace
        self.x = max(0.0, min(self.x, self.width - 1))
        self.y = max(0.0, min(self.y, self.height - 1))

        self.fuel -= 1

        if self.fuel <= 0 and self.state != UAVState.RETURNING and self.state != UAVState.HANGER:
            self.go_home()

        # Update waypoint travel only if traversing
        if self.state in (UAVState.TRAVELLING, UAVState.RETURNING):
            self._update_travelling(delta_time=delta_time)

        messages = []
        messages.append(self.x)
        messages.append(self.y)
        messages.append(self.fuel)
        messages.append(self.uav_type)
        
        if self.state == UAVState.CRUISING:
            messages.append(self.waypoint_x)
            messages.append(self.waypoint_y)
        else:
            messages.append(None)
            messages.append(None)
        

        detected_cells = self._get_detected_cells(ca_grid)
        detected_humans = self._get_detected_humans(humans)

        has_fire = False
        has_human = len(detected_humans) > 0
        
        for cell in detected_cells:
            if ca_grid[cell[0]][cell[1]].state == WildFireState.BURNING:
                has_fire = True
                break

        messages.append(has_fire)
        messages.append(has_human)
        
        self.latest_messages.append(messages)
        return messages
        
    def set_acceleration(self, acceleration: float):
        """
        Set the acceleration of the UAV.
        """
        self.acceleration = acceleration

    def set_waypoint(self, waypoint_x: float, waypoint_y: float):
        """
        Set the waypoint of the UAV.
        """
        is_home = (waypoint_x == self.home_base[0] and waypoint_y == self.home_base[1])
        if (self.state == UAVState.RETURNING or self.recalled) and not is_home:
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
            if self.waypoint_x == self.home_base[0] and self.waypoint_y == self.home_base[1]:
                self.fuel = 500.0
                self.state = UAVState.HANGER
                self.recalled = False
            return

        angle = math.atan2(dy, dx)
        diff = (angle - self.bank_angle + math.pi) % (2 * math.pi) - math.pi
        
        if diff > self.turn_rate:
            self.bank_angle += self.turn_rate
        elif diff < -self.turn_rate:
            self.bank_angle -= self.turn_rate
        else:
            self.bank_angle = angle
            
        self.bank_angle = (self.bank_angle + math.pi) % (2 * math.pi) - math.pi

    def get_report(self) -> dict:
        """
        Get status report of the UAV.
        """

        report = []
        for msg in self.latest_messages:
            report_entry = {
                "id": self.uav_id,
                "position": [msg[0], msg[1]],
                "uav_type": msg[3],
                "waypoint": [msg[4], msg[5]],
                "sensor_status": {
                    "sees_fire": msg[6],
                    "sees_human": msg[7],
                    "fuel": msg[2]
                }
            }
            report.append(report_entry)
        
        self.latest_messages.clear()
        return report


class ReconUAV(UAVBase):
    """
    Represents small, cheap, fast UAV's that can be deployed to monitor the fire front.
    They report back to the control center with an update.
    """
    def __init__(self, uav_id: int = 0, turn_rate:float = math.pi / 10, max_velocity:float = 2.0, width:float = 100, height:float = 100):
        super().__init__(uav_id, turn_rate, max_velocity, width, height)
        self.uav_type = UAVType.RECON
        self.detection_range = 5

    def update(self, delta_time: float, ca_grid: list[list[CACell]], humans: list[HumanAgent]) -> list[str]:
        messages = super().update(delta_time, ca_grid, humans)
        return messages


class FireControlUAV(UAVBase):
    """
    Represents the large, expensive UAV's that are deployed to fight fires.
    """
    def __init__(self, uav_id: int = 0, turn_rate:float = math.pi / 10, max_velocity:float = 1.0, width:float = 100, height:float = 100):
        super().__init__(uav_id, turn_rate, max_velocity, width, height)
        self.uav_type = UAVType.EXTINGUISH
        self.detection_range = 3.0
        self.velocity = 0.5

    def update(self, delta_time: float, ca_grid: list[list[CACell]], humans: list[HumanAgent]) -> list[str]:
        messages = super().update(delta_time, ca_grid, humans)

        # Extinguish all visible fire in range
        detected_cells = self._get_detected_cells(ca_grid)
        extinguished_any = False
        for cell in detected_cells:
            if ca_grid[cell[0]][cell[1]].state == WildFireState.BURNING:
                ca_grid[cell[0]][cell[1]].extinguish()
                extinguished_any = True
                
        if extinguished_any:
            print(f"[FireControlUAV] Extinguished fire near ({self.x:.1f}, {self.y:.1f})")
            
        return messages


class RescueUAV(UAVBase):
    """
    Represents the large UAV's that are deployed to rescue humans.
    """
    def __init__(self, uav_id: int = 0, turn_rate:float = math.pi / 10, max_velocity:float = 1.0, width:float = 100, height:float = 100):
        super().__init__(uav_id, turn_rate, max_velocity, width, height)
        self.uav_type = UAVType.RESCUE
        self.detection_range = 5
        self.velocity = 0.5

    def update(self, delta_time: float, ca_grid: list[list[CACell]], humans: list[HumanAgent]) -> list[str]:
        messages = super().update(delta_time, ca_grid, humans)

        # Automatically rescue any human in the same cell
        for human in humans:
            if human.activity_type not in (HumanAgentState.CASUALTY, HumanAgentState.RESCUED):
                if math.hypot(human.x - self.x, human.y - self.y) <= self.detection_range:
                    human.activity_type = HumanAgentState.RESCUED
                    print(f"[RescueUAV] Rescued human at ({human.x}, {human.y})")
                    
        return messages