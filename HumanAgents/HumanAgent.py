from enum import Enum
import random

class HumanAgentState(Enum):
    CAMPING = 0
    HIKING = 1
    CASUALTY = 2

class HumanAgent:
    """
    A human agent is a simple stochastic simulation. They'll hike and camp in the wilderness until
    either rescued or killed by the fire.
    """
    def __init__(self, x, y, max_x, max_y, activity_type=None):
        self.x = x
        self.y = y
        self.max_x = max_x
        self.max_y = max_y
        self.activity_type = activity_type
        self.hiker_transition_prob = 0.25

    def _can_update(self) -> bool:
        """
        Check if the agent can be updated.
        """
        return self.activity_type != HumanAgentState.CASUALTY

    def _update_hiker(self):
        """
        Update the state of the hiker.
        """
        # pick a random direction
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])

        # move the human in that direction
        self.x += dx
        self.y += dy

        # clamp human to the map
        self.x = max(0, min(self.x, self.max_x))
        self.y = max(0, min(self.y, self.max_y))

        if random.random() < self.hiker_transition_prob:
            self.activity_type = HumanAgentState.CAMPING

    def _update_camper(self):
        """
        Update the state of the camper.
        """
        if random.random() < self.hiker_transition_prob:
            self.activity_type = HumanAgentState.HIKING

    def update(self):
        """
        Update the state of the human agent.
        """
        if not self._can_update():
            return

        if self.activity_type == HumanAgentState.HIKING:
            self._update_hiker()
        elif self.activity_type == HumanAgentState.CAMPING:
            self._update_camper()
        
    def mark_casualty(self):
        """
        We will assume that any human caught in a burning cell is effectively a casualty.
        """
        self.activity_type = HumanAgentState.CASUALTY