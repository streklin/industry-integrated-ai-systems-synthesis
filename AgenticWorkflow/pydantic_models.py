from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

class CommandAction(str, Enum):
    DEPLOY = "DEPLOY"
    RECALL = "RECALL"

class UAVCommand(BaseModel):
    uav_id: int = Field(
        description="The unique ID of the target UAV (e.g., 0 for Recon, 1 for Extinguisher, 2 for Rescue)."
    )
    command: CommandAction = Field(
        description="The command action: DEPLOY to set/redirect to a new waypoint, or RECALL to return back to base."
    )
    target_x: Optional[float] = Field(
        default=None, 
        description="The target X coordinate in simulation space [0, 99]. Required if command is DEPLOY."
    )
    target_y: Optional[float] = Field(
        default=None, 
        description="The target Y coordinate in simulation space [0, 99]. Required if command is DEPLOY."
    )

class CommandCenterResponse(BaseModel):
    commands: List[UAVCommand] = Field(
        default_factory=list,
        description="The list of command instructions for each active UAV in flight."
    )