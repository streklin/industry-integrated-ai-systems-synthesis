from WildFireCA.WildFireCA import CACell, WildFireState
from HumanAgents.HumanAgent import HumanAgent
from UAVAgents.UAVBase import UAVBase, UAVType, UAVState

class UAVSimulator:
    """
    A module to update the state of multiple UAVs.
    """
    def __init__(self, uavs: list[UAVBase]):
        self.uavs = uavs
        self.fire_control_uav_count = 0
        self.fire_recon_uav_count = 0
        self.fire_extinguish_uav_count = 0

    def add_uav(self, uav: UAVBase):
        """
        Add a UAV to the simulator.
        """
        self.uavs.append(uav)

        if uav.uav_type == UAVType.BASE:
            self.fire_control_uav_count += 1
        elif uav.uav_type == UAVType.RECON:
            self.fire_recon_uav_count += 1
        elif uav.uav_type == UAVType.EXTINGUISH:
            self.fire_extinguish_uav_count += 1

    def update(self, delta_time: float, ca_grid: list[list[CACell]], humans: list[HumanAgent]) -> None:
        """
        Update the state of the UAVs.
        """
        for uav in self.uavs:
            uav.update(delta_time, ca_grid, humans)


        