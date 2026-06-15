import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import pygame

# Set dummy video driver for headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Ensure the parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AgenticWorkflow")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from WildFireCA.WildFireCA import WildfireCA
from HumanAgents.HumanAgent import HumanAgent
from UAVAgents.UAVBase import ReconUAV, FireControlUAV, RescueUAV, UAVState, UAVType
from visualizer import WildfireVisualizer
from simulation_coordinator import SimulationCoordinator

class TestSimulationCoordinator(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.width, self.height = 100, 100
        self.ca = WildfireCA(width=self.width, height=self.height)
        self.ca.regenerate(seed=42)
        
        self.humans = [HumanAgent(x=30, y=30, max_x=99, max_y=99)]
        
        # Instantiate UAVs with unique IDs
        self.recon = ReconUAV(uav_id=0, width=self.width, height=self.height)
        self.extinguish = FireControlUAV(uav_id=1, width=self.width, height=self.height)
        self.rescue = RescueUAV(uav_id=2, width=self.width, height=self.height)
        
        # Set initial cruising/Hanger states
        self.recon.state = UAVState.CRUISING
        self.recon.x, self.recon.y = 40.0, 40.0
        self.extinguish.state = UAVState.CRUISING
        self.extinguish.x, self.extinguish.y = 50.0, 50.0
        self.rescue.state = UAVState.HANGER
        
        self.uavs = [self.recon, self.extinguish, self.rescue]
        
        # Instantiate visualizer
        self.visualizer = WildfireVisualizer(
            self.ca, 
            human_agents=self.humans, 
            uavs=self.uavs, 
            cell_size=10, 
            window_title="Test Visualizer"
        )
        self.visualizer.refresh()
        
        # Instantiate coordinator
        self.coordinator = SimulationCoordinator(n_steps=5)

    def tearDown(self):
        pygame.quit()

    @patch('AgenticWorkflow.agentic_workflow.CommandCenterAgent.update')
    def test_on_step_triggers_and_delegates(self, mock_update):
        # Mock CommandCenterAgent to return a deploy command for uav_id 0 and recall for uav_id 1
        mock_commands = [
            {"uav_id": 0, "command": "DEPLOY", "target_x": 75.0, "target_y": 80.0},
            {"uav_id": 1, "command": "RECALL"}
        ]
        mock_update.return_value = mock_commands

        # Step 5 triggers the coordinator
        self.coordinator.on_step(5, self.visualizer, self.uavs, self.humans)

        # Assert update was called with telemetry, satellite image, and risk mask bytes
        self.assertTrue(mock_update.called)
        args, kwargs = mock_update.call_args
        
        # Check image formats
        uav_messages = kwargs.get("uav_messages")
        satellite_image = kwargs.get("satellite_image")
        risk_image = kwargs.get("risk_image")
        
        self.assertEqual(len(uav_messages), 3)
        self.assertEqual(uav_messages[0]["id"], 0)
        self.assertEqual(uav_messages[1]["id"], 1)
        self.assertEqual(uav_messages[2]["id"], 2)
        
        self.assertGreater(len(satellite_image), 0)
        self.assertGreater(len(risk_image), 0)
        
        # Assert commands were executed on the UAV agents
        # 1. UAV 0 should be redirected to (75, 80)
        self.assertEqual(self.recon.waypoint_x, 75.0)
        self.assertEqual(self.recon.waypoint_y, 80.0)
        self.assertEqual(self.recon.state, UAVState.TRAVELLING)
        self.assertFalse(self.recon.recalled)

        # 2. UAV 1 should be recalled
        self.assertTrue(self.extinguish.recalled)
        self.assertEqual(self.extinguish.waypoint_x, self.extinguish.home_base[0])
        self.assertEqual(self.extinguish.waypoint_y, self.extinguish.home_base[1])
        self.assertEqual(self.extinguish.state, UAVState.RETURNING)

        print("[Test Success] Coordinator executed steps, generated maps/masks, and updated UAV state successfully.")

if __name__ == "__main__":
    unittest.main()
