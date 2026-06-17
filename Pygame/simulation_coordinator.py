import os
import sys
import json
import io
import math
import pygame
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# Ensure the parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AgenticWorkflow")))

from FirePrediction.UNet import UNet
from AgenticWorkflow.agentic_workflow import CommandCenterAgent

# ------------------------------------------------------------------
# Coordinate-space constants
# Agents see 500×500 PNG images; the simulation grid is GRID_SIZE×GRID_SIZE.
# When the LLM returns target coordinates they should be in grid-space
# [0, GRID_SIZE-1], but if the model accidentally returns image-pixel
# coordinates (0–4999) we normalise them transparently.
# ------------------------------------------------------------------
IMAGE_SIZE = 500   # pixels sent to CommandCenterAgent
GRID_SIZE  = 100   # simulation grid cells (assumed square)

class SimulationCoordinator:
    """
    Coordinates simulation snapshots, model inference, telemetry gathering, 
    and command delegation with the CommandCenterAgent.
    """
    def __init__(self, n_steps: int = 20):
        self.n_steps = n_steps
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load UNet Risk Predictor Model
        self.unet = UNet(in_channels=3, num_classes=1).to(self.device)
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FirePrediction", "models", "fire_predictor_lrg.pth"))
        if not os.path.exists(model_path):
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FirePrediction", "models", "fire_predictor.pth"))
            
        if os.path.exists(model_path):
            try:
                self.unet.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.unet.eval()
                print(f"[Coordinator] Loaded UNet model from {model_path} on {self.device}")
            except Exception as e:
                print(f"[Coordinator] Error loading model state dict: {e}")
        else:
            print(f"[Coordinator] Warning: UNet model weights not found at {model_path}")

        # Load transforms matching training pipeline (transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)))
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Initialize CommandCenter agentic entry point
        self.command_center = CommandCenterAgent()

    def on_step(self, step: int, visualizer, uavs: list, humans: list) -> None:
        """
        Executed in the simulation loop. Gathers snapshots, telemetry, and updates dispatch at N-step intervals.
        """
        if step % self.n_steps != 0:
            return

        print(f"\n--- Coordinator Run (Step {step}) ---")

        # 1. Capture Pygame Screen & scale it down to 500x500 pixels for CommandCenter
        try:
            img_str = pygame.image.tostring(visualizer.screen, 'RGB')
            pil_image = Image.frombytes('RGB', visualizer.screen.get_size(), img_str)
            
            scaled_pil = pil_image.resize((500, 500))
            sat_buffer = io.BytesIO()
            scaled_pil.save(sat_buffer, format="PNG")
            satellite_image_bytes = sat_buffer.getvalue()
        except Exception as e:
            print(f"[Coordinator] Failed to capture satellite image: {e}")
            return

        # 2. Perform UNet prediction to get risk mask
        risk_mask_bytes = b""
        try:
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.unet(input_tensor)
                probs = torch.sigmoid(logits)
                pred_mask = (probs > 0.5).float().squeeze(0).squeeze(0).cpu().numpy() # shape (512, 512)
                
            mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
            scaled_mask_pil = mask_pil.resize((500, 500))
            mask_buffer = io.BytesIO()
            scaled_mask_pil.save(mask_buffer, format="PNG")
            risk_mask_bytes = mask_buffer.getvalue()
        except Exception as e:
            print(f"[Coordinator] Model inference failed: {e}")
            # Fallback to a blank image risk mask
            blank_mask = Image.new("L", (500, 500), color=0)
            blank_buffer = io.BytesIO()
            blank_mask.save(blank_buffer, format="PNG")
            risk_mask_bytes = blank_buffer.getvalue()

        # 3. Gather UAV Reports
        reports = [uav.get_report() for uav in uavs]
        print(f"[Coordinator] Telemetry reports collected: {reports}")

        # 4. Invoke CommandCenter Agentic Update
        print("[Coordinator] Querying CommandCenterAgent...")
        commands = self.command_center.update(
            uav_messages=reports,
            satellite_image=satellite_image_bytes,
            risk_image=risk_mask_bytes
        )

        # 5. Delegate commands back to the UAVs
        if not commands:
            print("[Coordinator] No commands received.")
            return

        # Parse commands if returned as a JSON string
        if isinstance(commands, str):
            try:
                commands = json.loads(commands)
            except Exception as e:
                print(f"[Coordinator] Failed to parse command string as JSON: {e}")
                return

        # Map command list to actual UAV agents
        if isinstance(commands, list):
            print(f"[Coordinator] Executing commands: {commands}")
            uav_map = {uav.uav_id: uav for uav in uavs}
            
            for cmd in commands:
                uav_id = cmd.get("uav_id")
                action = cmd.get("command")
                
                if uav_id not in uav_map:
                    print(f"[Coordinator] Warning: Command targets unknown uav_id {uav_id}")
                    continue
                    
                uav = uav_map[uav_id]
                
                if action == "RECALL":
                    uav.recalled = True
                    uav.go_home()
                    print(f"[Coordinator] UAV {uav_id} RECALLED. Heading base.")
                elif action == "DEPLOY":
                    target_x = cmd.get("target_x")
                    target_y = cmd.get("target_y")
                    if target_x is not None and target_y is not None:
                        # Normalise coordinates: the LLM should return grid-space [0, GRID_SIZE-1].
                        # If it accidentally returns image-pixel coords (values > GRID_SIZE),
                        # scale them back to grid space.
                        target_x = float(target_x)
                        target_y = float(target_y)
                        if target_x >= GRID_SIZE or target_y >= GRID_SIZE:
                            target_x = (target_x / IMAGE_SIZE) * GRID_SIZE
                            target_y = (target_y / IMAGE_SIZE) * GRID_SIZE
                            print(f"[Coordinator] Rescaled image-space coords to grid-space: ({target_x:.1f}, {target_y:.1f})")
                        # Clamp to valid grid bounds
                        target_x = max(0.0, min(float(GRID_SIZE - 1), target_x))
                        target_y = max(0.0, min(float(GRID_SIZE - 1), target_y))
                        uav.recalled = False
                        uav.set_waypoint(target_x, target_y)
                        uav.set_acceleration(1.0)
                        print(f"[Coordinator] UAV {uav_id} DEPLOYED/REDIRECTED to ({target_x:.1f}, {target_y:.1f}).")
                    else:
                        print(f"[Coordinator] Warning: DEPLOY command for UAV {uav_id} missing coordinates.")
