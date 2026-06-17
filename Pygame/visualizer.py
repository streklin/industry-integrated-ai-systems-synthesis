import os
import sys
import pygame

import imageio
from IPython.display import Video

# Ensure the parent directory is in sys.path so we can import WildFireCA and HumanAgents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from WildFireCA.WildFireCA import WildFireState, WildfireCA
from HumanAgents.HumanAgent import HumanAgent, HumanAgentState
from UAVAgents.UAVBase import UAVType, UAVState


class WildfireVisualizer:
    """
    Handles visualizing the current state of a WildfireCA simulation grid and Human Agents using Pygame.
    """
    def __init__(self, ca: WildfireCA, human_agents=None, uavs=None, cell_size: int = 10, window_title: str = "Wildfire CA Simulation", sim_width: int = 100, sim_height: int = 100, image_size: int = 500, is_recording: bool = False):
        """
        Initialize the visualizer.
        
        Args:
            ca: The WildfireCA simulation instance to visualize.
            human_agents: Optional list of HumanAgent instances.
            uavs: Optional list of UAV instances.
            cell_size: Size in pixels of each grid cell.
            window_title: Window title.
        """
        self.ca = ca
        self.human_agents = human_agents if human_agents is not None else []
        self.uavs = uavs if uavs is not None else []
        self.cell_size = cell_size
        # Coordinate scaling: agents reason in image-pixel space (image_size × image_size).
        # This scale converts an image-space coordinate back to a pixel position on screen.
        self.img_to_screen = (sim_width * cell_size) / image_size
        self.width = ca.width * cell_size
        self.height = ca.height * cell_size
        
        # Color coding:
        # - Ash is grey, Empty is black.
        # - Trees/Shrubs/Grass various shades of green.
        # - Water Blue, Housing orange, urban white, fire red.
        self.colors = {
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
        }
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(window_title)
        self.clock = pygame.time.Clock()
        self.fps = 10  # Default simulation speed (steps per second)

        # recording
        self.is_recording = is_recording
        self.frames = []

    def _draw_waypoint_diamond(self, gx: float, gy: float):
        """
        Draws a green diamond (rotated square) at the given grid-space coordinates
        to mark a CommandCenter-issued waypoint.

        Args:
            gx: Grid-space X coordinate (0..sim_width-1).
            gy: Grid-space Y coordinate (0..sim_height-1).
        """
        cx = int(gx * self.cell_size)
        cy = int(gy * self.cell_size)
        half = max(5, self.cell_size)
        points = [
            (cx,          cy - half),   # top
            (cx + half,   cy),          # right
            (cx,          cy + half),   # bottom
            (cx - half,   cy),          # left
        ]
        pygame.draw.polygon(self.screen, (0, 220, 0), points, 2)   # green outline
        pygame.draw.polygon(self.screen, (0, 255, 0, 80), points, 1) # subtle inner fill line

    def _draw_happy_face(self, cx: int, cy: int, size: int):
        """
        Draws a happy face representation for alive agents.
        """
        r = max(3, size // 2 - 1)
        # Yellow face
        pygame.draw.circle(self.screen, (255, 255, 0), (cx, cy), r)
        # Eyes
        eye_offset = max(1, r // 3)
        pygame.draw.circle(self.screen, (0, 0, 0), (cx - eye_offset, cy - eye_offset), 1)
        pygame.draw.circle(self.screen, (0, 0, 0), (cx + eye_offset, cy - eye_offset), 1)
        # Smile
        smile_y = cy + max(1, r // 3)
        smile_w = max(2, r // 2)
        pygame.draw.line(self.screen, (0, 0, 0), (cx - smile_w, smile_y), (cx + smile_w, smile_y), 1)
        pygame.draw.line(self.screen, (0, 0, 0), (cx - smile_w, smile_y), (cx - smile_w, smile_y - 1), 1)
        pygame.draw.line(self.screen, (0, 0, 0), (cx + smile_w, smile_y), (cx + smile_w, smile_y - 1), 1)

    def _draw_skull(self, cx: int, cy: int, size: int):
        """
        Draws a skull representation for casualties.
        """
        r = max(3, size // 2 - 1)
        # White skull head (centered slightly higher)
        pygame.draw.circle(self.screen, (240, 240, 240), (cx, cy - 1), r)
        # Jaw (white rectangle at the bottom)
        jaw_w = max(3, r)
        jaw_h = max(2, r // 2 + 1)
        pygame.draw.rect(self.screen, (240, 240, 240), pygame.Rect(cx - jaw_w // 2, cy, jaw_w, jaw_h))
        # Dark eye holes
        eye_offset = max(1, r // 3)
        pygame.draw.circle(self.screen, (40, 40, 40), (cx - eye_offset, cy - eye_offset), 1)
        pygame.draw.circle(self.screen, (40, 40, 40), (cx + eye_offset, cy - eye_offset), 1)
        # Teeth line (dark line down the jaw)
        pygame.draw.line(self.screen, (100, 100, 100), (cx, cy), (cx, cy + jaw_h - 1), 1)

    def refresh(self):
        """
        Updates the pygame visualization window to match the current grid state.
        Handles window events to keep the window responsive.
        """
        # Event pump & loop handling to avoid OS "Not Responding" and window hang
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_UP:
                    self.fps = min(60, self.fps + 1)
                    print(f"Simulation speed increased to {self.fps} FPS")
                elif event.key == pygame.K_DOWN:
                    self.fps = max(1, self.fps - 1)
                    print(f"Simulation speed decreased to {self.fps} FPS")
                elif event.key == pygame.K_s:
                    import time
                    os.makedirs("screenshots", exist_ok=True)
                    filename = f"screenshots/screenshot_{int(time.time())}.png"
                    self.save_screenshot(filename)

        # Clear screen
        self.screen.fill((0, 0, 0))

        # Render each cell in the grid
        for x in range(self.ca.width):
            for y in range(self.ca.height):
                cell = self.ca.grid[x][y]
                color = self.colors.get(cell.state, (0, 0, 0))
                
                # Draw the cell
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

        # Draw human agents on top of the terrain grid
        for agent in self.human_agents:
            if agent.activity_type == HumanAgentState.RESCUED:
                continue
            cx = int((agent.x + 0.5) * self.cell_size)
            cy = int((agent.y + 0.5) * self.cell_size)
            
            if agent.activity_type == HumanAgentState.CASUALTY:
                self._draw_skull(cx, cy, self.cell_size)
            else:
                self._draw_happy_face(cx, cy, self.cell_size)

        # Draw home base — position is read from the first UAV (all share the same base)
        base = self.uavs[0].home_base if self.uavs else (0.0, 0.0)
        home_x = int(base[0] * self.cell_size)
        home_y = int(base[1] * self.cell_size)
        pygame.draw.rect(
            self.screen, 
            (0, 191, 255),  # Deep Sky Blue
            pygame.Rect(home_x - 8, home_y - 8, 16, 16),
            2
        )
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            pygame.Rect(home_x - 4, home_y - 4, 8, 8)
        )

        # Draw UAV agents
        for uav in self.uavs:
            cx = int(uav.x * self.cell_size)
            cy = int(uav.y * self.cell_size)
            
            # Determine color
            if uav.uav_type == UAVType.RECON:
                color = (255, 255, 0)      # Yellow
            elif uav.uav_type == UAVType.EXTINGUISH:
                color = (0, 255, 255)      # Cyan
            else:
                color = (255, 0, 255)      # Magenta

            # Draw detection range outline
            detection_radius = int(uav.detection_range * self.cell_size)
            pygame.draw.circle(self.screen, (100, 149, 237), (cx, cy), detection_radius, 1) # Cornflower Blue

            # Draw waypoint path + green diamond — only while actively travelling
            if uav.state == UAVState.TRAVELLING:
                target_x = int(uav.waypoint_x * self.cell_size)
                target_y = int(uav.waypoint_y * self.cell_size)
                pygame.draw.line(self.screen, (180, 180, 180), (cx, cy), (target_x, target_y), 1)
                self._draw_waypoint_diamond(uav.waypoint_x, uav.waypoint_y)

            # Draw the drone body and border
            pygame.draw.circle(self.screen, color, (cx, cy), 6)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), 6, 1)

        # Update full display
        pygame.display.flip()

        # add a frame to the recording
        if self.is_recording:
            self.frames.append(self.screen.copy())
    
    def save_screenshot(self, filename: str):
        """
        Saves the current display frame to a file.
        
        Args:
            filename: Path to the image file to save (e.g. 'screenshots/step_001.png')
        """
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        pygame.image.save(self.screen, filename)
        print(f"Screenshot saved to {filename}")

    def save_recording(self, filename: str):
        """
        Saves the frames as a video output file.
        
        Args:
            filename: Path to the video file to save (e.g. 'video/recording.mp4')
        """
        import numpy as np

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        # pygame.surfarray.array3d returns (width, height, 3); imageio wants (height, width, 3)
        rgb_frames = [
            np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
            for surface in self.frames
        ]
        imageio.mimsave(filename, rgb_frames, fps=30)
        print(f"Saved human demonstration video to {filename}")