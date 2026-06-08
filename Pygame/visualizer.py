import os
import sys
import pygame

# Ensure the parent directory is in sys.path so we can import WildFireCA and HumanAgents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from WildFireCA.WildFireCA import WildFireState, WildfireCA
from HumanAgents.HumanAgent import HumanAgent, HumanAgentState


class WildfireVisualizer:
    """
    Handles visualizing the current state of a WildfireCA simulation grid and Human Agents using Pygame.
    """
    def __init__(self, ca: WildfireCA, human_agents=None, cell_size: int = 10, window_title: str = "Wildfire CA Simulation"):
        """
        Initialize the visualizer.
        
        Args:
            ca: The WildfireCA simulation instance to visualize.
            human_agents: Optional list of HumanAgent instances.
            cell_size: Size in pixels of each grid cell.
            window_title: Window title.
        """
        self.ca = ca
        self.human_agents = human_agents if human_agents is not None else []
        self.cell_size = cell_size
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
            cx = int((agent.x + 0.5) * self.cell_size)
            cy = int((agent.y + 0.5) * self.cell_size)
            
            if agent.activity_type == HumanAgentState.CASUALTY:
                self._draw_skull(cx, cy, self.cell_size)
            else:
                self._draw_happy_face(cx, cy, self.cell_size)

        # Update full display
        pygame.display.flip()
