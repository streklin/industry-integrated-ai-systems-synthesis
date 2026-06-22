import os
import sys
import random
import argparse
import pygame
import pickle

# Force UTF-8 output so LLM responses containing Unicode (e.g. → arrows)
# don't crash with UnicodeEncodeError on Windows consoles using cp1252.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure the parent directory is in sys.path so we can import WildFireCA and HumanAgents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Ensure Pygame directory is also in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from WildFireCA.WildFireCA import WildfireCA, WildFireState
from HumanAgents.HumanAgent import HumanAgent, HumanAgentState
from visualizer import WildfireVisualizer
from UAVAgents.UAVBase import ReconUAV, FireControlUAV, RescueUAV, UAVState, UAVType
from UAVAgents.UAVSimulatorModule import UAVSimulator
from simulation_coordinator import SimulationCoordinator

def main():
    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(description="Wildfire UAV Simulation")
    parser.add_argument(
        "--no-agentic", action="store_true",
        help="Disable the Agentic Workflow (run UAVs with autonomous behaviors only)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=400,
        help="Hard stop after this many simulation steps (default: 400)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without visualization for fast data collection"
    )
    args = parser.parse_args()

    enable_agentic = not args.no_agentic
    max_steps = args.max_steps
    headless = args.headless

    # Simulation Parameters
    width, height = 100, 100
    seed = random.randint(0, 100000)
    num_humans = 15
    cell_size = 10 # 10 pixels per cell gives a 1000x1000 window
    
    print(f"Initializing Wildfire CA simulation on {width}x{height} grid with seed={seed}...")
    ca = WildfireCA(width=width, height=height, seed=seed)
    ca.regenerate(seed=seed)
    
    # 1. Spawn human agents on random dry land (not water, not burning)
    human_agents = []
    attempts = 0
    while len(human_agents) < num_humans and attempts < 1000:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        cell = ca.grid[x][y]
        
        if cell.state not in [WildFireState.WATER, WildFireState.BURNING, WildFireState.FIRE]:
            # Alternate activity types
            activity = random.choice([HumanAgentState.HIKING, HumanAgentState.CAMPING])
            # Bounding limits should be width-1 and height-1 to match index ranges
            agent = HumanAgent(x=x, y=y, max_x=width - 1, max_y=height - 1, activity_type=activity)
            human_agents.append(agent)
        attempts += 1

    print(f"Successfully spawned {len(human_agents)} human agents in the wilderness.")
    
    # Ignite a starting fire in any burnable cell on the grid randomly
    burnable_cells = []
    for x in range(width):
        for y in range(height):
            cell = ca.grid[x][y]
            if cell.state in [WildFireState.GRASSLAND, WildFireState.SHRUB, WildFireState.TREE, WildFireState.HOUSING]:
                burnable_cells.append(cell)
                
    if burnable_cells:
        selected_cell = random.choice(burnable_cells)
        selected_cell.ignite()
        print(f"Starting fire ignited randomly at ({selected_cell.x}, {selected_cell.y}) state={selected_cell.state.name}")
    else:
        center_x = width // 2
        center_y = height // 2
        ca.grid[center_x][center_y].ignite()
        print(f"Forced start fire ignited at center ({center_x}, {center_y})")

    # Pick a random home base on dry, non-burning land
    safe_cells = [
        (x, y)
        for x in range(width) for y in range(height)
        if ca.grid[x][y].state not in [
            WildFireState.WATER, WildFireState.BURNING, WildFireState.FIRE, WildFireState.ASH
        ]
    ]
    base_x, base_y = random.choice(safe_cells)
    base_x, base_y = float(base_x), float(base_y)
    print(f"Home base placed at ({base_x:.0f}, {base_y:.0f})")

    # Initialize UAV agents
    uavs = []
    uav_id_counter = 0

    # 5 Recon UAVs
    for i in range(5):
        uav = ReconUAV(uav_id=uav_id_counter, width=width, height=height)
        uavs.append(uav)
        uav_id_counter += 1

    # 2 Extinguish UAVs
    for i in range(2):
        uav = FireControlUAV(uav_id=uav_id_counter, width=width, height=height)
        uavs.append(uav)
        uav_id_counter += 1

    # 1 Rescue UAV
    for i in range(1):
        uav = RescueUAV(uav_id=uav_id_counter, width=width, height=height)
        uavs.append(uav)
        uav_id_counter += 1

    # Apply shared home base to all UAVs — must be done after construction
    # so that home_base, x, and y are all consistent from the first tick.
    for uav in uavs:
        uav.home_base = (base_x, base_y)
        uav.x = base_x
        uav.y = base_y

    # Initialize Coordinator (only when agentic workflow is enabled)
    coordinator = None
    if enable_agentic:
        coordinator = SimulationCoordinator(n_steps=40)
        print("[Config] Agentic Workflow ENABLED")
    else:
        print("[Config] Agentic Workflow DISABLED — UAVs operate autonomously only")


    uav_simulator = UAVSimulator(uavs)

    # Initialize Pygame Visualizer (or skip in headless mode)
    visualizer = None
    if not headless:
        visualizer = WildfireVisualizer(ca, human_agents=human_agents, uavs=uav_simulator.uavs, cell_size=cell_size, window_title="Wildfire, Humans & UAV Simulation", is_recording=True)
        visualizer.refresh()
    elif enable_agentic:
        # Agentic workflow needs a Pygame surface for screenshots even in headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        # Create a hidden surface the coordinator can render to
        visualizer = WildfireVisualizer(ca, human_agents=human_agents, uavs=uav_simulator.uavs, cell_size=cell_size, window_title="Headless", is_recording=False)
        print("[Config] Headless mode with hidden Pygame surface (agentic needs screenshots)")
    else:
        print("[Config] Headless mode — Pygame not initialized")

    running = True
    step = 0
    _recording_saved = False

    mode_label = "HEADLESS" if headless else "VISUAL"
    print(f"Starting simulation loop [{mode_label}] (max {max_steps} steps).")
    if not headless:
        print("Press ESC or close the window to quit.")

    def _save_recording():
        """Save recording once. Subsequent calls are no-ops."""
        nonlocal _recording_saved
        if _recording_saved:
            return
        _recording_saved = True
        if visualizer and visualizer.is_recording and visualizer.frames:
            print("SAVING RECORDING...")
            visualizer.save_recording("video/human_behavior.mp4")

    def _save_and_quit():
        """Save the recording (if enabled) then exit cleanly."""
        print("EXITING...")
        _save_recording()
        if not headless:
            pygame.quit()
        sys.exit()

    # max_steps is set via CLI args (default 500)

    try:
        while running:
            # Handle Pygame events only in visual mode
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        _save_and_quit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                            _save_and_quit()
                        elif event.key == pygame.K_SPACE and coordinator is not None:
                            print("\n[Simulation Paused]")
                            user_query = input("Enter your query for the Command Center Assistant: ")
                            if user_query.strip():
                                print("Querying Assistant Agent...")
                                response = coordinator.command_center.query(user_query)
                                print(f"\n[Assistant Response]:\n{response}\n")
                            print("[Simulation Resumed]\n")

                if not running:
                    break

            # Check active fire count
            burning_cells = sum(
                1 for x in range(width) for y in range(height) 
                if ca.grid[x][y].state == WildFireState.BURNING
            )
            
            # 2. Update human agents logic
            for agent in human_agents:
                agent.update()
                
                # Check if current cell is burning
                if ca.grid[agent.x][agent.y].state == WildFireState.BURNING:
                    agent.mark_casualty()
            
            # 3. Update UAV simulation and Dispatch Decision logic
            uav_simulator.update(delta_time=1.0, ca_grid=ca.grid, humans=human_agents)
            
            # Trigger Coordinator to communicate with CommandCenterAgent
            if coordinator is not None:
                # In headless+agentic mode, refresh the hidden surface so the
                # coordinator gets an up-to-date screenshot.
                if headless and visualizer is not None:
                    visualizer.refresh()
                coordinator.on_step(step, visualizer, uav_simulator.uavs, human_agents)

            # 4. Update CA simulation step (once every 10 steps)
            if step % 10 == 0:
                ca.update()
            step += 1
            
            # Refresh the pygame visualization (visual mode only)
            if not headless and visualizer is not None:
                visualizer.refresh()
            
            # Calculate human statistics
            casualties = sum(1 for a in human_agents if a.activity_type == HumanAgentState.CASUALTY)
            rescued = sum(1 for a in human_agents if a.activity_type == HumanAgentState.RESCUED)
            alive = len(human_agents) - casualties - rescued
            
            # Show status in console (less frequently in headless for speed)
            if step % (50 if headless else 10) == 0 or burning_cells == 0:
                print(f"Step {step:04d}/{max_steps} | Burning: {burning_cells:4d} | Alive: {alive:2d} | Rescued: {rescued:2d} | Casualties: {casualties:2d}")

            # If fire has burned out or hard step limit reached, print final stats and wait
            if burning_cells == 0 or step >= max_steps:
                # --- Final Statistics ---
                ash_cells = sum(
                    1 for x in range(width) for y in range(height)
                    if ca.grid[x][y].state == WildFireState.ASH
                )
                total_burned_area = ash_cells + burning_cells  # ASH + still-BURNING
                total_cells = width * height
                burned_pct = (total_burned_area / total_cells) * 100

                stop_reason = "Fire burned out" if burning_cells == 0 else f"Hard stop at {max_steps} steps"
                agentic_mode = "ENABLED" if enable_agentic else "DISABLED"

                print("\n" + "=" * 60)
                print("           SIMULATION COMPLETE — FINAL STATISTICS")
                print("=" * 60)
                print(f"  Stop Reason        : {stop_reason}")
                print(f"  Agentic Workflow   : {agentic_mode}")
                print(f"  Total Steps        : {step}")
                print(f"  Seed               : {seed}")
                print(f"  Grid Size          : {width}x{height} ({total_cells} cells)")
                print(f"  ─── Burned Area ───")
                print(f"  Ash Cells          : {ash_cells}")
                print(f"  Still Burning      : {burning_cells}")
                print(f"  Total Burned Area  : {total_burned_area} cells ({burned_pct:.1f}%)")
                print(f"  ─── Human Outcomes ───")
                print(f"  Alive              : {alive}")
                print(f"  Rescued            : {rescued}")
                print(f"  Casualties         : {casualties}")
                print("=" * 60)

                if headless:
                    # Headless: auto-exit after printing stats
                    _save_and_quit()
                else:
                    print("Press ESC or close the window to exit.")
                    while True:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                _save_and_quit()
                            elif event.type == pygame.KEYDOWN:
                                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                                    _save_and_quit()
                        pygame.time.delay(100)

            # Control simulation speed (no throttle in headless for max speed)
            if not headless and visualizer is not None:
                visualizer.clock.tick(visualizer.fps)

    finally:
        # Guaranteed to run no matter how the loop exits: normal exit,
        # exception from coordinator API call, KeyboardInterrupt, etc.
        _save_recording()
        if not headless:
            pygame.quit()


if __name__ == "__main__":
    main()

