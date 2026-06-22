# Integrated AI Systems for Wildfire Emergency Response

> **MSc Artificial Intelligence – Final Project**  
> A multi-agent, AI-driven simulation platform for wildfire monitoring, containment, and human rescue using a heterogeneous UAV fleet.

---

## Overview

This project implements a fully integrated AI system that autonomously manages a fleet of unmanned aerial vehicles (UAVs) responding to a wildfire event. The system combines several AI paradigms:

| Layer | Technology | Role |
|---|---|---|
| **Environment** | Cellular Automaton (CA) | Probabilistic wildfire spread simulation |
| **UAV Behaviour** | Subsumption Architecture | Per-UAV autonomous flight, fire-fighting, and rescue behaviour |
| **Risk Prediction** | UNet (CNN) | Real-time fire-spread risk map from satellite imagery |
| **Command & Control** | LLM Agentic Workflow (Claude Haiku) | High-level strategy planning and UAV dispatch (using Haiku for budget reasons) |
| **State Memory** | Knowledge Graph (MGraph) | Persistent goal and priority tracking across planning cycles |
| **Human Interface** | Natural language console | Operator queries and priority overrides mid-simulation |
| **Evaluation** | Statistical Analysis | A/B comparison of agentic vs. non-agentic runs via burned area and human outcome metrics |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    run_simulation.py                    │
│   (Pygame loop · visualiser · video recording)          │
└──────────────────────┬──────────────────────────────────┘
                       │ every N steps
          ┌────────────▼────────────┐
          │  SimulationCoordinator  │
          │  • Capture screen PNG   │
          │  • UNet risk inference  │
          │  • Gather UAV telemetry │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   CommandCenterAgent    │
          │                         │
          │  ┌─────────────────┐    │
          │  │ StrategyAgent   │    │  claude-haiku-4-5
          │  │ (PlanGuidelines)│◄───┤  + Knowledge Graph
          │  └────────┬────────┘    │
          │           │             │
          │  ┌────────▼────────┐    │
          │  │ DispatchAgent   │    │  claude-haiku-4-5
          │  │ (UAV Commands)  │◄───┤  + Knowledge Graph
          │  └─────────────────┘    │
          │                         │
          │  ┌─────────────────┐    │
          │  │ AssistantAgent  │    │  claude-haiku-4-5
          │  │ (Operator Q&A)  │    │  (text-only)
          │  └─────────────────┘    │
          └────────────┬────────────┘
                       │ DEPLOY / RECALL commands
          ┌────────────▼────────────┐
          │      UAV Fleet (×8)     │
          │  5× RECON               │
          │  2× EXTINGUISH          │
          │  1× RESCUE              │
          └─────────────────────────┘
```

---

## Modules

### `WildFireCA/`
Cellular automaton that models wildfire spread across a 100×100 grid. Cell types include Grassland, Shrub, Tree, Housing, Urban, Water, Fire, Burning, and Ash. Terrain is generated procedurally using Perlin noise.

### `HumanAgents/`
Simulates civilian agents (hikers and campers) with probabilistic movement, injury, and rescue states. Humans flee fire, get injured near flames, and can be rescued by RESCUE UAVs.

### `UAVAgents/`
| File | Description |
|---|---|
| `UAVBase.py` | Base UAV class with physics (velocity, turn-rate, boundary repulsion), state machine (HANGER → TRAVELLING → CRUISING), and fuel management. Subclasses: `ReconUAV`, `FireControlUAV`, `RescueUAV`. |
| `UAVSimulatorModule.py` | Thin wrapper that creates and steps the full UAV fleet. |

### `FirePrediction/`
| File | Description |
|---|---|
| `UNet.py` | Custom lightweight UNet (encoder depth 4, bottleneck 512ch) for binary fire-risk segmentation. |
| `fire_prediction_model.ipynb` | Training notebook – generates the `models/fire_predictor_lrg.pth` weights. |

### `AgenticWorkflow/`
| File | Description |
|---|---|
| `agentic_workflow.py` | Three `pydantic-ai` agents sharing a common MGraph knowledge graph. `StrategyAgent` (Haiku) produces a `PlanGuidelines` object every N cycles. `DispatchAgent` (Haiku) translates the plan into per-UAV `UAVCommand` objects. `AssistantAgent` (Haiku) answers operator queries in natural language. *(All agents use Claude Haiku 4.5 for budget reasons).* |
| `graphdb.py` | Thread-safe wrapper around MGraph with a domain ontology for LONG TERM GOAL → GOAL → GOAL_TYPE / POSITION / PRIORITY. All methods serialised with `threading.Lock` to prevent concurrent-write crashes from pydantic-ai's parallel tool dispatch. |
| `pydantic_models.py` | Shared Pydantic response schemas: `CommandCenterResponse`, `UAVCommand`, `PlanGuidelines`, `PlanGuideline`. |

### `Pygame/`
| File | Description |
|---|---|
| `run_simulation.py` | Main entry point. Supports visual/headless modes, agentic/non-agentic toggling, and configurable step limits. Prints burned-area and human-outcome statistics on completion. |
| `simulation_coordinator.py` | Invoked every N simulation steps. Captures the screen as a 500×500 PNG, runs UNet inference to produce a red/green risk mask PNG, and calls `CommandCenterAgent.update()`. |
| `visualizer.py` | Renders the CA grid, UAV positions/waypoints/direction arrows, human states, home base (black square), and records MP4 output via `imageio`. |

### `DatasetGenerator/`
Utility scripts for generating labelled fire-spread image datasets used to train the UNet.

---

## UAV Roles & Rules

| UAV | IDs | Capability | Constraint |
|---|---|---|---|
| RECON | 0–4 | Large detection range (5 units). Detects fire, burning cells, and nearby humans. | Cannot extinguish fire. Cannot rescue humans. |
| EXTINGUISH | 5–6 | Actively suppresses FIRE / BURNING cells. | Must be within 20 grid units of active fire. Cannot rescue. |
| RESCUE | 7 | Picks up and transports endangered humans to safety. | Must ONLY be used for human rescue. |

All UAVs start at a randomly placed **home base** (black square on the map) at the beginning of each simulation. UAVs autonomously avoid boundaries via a continuous repulsion force.

---

## Agentic Planning Cycle

Every `N` simulation steps (default: 20):

1. **Screen capture** → 500×500 satellite PNG  
2. **UNet inference** → 500×500 red/green risk-mask PNG  
3. **StrategyAgent** reads both images + UAV telemetry → updates the knowledge graph → returns a prioritised `PlanGuidelines` list (refreshed every 5 dispatch cycles)  
4. **DispatchAgent** reads both images + telemetry + plan → issues a `DEPLOY`/`RECALL` command for **every** UAV  
5. Commands are applied to the UAV fleet for the next N steps

The operator can pause the simulation at any time (press **`SPACE`**) and type a natural-language query or directive into the console. The `AssistantAgent` processes it and can update the knowledge graph to override priorities for subsequent planning cycles.

---

## Quick Start

### Prerequisites
- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) *(only required when running with the agentic workflow enabled)*

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** install PyTorch with CUDA support first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> pip install -r requirements.txt
> ```

### 2. Configure API key

Create `AgenticWorkflow/.env` *(skip if running with `--no-agentic`)*:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the simulation

```bash
cd Pygame
python run_simulation.py
```

#### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--no-agentic` | Off | Disable the LLM-based Agentic Workflow. UAVs operate on autonomous subsumption behaviours only. |
| `--headless` | Off | Run without a Pygame window for fast data collection. Auto-exits after printing statistics. |
| `--max-steps N` | `400` | Hard-stop the simulation after N steps. |

#### Example Run Modes

```bash
# Full visual simulation with agentic workflow (default)
python run_simulation.py

# Visual simulation, UAVs only (no LLM calls)
python run_simulation.py --no-agentic

# Fast headless data collection, no agentic (maximum speed)
python run_simulation.py --headless --no-agentic

# Headless with agentic workflow (hidden Pygame surface for LLM screenshots)
python run_simulation.py --headless

# Custom step limit
python run_simulation.py --headless --no-agentic --max-steps 1000
```

In visual mode, a 1000×1000 Pygame window opens. The simulation runs for up to 400 steps (configurable via `--max-steps`). An MP4 recording is automatically saved to `Pygame/video/human_behavior.mp4` on exit.

### Controls (visual mode only)

| Key | Action |
|---|---|
| `ESC` / `Q` / close window | Exit and save video |
| `SPACE` | Pause and open operator console (requires agentic workflow) |
| `↑` / `↓` | Increase / decrease simulation speed (FPS) |
| `S` | Save a screenshot to `screenshots/` |

---

## Simulation Output & Statistics

At the end of every run, the simulation prints a statistics summary:

```
============================================================
           SIMULATION COMPLETE — FINAL STATISTICS
============================================================
  Stop Reason        : Hard stop at 400 steps
  Agentic Workflow   : DISABLED
  Total Steps        : 400
  Seed               : 42315
  Grid Size          : 100x100 (10000 cells)
  ─── Burned Area ───
  Ash Cells          : 1234
  Still Burning      : 56
  Total Burned Area  : 1290 cells (12.9%)
  ─── Human Outcomes ───
  Alive              : 8
  Rescued            : 5
  Casualties         : 2
============================================================
```

**Total Burned Area** = Ash cells + cells still actively burning at termination. This is the primary metric for statistical comparison between agentic and non-agentic runs.

The random `seed` is printed so runs can be reproduced or paired across experimental conditions.

---

## Training the Fire-Risk UNet

Open and run `FirePrediction/fire_prediction_model.ipynb`. The trained weights are saved to `FirePrediction/models/fire_predictor_lrg.pth`.

---

## Environment Colour Key

| Colour | Cell State |
|---|---|
| 🟩 Lawn Green | Grassland |
| 🟢 Forest Green | Shrub |
| 🌲 Dark Green | Tree |
| 🟧 Dark Orange | Housing |
| ⬜ White | Urban |
| 🔵 Blue | Water |
| 🟥 Red | Burning |
| 🔴 Orange-Red | Fire (active front) |
| ⬛ Grey | Ash (burned out) |
| ⬛ Black | Empty |
| ■ Black Square | UAV Home Base |

---

## Project Structure

```
industry-integrated-ai-systems-synthesis/
├── AgenticWorkflow/          # LLM agents, knowledge graph, pydantic schemas
├── DatasetGenerator/         # Dataset generation utilities
├── FirePrediction/           # UNet architecture and training notebook
├── HumanAgents/              # Civilian agent simulation
├── Pygame/                   # Main simulation runner, visualiser, coordinator
│   └── video/                # Saved MP4 recordings
├── UAVAgents/                # UAV base classes, RL training, saved models
├── WildFireCA/               # Cellular automaton wildfire model
├── requirements.txt
└── README.md
```

---

## License

MIT – see [LICENSE](LICENSE).
