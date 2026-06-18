# Integrated AI Systems for Wildfire Emergency Response

> **MSc Artificial Intelligence – Final Project**  
> A multi-agent, AI-driven simulation platform for wildfire monitoring, containment, and human rescue using a heterogeneous UAV fleet.

---

## Overview

This project implements a fully integrated AI system that autonomously manages a fleet of unmanned aerial vehicles (UAVs) responding to a wildfire event. The system combines several AI paradigms:

| Layer | Technology | Role |
|---|---|---|
| **Environment** | Cellular Automaton (CA) | Probabilistic wildfire spread simulation |
| **UAV Policy** | Reinforcement Learning (SVM / RandomForest) | Per-UAV autonomous flight, fire-fighting, and rescue behaviour |
| **Risk Prediction** | UNet (CNN) | Real-time fire-spread risk map from satellite imagery |
| **Command & Control** | LLM Agentic Workflow (Claude Sonnet/Haiku) | High-level strategy planning and UAV dispatch |
| **State Memory** | Knowledge Graph (MGraph) | Persistent goal and priority tracking across planning cycles |
| **Human Interface** | Natural language console | Operator queries and priority overrides mid-simulation |

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
          │  │ StrategyAgent   │    │  claude-sonnet-4-5
          │  │ (PlanGuidelines)│◄───┤  + Knowledge Graph
          │  └────────┬────────┘    │
          │           │             │
          │  ┌────────▼────────┐    │
          │  │ DispatchAgent   │    │  claude-sonnet-4-5
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
| `RLTraining.py` | Supervised / RL training pipeline. Collects experience from random agent rollouts and trains SVM/RF policy models saved to `models/`. |

### `FirePrediction/`
| File | Description |
|---|---|
| `UNet.py` | Custom lightweight UNet (encoder depth 4, bottleneck 512ch) for binary fire-risk segmentation. |
| `fire_prediction_model.ipynb` | Training notebook – generates the `models/fire_predictor_lrg.pth` weights. |

### `AgenticWorkflow/`
| File | Description |
|---|---|
| `agentic_workflow.py` | Three `pydantic-ai` agents sharing a common MGraph knowledge graph. `StrategyAgent` (Sonnet) produces a `PlanGuidelines` object every N cycles. `DispatchAgent` (Sonnet) translates the plan into per-UAV `UAVCommand` objects. `AssistantAgent` (Haiku) answers operator queries in natural language. |
| `graphdb.py` | Thread-safe wrapper around MGraph with a domain ontology for LONG TERM GOAL → GOAL → GOAL_TYPE / POSITION / PRIORITY. All methods serialised with `threading.Lock` to prevent concurrent-write crashes from pydantic-ai's parallel tool dispatch. |
| `pydantic_models.py` | Shared Pydantic response schemas: `CommandCenterResponse`, `UAVCommand`, `PlanGuidelines`, `PlanGuideline`. |

### `Pygame/`
| File | Description |
|---|---|
| `run_simulation.py` | Main entry point. Creates all agents, runs the Pygame event loop, captures video frames, and guarantees video save on exit (via `try/finally`). |
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

The operator can pause the simulation at any time (press **`P`**) and type a natural-language query or directive into the console. The `AssistantAgent` processes it and can update the knowledge graph to override priorities for subsequent planning cycles.

---

## Quick Start

### Prerequisites
- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

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

Create `AgenticWorkflow/.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the simulation

```bash
cd Pygame
python run_simulation.py
```

A 1000×1000 Pygame window opens. The simulation runs for up to 750 steps (configurable). An MP4 recording is automatically saved to `Pygame/video/human_behavior.mp4` on exit.

### Controls

| Key | Action |
|---|---|
| `ESC` / close window | Exit and save video |
| `P` | Pause and open operator console |

---

## Training the UAV Policies

```bash
cd UAVAgents
python RLTraining.py
```

Trained models are saved to `UAVAgents/models/`:
- `recon_policy.pkl`
- `extinguish_policy.pkl`
- `rescue_policy.pkl`

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
