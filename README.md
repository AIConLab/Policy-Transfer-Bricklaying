# Spatially Adaptive Policy Transfer for Accelerated Reinforcement Learning in Sequential Robotic Bricklaying

This repository contains the official implementation of the paper:

> **Spatially Adaptive Policy Transfer for Accelerated Reinforcement Learning in Sequential Robotic Bricklaying**
> Mohsen Navazani, Ao Du, Shuai Li, Jiannan Cai
> *ASCE Journal of Computing in Civil Engineering (JCCE), under review.*

## 🚧 Code Release Notice

This repository currently provides a **preview of the core training code** for the policy transfer task of the paper, so that reviewers and interested readers can inspect the PPO algorithm, the network architecture, and the overall training pipeline used to produce the reported results.

**The complete codebase — including the MuJoCo simulation environment (`Mujoco_Gripper` package with the scene XML, robot model, and mesh assets), all task configurations (source task, sequential multi-brick task, spatial-adaptation task), the full set of trained checkpoints, and the robustness-evaluation scripts — will be publicly released upon publication of the paper.**

Because the simulation environment is not yet included, the code in this repository is **not runnable as-is**; it is provided for inspection only. The commands in the [Usage](#usage) section below describe how the released codebase will be used once the full code is published.

## Overview

This project implements a two-stage learning framework that combines **phase-based Proximal Policy Optimization (PPO)** with **Policy Transfer (PT)** for long-horizon sequential robotic bricklaying. A Universal Robots **UR5e** manipulator mounted on a **Clearpath Husky** mobile base is trained in a high-fidelity **MuJoCo** physics simulation to pick and place bricks in a sequential manner.

The framework:

1. First learns a robust single-brick pick-and-place policy using a phase-based reward decomposition (Approach → Grasp → Transport → Place).
2. Then transfers the learned policy weights to accelerate training on a multi-brick sequential task.
3. Finally demonstrates spatial adaptation to previously unseen geometric configurations with only a few iterations of fine-tuning.

Key experimental results from the paper:

* **81.1% reduction** in training convergence time compared to training from scratch.
* **61.8% improvement** in placement precision.
* **100% success rate** under friction (-70% to +50%) and mass (-20% to +50%) perturbations without retraining.
* **< 2 cm placement error** on a spatially shifted third brick with only **25 iterations** of fine-tuning.

## Repository Structure

```
.
├── README.md               # This file
├── LICENSE                 # MIT License
├── gitignore               # Files/directories ignored by git
├── requirements.txt        # Python dependencies
├── setup.py                # Registers the BrickLaying gym environment
│
├── main.py                 # Entry point: training and evaluation
├── arguments.py            # Command-line argument parsing
├── ppo.py                  # PPO algorithm implementation
├── network.py              # Actor and critic MLP networks
├── evaluate_model.py       # Policy evaluation utilities
│
└── checkpoints/
    └── target_task/        # Trained model weights for the policy transfer task
```

## Installation

### Prerequisites

* Python **3.8** or later (tested on 3.8 and 3.10)
* Linux (tested on Ubuntu 20.04 / 22.04)
* A MuJoCo installation accessible to `mujoco` and/or `mujoco-py`

### Troubleshooting: GLEW initialization error

If, while running the code, you encounter the error:

```
GLEW initialization error: Missing GL version
```

set the following environment variable before running again:

```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

## Usage

> The following commands describe the intended workflow once the full code (including the MuJoCo environment package) is released. They are provided here for reference.

First I recommend creating a python virtual environment:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To train from scratch:

```
python main.py
```

To test model:

```
python main.py --mode test --actor_model checkpoints/target_task/ppo_actor.pth
```

To train with existing actor/critic models:

```
python main.py --actor_model checkpoints/target_task/ppo_actor.pth --critic_model checkpoints/target_task/ppo_critic.pth
```

NOTE: to change hyperparameters, environments, etc. do it in **main.py**; I didn't have them as command line arguments because I don't like how long it makes the command.

## Environment Details

The `AIconGrapper` environment (released with the full codebase) implements a 45-dimensional observation space and a 7-dimensional continuous action space as described in the paper:

* **Observation (45-D):** end-effector position (3), target brick position (3), system joint positions (20), system joint velocities (19).
* **Action (7-D):** six UR5e joint position commands and one gripper command. The gripper follows the paper's convention: **-1 commands the gripper to close** and **+1 commands it to open**.
* **Reward:** phase-based decomposition with per-step distance rewards, wrist orientation penalties, and one-time milestone bonuses. Exact coefficients are listed in Table 3 of the paper.

## Citation

If you use this code or build on this work, please cite the paper:

```bibtex
@article{navazani2026policytransfer,
  title   = {Spatially Adaptive Policy Transfer for Accelerated Reinforcement Learning in Sequential Robotic Bricklaying},
  author  = {Navazani, Mohsen and Du, Ao and Li, Shuai and Cai, Jiannan},
  journal = {ASCE Journal of Computing in Civil Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgments

* This research was funded by the US National Science Foundation (NSF) via 2138514, 2222670 and 2222810, and the Transportation Infrastructure Precast Innovation Center (TRANS-IPIC), Tier 1 University Transportation Center (UTC) via Project UT-25-RP-01. The authors gratefully acknowledge the support. Any opinions, findings, recommendations, and conclusions in this paper are those of the authors, and do not necessarily reflect the views of NSF, TRANS-IPIC, The University of Texas at San Antonio, or The University of Florida.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

## Contact

For questions about the code or the paper, please contact:

* **Mohsen Navazani** — mohsen.navazani@utsa.edu 
The University of Texas at San Antonio, School of Civil & Environmental Engineering, and Construction Management.