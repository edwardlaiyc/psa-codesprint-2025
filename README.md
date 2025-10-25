# PSA CodeSprint 2025 CHAMPIONS

## Overview
Problem statement 1 - AI for horizontal transport optimisation
Solution - a multi-agent path finding (MAPF) problem coordinating 80 horizontal transport vehicles between 8 quay cranes and 16 yard blocks, processing 20,000 container jobs while navigating shared lanes with directional constraints.

## Approach

Two-Phase Optimization Strategy:

1. Hard Heuristics - Custom algorithms for resource allocation:
   - HT Selector: Distance-based assignment with load balancing, lateral movement penalties, and side preference logic
   - Yard Selector: Congestion-aware allocation with capacity limits and idle exploration bonuses
   - Path Planning: Lane-specific routing rules (even lanes down, odd lanes up) to minimize conflicts

2. Bayesian Optimization - Automated hyperparameter tuning:
   - Gaussian Process surrogate modeling of simulation performance
   - Expected Improvement acquisition function for parameter exploration
   - 7 parameters for HT selector, 4 for yard selector
   - Converged to optimal weights minimizing total simulation time

## Key Files

- src/plan/job_planner.py - Main planning orchestrator with navigation logic
- src/plan/ht_selector_ai.py - Adaptive HT assignment algorithm
- src/plan/yard_selector_ai.py - Dynamic yard allocation algorithm
- src/plan/tune_ht_selector.py - Bayesian optimization for HT parameters
- src/plan/tune_yard_selector.py - Bayesian optimization for yard parameters

## Results

~45% reduction in simulation time - from approximately 1,160,000 seconds (baseline) to 640,000 seconds through optimized resource allocation and conflict-free path planning.

## Environment Setup

Before working on the problem, please create a Python virtual environment and install all required dependencies.

If you are familiar with Python environments, you can directly install from `requirements.txt`. Otherwise, follow these step-by-step instructions in your Command Prompt (Windows):

```shell
# Create a Python virtual environment. Here I used Python 3.11.
python -m pip install virtualenv
python -m virtualenv py_env_codesprint
py_env_codesprint\Scripts\activate

# Install pip-tools to manage dependencies
python -m pip install pip-tools

# Compile and synchronize dependencies from requirements.in to requirements.txt
pip-compile -v --rebuild -o requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
pip-sync requirements.txt

# (Optional) Set up Jupyter kernel for experimenting with notebooks
python -m ipykernel install --user --name=py_env_codesprint_kernel
```

To activate your environment:

```shell
py_env_codesprint\Scripts\activate
```

## Development

To get started, run the simulation with a GUI by executing:

```shell
python gui.py
```

You will see a simulation of the port terminal operations.

### About the Code Base

The simulation consists of two main components:

- **Planning Engine**: Where scheduling and navigating algorithms are implemented.
- **Operation Engine**: Simulates execution of planned jobs.

Your task is to improve the default planning algorithm in the Planning Engine for better performance.

### Project Structure

```
INA_DS_CS/
│
├── data/
│   └── input.csv
├── logs/
├── src/
│   ├── operate/
│   ├── plan/
│   │   ├── __init__.py
│   │   ├── job_planner.py                # Main planning orchestrator with navigation logic
│   │   ├── job_tracker.py                # Tracks job status, progress, and completion times
│   │   ├── progress_monitor.py           # Tracks overall progress and runtime statistics
│   │   ├── ht_selector_ai.py             # Adaptive HT assignment (distance + penalties)
│   │   ├── yard_selector_ai.py           # Dynamic yard allocation (congestion-aware)
│   │   ├── tune_ht_selector.py           # Bayesian optimization for HT selector parameters
│   │   └── tune_yard_selector.py         # Bayesian optimization for yard selector parameters
│   ├── ui/
│   └── utils/
├── cli.py
└── gui.py

```

- `input.csv` contains the job details fed into the planning algorithm.
- `job_planner.py` is the core of the Planning Engine — your modifications should be made here (Search for **YOUR TASK HERE**). Refer to the provided documentation for details on the default algorithm logic.
- `gui.py` runs the simulation with a graphical interface so you can observe and fine-tune your algorithm. You can adjust `number_of_refresh_per_physical_second` in the code to control simulation speed.
- `cli.py` runs the simulation without GUI for faster batch execution. It outputs `output.csv` in the data folder and logs in `logs/`.

### Running the CLI simulation

Run:

```
python cli.py
```

The simulation may take a few minutes. At the end, it will report the total simulation time (e.g., 1,167,610 seconds) — **this is the key metric you need to optimize and reduce**. Note that this is the simulation time when system checks for all job completion condition. A precise time of the last job completion can be found in `end_time` column of `output.csv` file.
