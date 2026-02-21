 ## MDP-Control-in-ATO-Systems
 
 Markov decision process (MDP) control for assembly-to-order (ATO) production systems under stochastic demand and lead times.
-This codebase compares optimal dynamic policies (via DP), heuristic controls, and baseline rules (e.g., FCFS) on multi-product ATO settings with backlogs, buffers, and order acceptance.
 
-## Why this project
+This repository compares three policy families in the same simulator-style setup:
 
-Realistic ATO dynamics: stochastic order arrivals, finite buffers/backlogs, acceptance/rejection, and fulfillment.
+- **Optimal dynamic policy** via value iteration (`newoptimal.py` + `stationaction.py`)
+- **Threshold heuristic policy** with grid-search tuning (`newheuristic.py`)
+- **FCFS baseline policy** for simple benchmarking (`fcfs_policy.py`)
 
-Decision analytics: optimal control vs. practical heuristics, with cost accounting (holding, backlog/unsatisfied order, rejection).
+---
 
-Research-ready: runs controlled experiments and sensitivity analyses with reproducible random seeds.
+## Quick mental model (for newcomers)
+
+Think of the system state as:
+
+- `state[0]`: on-hand production inventory (shared supply)
+- `state[1:]`: backlog/order buffers per product class
+
+At each time step, the model simulates:
+
+1. **Order arrivals** (accept or reject)
+2. **Order due events** (satisfy or delay)
+3. **Production replenishment** (raise inventory toward a target)
+4. **Cost/profit accounting**
+
+The objective is to minimize discounted cost (holding + rejection + unsatisfied penalties), while tracking fulfillment/profit metrics.
+
+---
 
 ## Repository structure
 
-newoptimal.py – Optimal control via dynamic programming / value iteration style routines; produces benchmark policies and costs. 
-GitHub
+### Core dynamics and optimal control
+
+- **`stationaction.py`**
+  - Defines `StateAction`, the core MDP helper.
+  - Handles state-index conversion (`state_to_number`, `number_to_state`), Bellman-style value update (`state_trans_value`), one value-iteration sweep (`one_iteration`), and greedy control extraction (`control_decision`).
+  - If you only read one file first, read this one.
+
+- **`newoptimal.py`**
+  - General **N-product** runner for the optimal policy.
+  - Builds a `StateAction`, runs value iteration to convergence, then performs Monte Carlo simulation and logs trajectories to CSV.
+  - Includes `run_from_csv(...)` to batch-run systems from parameter files.
+
+### Heuristic and baseline policies
+
+- **`newheuristic.py`**
+  - Implements a threshold policy class (`ThresholdHeuristicMDP`).
+  - Simulates using threshold decisions.
+  - Provides `optimize_thresholds_grid_search(...)` to tune threshold values by brute force.
+
+- **`fcfs_policy.py`**
+  - Implements an FCFS-like baseline (accept when queue has space, fulfill when inventory exists, refill to cap).
+  - Useful as a lower-complexity comparison to optimal/heuristic methods.
+
+### Scenario script and parameter generation
+
+- **`newcase3_1.py`**
+  - Specialized two-product experiment script.
+  - Runs optimal control, exports control decisions, and visualizes a production switching surface in 3D.
+
+- **`randomsystemparameters.py`**
+  - Generates random system parameter sets and writes `random_system_parameters.csv`.
+  - Entry point for reproducible experiment batches.
+
+---
+
+## Important implementation details to know
+
+- **State space grows exponentially** with number of buffers and capacities (`∏(B_i+1)`), so exact value iteration becomes expensive quickly.
+- **Action encoding differs by policy file**:
+  - In `StateAction.control_decision`, acceptance uses `0=accept, 1=reject`, while fulfillment uses `1=satisfy`.
+  - In threshold heuristic code, acceptance control uses `1=accept` convention.
+  - Keep this in mind when comparing logged controls across scripts.
+- **Discounting** uses `gamma = 1 / (1 + theta)` in simulators.
+- **Stochastic events** are Bernoulli checks per time step based on `lmd`, `omg`, and `mu`-style parameters.
+- **CSV logs** are first-class outputs across scripts; they are the easiest way to inspect policy behavior over time.
+
+---
+
+## Typical workflows
+
+### 1) Generate random systems
 
-newheuristic.py – Implementations of threshold/priority heuristics for fast, near-optimal control in larger instances. 
-GitHub
+```bash
+python randomsystemparameters.py
+```
 
-fcfs_policy.py – First-Come-First-Served (or similar simple rule) baseline for comparison. 
-GitHub
+### 2) Run optimal policy on all generated systems
 
-stationaction.py – Core data structures for states/actions and helper utilities used by the simulators/policies. 
-GitHub
+```bash
+python newoptimal.py
+```
 
-newcase3_1.py – Example experiment script (a specific scenario “case 3.1”); good starting point to reproduce paper-style results. 
-GitHub
+### 3) Run threshold heuristic and tune thresholds
 
-randomsystemparameters.py – Parameter samplers and canned instances (arrival/service rates, costs, capacities, seeds). 
-GitHub
+```bash
+python newheuristic.py
+```
 
+### 4) Run FCFS baseline
 
-## Modeling summary
+```bash
+python fcfs_policy.py
+```
 
-System: multi-product ATO with order arrivals (e.g., Poisson), acceptance/rejection on arrival, backlog queues, and fulfillment subject to capacity.
+### 5) Run the focused 2-product analysis
 
-State: (buffers/backlogs by class, possibly WIP/availability flags, etc.).
+```bash
+python newcase3_1.py
+```
 
-Action: accept vs. reject an order; optional production/expedite toggles depending on scenario.
+---
 
-Objective: minimize long-run average (or discounted) cost combining inventory/holding, backlog/unsatisfied penalties, and rejection penalties.
+## Suggested learning path
 
-Policies compared:
+1. Start with `stationaction.py` to understand state/action/value mechanics.
+2. Read `newoptimal.py` to see how MDP solution + simulation are stitched together.
+3. Compare with `newheuristic.py` and `fcfs_policy.py` to understand tradeoffs (quality vs. runtime/complexity).
+4. Use `newcase3_1.py` to connect policy logic to visual intuition (switching thresholds).
+5. Experiment by changing capacities, costs, and rates in generated CSV parameters, then compare outcome metrics.
 
-- Optimal DP (value iteration / policy evaluation) in newoptimal.py.
+---
 
-- Heuristics (e.g., threshold/priority/ratio rules) in newheuristic.py.
+## Practical next improvements (if you continue development)
 
-- Simple baseline in fcfs_policy.py.
+- Add a shared evaluator so all policies output identical metrics/format.
+- Add unit tests for state-index conversion and control conventions.
+- Separate model dynamics from policy logic into reusable modules.
+- Add plotting notebooks for cost/profit/acceptance tradeoff frontiers.
+- Add reproducibility controls (explicit seeds in all scripts and saved config manifests).
