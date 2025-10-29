## MDP-Control-in-ATO-Systems

Markov decision process (MDP) control for assembly-to-order (ATO) production systems under stochastic demand and lead times.
This codebase compares optimal dynamic policies (via DP), heuristic controls, and baseline rules (e.g., FCFS) on multi-product ATO settings with backlogs, buffers, and order acceptance.

## Why this project

Realistic ATO dynamics: stochastic order arrivals, finite buffers/backlogs, acceptance/rejection, and fulfillment.

Decision analytics: optimal control vs. practical heuristics, with cost accounting (holding, backlog/unsatisfied order, rejection).

Research-ready: runs controlled experiments and sensitivity analyses with reproducible random seeds.

## Repository structure

newoptimal.py – Optimal control via dynamic programming / value iteration style routines; produces benchmark policies and costs. 
GitHub

newheuristic.py – Implementations of threshold/priority heuristics for fast, near-optimal control in larger instances. 
GitHub

fcfs_policy.py – First-Come-First-Served (or similar simple rule) baseline for comparison. 
GitHub

stationaction.py – Core data structures for states/actions and helper utilities used by the simulators/policies. 
GitHub

newcase3_1.py – Example experiment script (a specific scenario “case 3.1”); good starting point to reproduce paper-style results. 
GitHub

randomsystemparameters.py – Parameter samplers and canned instances (arrival/service rates, costs, capacities, seeds). 
GitHub


## Modeling summary

System: multi-product ATO with order arrivals (e.g., Poisson), acceptance/rejection on arrival, backlog queues, and fulfillment subject to capacity.

State: (buffers/backlogs by class, possibly WIP/availability flags, etc.).

Action: accept vs. reject an order; optional production/expedite toggles depending on scenario.

Objective: minimize long-run average (or discounted) cost combining inventory/holding, backlog/unsatisfied penalties, and rejection penalties.

Policies compared:

- Optimal DP (value iteration / policy evaluation) in newoptimal.py.

- Heuristics (e.g., threshold/priority/ratio rules) in newheuristic.py.

- Simple baseline in fcfs_policy.py.
