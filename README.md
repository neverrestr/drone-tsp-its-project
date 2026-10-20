# Drone Delivery Route Optimization — Procedural Python Version

This is a procedural version of the KKA IUP 2026 project. It uses **no user-defined classes, no dataclasses, and no custom objects**. Results and distance data are represented with dictionaries and ordinary lists.

## What it does

- Models a closed Traveling Salesperson Problem (TSP): start at the depot, visit every delivery point once, return to the depot.
- Compares three local-search / optimization algorithms:
  1. Steepest-Improvement Hill Climbing
  2. Simulated Annealing
  3. Genetic Algorithm
- Runs 20 trials per algorithm for 20 delivery points.
- Runs a scalability experiment for 10, 20, 30, and 50 delivery points.
- Saves CSV summaries and PNG plots to `results/`.

## Important model limitation

The delivery coordinates are synthetic points in a 10 km × 10 km plane, not real geocoded Surabaya addresses. Euclidean distance is a simplified proxy for flight distance. The model does not account for buildings, no-fly zones, wind, payload, or battery dynamics.

## How to run

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python tsp_drone_optimization.py
```

When finished, output files are written to the `results/` folder.

## Algorithm details

- **Hill Climbing:** examines every possible contiguous segment reversal and chooses the best strictly improving route until no improving neighbour is found.
- **Simulated Annealing:** randomly reverses a route segment; it always accepts improvements and may accept worse routes with probability `exp(-delta / temperature)`. It returns the best route found during the run. The neighbour operation is a segment reversal (2-opt-style), not a simple swap.
- **Genetic Algorithm:** represents each route as a permutation, uses tournament selection, ordered crossover (OX), swap mutation, and elitism.

The three algorithms use different search budgets/stopping rules, so objective-evaluation counts are informative but not a perfectly equal-budget comparison.
