"""
Optimizing Drone Delivery Routes in Surabaya using Search / Local-Search Algorithms
Course: KKA IUP 2026

Procedural version: no user-defined classes, no dataclasses, no OOP.
Algorithms:
1. Steepest-Improvement Hill Climbing
2. Simulated Annealing
3. Genetic Algorithm (permutation encoding, tournament selection, OX crossover, swap mutation)

The coordinates are synthetic points in a 10 km x 10 km plane, not real Surabaya addresses.
Euclidean distance is a simplified proxy: buildings, no-fly zones, wind, payload and battery
constraints are not modelled.

Usage:
    pip install matplotlib
    python tsp_drone_optimization.py

Outputs are written to ./results/
"""

import csv
import math
import os
import random
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------- Configuration ----------------------------
MASTER_SEED = 20261019
COORDINATE_RANGE_KM = (0.0, 10.0)
DEPOT = (5.0, 5.0)
MAIN_NUM_DELIVERIES = 20
MAIN_RUNS = 20
SCALABILITY_SIZES = [10, 20, 30, 50]
RUNS_PER_SIZE = 10

SA_INITIAL_TEMPERATURE = 25.0
SA_COOLING_RATE = 0.999
SA_MIN_TEMPERATURE = 0.05
SA_MAX_ITERATIONS = 7000

GA_POPULATION_SIZE = 100
GA_GENERATIONS = 300
GA_MUTATION_RATE = 0.15
GA_ELITE_COUNT = 5
GA_TOURNAMENT_SIZE = 4

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Results are dictionaries instead of instances of a custom class.
# Distance data are stored as a dictionary containing points, distance matrix and counter.

def create_distance_data(points):
    """Precompute pairwise Euclidean distances and initialize evaluation counter."""
    matrix = []
    for i in range(len(points)):
        row = []
        for j in range(len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            row.append(math.hypot(dx, dy))
        matrix.append(row)
    return {"points": list(points), "n": len(points), "matrix": matrix,
            "objective_evaluations": 0}


def tour_length(distance_data, route):
    """Closed route length: depot -> deliveries -> depot. Depot is matrix index 0."""
    distance_data["objective_evaluations"] += 1
    if not route:
        return 0.0
    matrix = distance_data["matrix"]
    total = matrix[0][route[0] + 1]
    for a, b in zip(route, route[1:]):
        total += matrix[a + 1][b + 1]
    total += matrix[route[-1] + 1][0]
    return total


def generate_points(n_deliveries, seed):
    """Generate deterministic synthetic delivery coordinates; depot is stored separately."""
    rng = random.Random(seed)
    lo, hi = COORDINATE_RANGE_KM
    points = []
    for _ in range(n_deliveries):
        points.append((rng.uniform(lo, hi), rng.uniform(lo, hi)))
    return points


def validate_route(route, n_deliveries):
    """Every delivery index 0..n-1 must occur exactly once."""
    return len(route) == n_deliveries and sorted(route) == list(range(n_deliveries))


def random_route(n, rng):
    route = list(range(n))
    rng.shuffle(route)
    return route


def two_opt_reverse(route, i, j):
    """Reverse an inclusive segment; this preserves the permutation representation."""
    candidate = list(route)
    candidate[i:j + 1] = reversed(candidate[i:j + 1])
    return candidate


# --------------------------- Algorithm 1 --------------------------------
def hill_climbing(distance_data, rng):
    """Steepest-improvement hill climbing with 2-opt reversal neighbours."""
    n = distance_data["n"] - 1
    current = random_route(n, rng)
    current_distance = tour_length(distance_data, current)

    while True:
        best_route = None
        best_distance = current_distance
        for i in range(n - 1):
            for j in range(i + 1, n):
                neighbour = two_opt_reverse(current, i, j)
                neighbour_distance = tour_length(distance_data, neighbour)
                if neighbour_distance < best_distance - 1e-12:
                    best_distance = neighbour_distance
                    best_route = neighbour
        if best_route is None:
            break
        current = best_route
        current_distance = best_distance
    return current, current_distance


# --------------------------- Algorithm 2 --------------------------------
def simulated_annealing(distance_data, rng):
    """Simulated Annealing with 2-opt reversal neighbours and exponential cooling."""
    n = distance_data["n"] - 1
    current = random_route(n, rng)
    current_distance = tour_length(distance_data, current)
    best_route = current[:]
    best_distance = current_distance
    temperature = SA_INITIAL_TEMPERATURE
    iterations = 0

    while temperature > SA_MIN_TEMPERATURE and iterations < SA_MAX_ITERATIONS:
        i, j = sorted(rng.sample(range(n), 2))
        neighbour = two_opt_reverse(current, i, j)
        neighbour_distance = tour_length(distance_data, neighbour)
        delta = neighbour_distance - current_distance

        # Always accept improvements; sometimes accept worse routes to escape local optima.
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current = neighbour
            current_distance = neighbour_distance
            if current_distance < best_distance:
                best_route = current[:]
                best_distance = current_distance

        temperature *= SA_COOLING_RATE
        iterations += 1
    return best_route, best_distance


# --------------------------- Algorithm 3 --------------------------------
def tournament_selection(population, distances, rng, tournament_size):
    indices = rng.sample(range(len(population)), min(tournament_size, len(population)))
    winner = min(indices, key=lambda idx: distances[idx])
    return list(population[winner])


def ordered_crossover(parent1, parent2, rng):
    """Ordered crossover (OX) for permutation chromosomes."""
    n = len(parent1)
    if n < 2:
        return list(parent1)
    left, right = sorted(rng.sample(range(n), 2))
    child = [-1] * n
    child[left:right + 1] = parent1[left:right + 1]
    genes_from_parent2 = [gene for gene in parent2 if gene not in child]
    fill_positions = [i for i, gene in enumerate(child) if gene == -1]
    for position, gene in zip(fill_positions, genes_from_parent2):
        child[position] = gene
    return child


def genetic_algorithm(distance_data, rng):
    """Genetic Algorithm: tournament selection, OX crossover, swap mutation and elitism."""
    n = distance_data["n"] - 1
    population = [random_route(n, rng) for _ in range(GA_POPULATION_SIZE)]
    distances = [tour_length(distance_data, route) for route in population]

    for _ in range(GA_GENERATIONS):
        ranked_indices = sorted(range(len(population)), key=lambda idx: distances[idx])
        new_population = [population[idx][:] for idx in ranked_indices[:GA_ELITE_COUNT]]
        while len(new_population) < GA_POPULATION_SIZE:
            parent1 = tournament_selection(population, distances, rng, GA_TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, distances, rng, GA_TOURNAMENT_SIZE)
            child = ordered_crossover(parent1, parent2, rng)
            if rng.random() < GA_MUTATION_RATE and n >= 2:
                i, j = rng.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]
            new_population.append(child)
        population = new_population
        distances = [tour_length(distance_data, route) for route in population]

    best_index = min(range(len(population)), key=lambda idx: distances[idx])
    return population[best_index][:], distances[best_index]


ALGORITHMS = {
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing,
    "Genetic Algorithm": genetic_algorithm,
}


# ---------------------------- Experiments -------------------------------
def closed_tour_length(points_with_depot, route):
    """Independent distance calculation; route indices refer to points after the depot."""
    if not route:
        return 0.0
    ordered_points = [points_with_depot[idx + 1] for idx in route]
    first = ordered_points[0]
    total = math.hypot(points_with_depot[0][0] - first[0], points_with_depot[0][1] - first[1])
    for a, b in zip(ordered_points, ordered_points[1:]):
        total += math.hypot(a[0] - b[0], a[1] - b[1])
    last = ordered_points[-1]
    total += math.hypot(last[0] - points_with_depot[0][0], last[1] - points_with_depot[0][1])
    return total


def run_one(algorithm_name, algorithm, delivery_points, seed):
    # Index 0 in the distance matrix is the depot; delivery indices map to points[1:].
    distance_data = create_distance_data([DEPOT] + list(delivery_points))
    rng = random.Random(seed)
    start = time.perf_counter()
    route, distance_km = algorithm(distance_data, rng)
    runtime = time.perf_counter() - start

    if not validate_route(route, len(delivery_points)):
        raise RuntimeError(f"{algorithm_name} returned an invalid route: {route}")

    # Independent recomputation checks correctness; this check is not counted as an evaluation.
    check_distance = closed_tour_length([DEPOT] + list(delivery_points), route)
    if not math.isclose(distance_km, check_distance, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(f"Distance mismatch for {algorithm_name}: {distance_km} != {check_distance}")

    return {
        "algorithm": algorithm_name,
        "n_deliveries": len(delivery_points),
        "seed": seed,
        "distance_km": distance_km,
        "runtime_seconds": runtime,
        "objective_evaluations": distance_data["objective_evaluations"],
        "route": route,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(results):
    summary = []
    for name in ALGORITHMS:
        group = [result for result in results if result["algorithm"] == name]
        distances = [result["distance_km"] for result in group]
        times = [result["runtime_seconds"] for result in group]
        evaluations = [result["objective_evaluations"] for result in group]
        summary.append({
            "algorithm": name,
            "n_deliveries": group[0]["n_deliveries"],
            "runs": len(group),
            "best_distance_km": min(distances),
            "mean_distance_km": statistics.mean(distances),
            "std_distance_km": statistics.stdev(distances) if len(distances) > 1 else 0.0,
            "mean_runtime_seconds": statistics.mean(times),
            "std_runtime_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
            "mean_objective_evaluations": statistics.mean(evaluations),
        })
    return summary


def save_route_plot(delivery_points, result_by_algorithm, filename, title_suffix=""):
    fig, axes = plt.subplots(1, len(result_by_algorithm), figsize=(6 * len(result_by_algorithm), 5))
    if len(result_by_algorithm) == 1:
        axes = [axes]
    for ax, (name, result) in zip(axes, result_by_algorithm.items()):
        route_points = [DEPOT] + [delivery_points[i] for i in result["route"]] + [DEPOT]
        xs = [p[0] for p in route_points]
        ys = [p[1] for p in route_points]
        ax.plot(xs, ys, marker="o", linewidth=1.4)
        ax.scatter([DEPOT[0]], [DEPOT[1]], marker="s", s=100, label="Depot")
        ax.scatter([p[0] for p in delivery_points], [p[1] for p in delivery_points],
                   s=28, label="Delivery points")
        ax.set_title(f"{name}\nDistance = {result['distance_km']:.2f} km")
        ax.set_xlabel("X coordinate (km)")
        ax.set_ylabel("Y coordinate (km)")
        ax.set_xlim(COORDINATE_RANGE_KM)
        ax.set_ylim(COORDINATE_RANGE_KM)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Best routes from recorded experimental runs {title_suffix}")
    fig.tight_layout()
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scalability_plot(summaries_by_size):
    # Mean route length vs problem size.
    plt.figure(figsize=(8, 5))
    for algorithm_name in ALGORITHMS:
        sizes, means, stds = [], [], []
        for n in sorted(summaries_by_size):
            row = next(row for row in summaries_by_size[n] if row["algorithm"] == algorithm_name)
            sizes.append(n)
            means.append(row["mean_distance_km"])
            stds.append(row["std_distance_km"])
        plt.errorbar(sizes, means, yerr=stds, marker="o", capsize=4, label=algorithm_name)
    plt.xlabel("Number of delivery points")
    plt.ylabel("Mean route distance (km)")
    plt.title("Solution quality vs. problem size (error bars = 1 standard deviation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scalability_distance.png"), dpi=180)
    plt.close()

    # Mean runtime vs problem size.
    plt.figure(figsize=(8, 5))
    for algorithm_name in ALGORITHMS:
        sizes, times = [], []
        for n in sorted(summaries_by_size):
            row = next(row for row in summaries_by_size[n] if row["algorithm"] == algorithm_name)
            sizes.append(n)
            times.append(row["mean_runtime_seconds"])
        plt.plot(sizes, times, marker="o", label=algorithm_name)
    plt.xlabel("Number of delivery points")
    plt.ylabel("Mean runtime (seconds)")
    plt.title("Runtime vs. problem size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scalability_runtime.png"), dpi=180)
    plt.close()

    # Objective function evaluations vs size.
    plt.figure(figsize=(8, 5))
    for algorithm_name in ALGORITHMS:
        sizes, evaluations = [], []
        for n in sorted(summaries_by_size):
            row = next(row for row in summaries_by_size[n] if row["algorithm"] == algorithm_name)
            sizes.append(n)
            evaluations.append(row["mean_objective_evaluations"])
        plt.plot(sizes, evaluations, marker="o", label=algorithm_name)
    plt.xlabel("Number of delivery points")
    plt.ylabel("Mean objective-function evaluations")
    plt.title("Search effort vs. problem size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scalability_evaluations.png"), dpi=180)
    plt.close()


def run_experiment(n_deliveries, runs, dataset_seed, seed_offset=0):
    delivery_points = generate_points(n_deliveries, dataset_seed)
    all_results = []
    print(f"\nRunning {runs} runs per algorithm for {n_deliveries} delivery points...")
    for algorithm_index, (name, algorithm) in enumerate(ALGORITHMS.items()):
        for run_index in range(runs):
            seed = MASTER_SEED + seed_offset + algorithm_index * 100_000 + run_index
            result = run_one(name, algorithm, delivery_points, seed)
            all_results.append(result)
            print(f"  {name:20s} run {run_index + 1:02d}/{runs}: "
                  f"{result['distance_km']:8.3f} km, "
                  f"{result['runtime_seconds']:7.3f} s, "
                  f"{result['objective_evaluations']:9d} evaluations")
    return all_results


def main():
    print("=" * 78)
    print("DRONE DELIVERY TSP: HILL CLIMBING vs SIMULATED ANNEALING vs GENETIC ALGORITHM")
    print("=" * 78)
    print(f"Depot: {DEPOT}")
    print(f"Main dataset: {MAIN_NUM_DELIVERIES} delivery points")
    print(f"Main comparison runs per algorithm: {MAIN_RUNS}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    # Main comparison.
    main_points = generate_points(MAIN_NUM_DELIVERIES, MASTER_SEED)
    main_results = run_experiment(MAIN_NUM_DELIVERIES, MAIN_RUNS,
                                 dataset_seed=MASTER_SEED, seed_offset=0)
    main_summary = summarize_results(main_results)

    write_csv(os.path.join(OUTPUT_DIR, "main_run_results.csv"), [
        {
            "algorithm": r["algorithm"],
            "n_deliveries": r["n_deliveries"],
            "seed": r["seed"],
            "distance_km": f"{r['distance_km']:.6f}",
            "runtime_seconds": f"{r['runtime_seconds']:.6f}",
            "objective_evaluations": r["objective_evaluations"],
            "route_delivery_indices": " ".join(map(str, r["route"])),
        } for r in main_results
    ], ["algorithm", "n_deliveries", "seed", "distance_km", "runtime_seconds",
        "objective_evaluations", "route_delivery_indices"])
    write_csv(os.path.join(OUTPUT_DIR, "main_summary.csv"), main_summary,
              list(main_summary[0].keys()))

    # Best route observed in the recorded runs for each algorithm.
    best_results = {}
    for name in ALGORITHMS:
        group = [r for r in main_results if r["algorithm"] == name]
        best_results[name] = min(group, key=lambda r: r["distance_km"])
    save_route_plot(main_points, best_results,
                    os.path.join(OUTPUT_DIR, "main_best_routes.png"),
                    title_suffix=f"({MAIN_NUM_DELIVERIES} delivery points)")

    # Dataset plot without routes.
    plt.figure(figsize=(7, 6))
    plt.scatter([p[0] for p in main_points], [p[1] for p in main_points],
                marker="o", label="Delivery points")
    plt.scatter([DEPOT[0]], [DEPOT[1]], marker="s", s=130, label="Depot")
    for idx, (x, y) in enumerate(main_points, start=1):
        plt.annotate(str(idx), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("X coordinate (km)")
    plt.ylabel("Y coordinate (km)")
    plt.title("Synthetic drone-delivery dataset")
    plt.xlim(COORDINATE_RANGE_KM)
    plt.ylim(COORDINATE_RANGE_KM)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dataset_points.png"), dpi=180)
    plt.close()

    # Scalability experiment, including the main size for direct comparison.
    summaries_by_size = {}
    for size_index, n in enumerate(SCALABILITY_SIZES):
        if n == MAIN_NUM_DELIVERIES:
            results = main_results
        else:
            results = run_experiment(n, RUNS_PER_SIZE,
                                     dataset_seed=MASTER_SEED + 10_000 + n,
                                     seed_offset=1_000_000 + size_index * 100_000)
        summary = summarize_results(results)
        summaries_by_size[n] = summary
        write_csv(os.path.join(OUTPUT_DIR, f"scalability_summary_{n}_points.csv"),
                  summary, list(summary[0].keys()))

    save_scalability_plot(summaries_by_size)

    print("\nMAIN EXPERIMENT SUMMARY")
    print("-" * 112)
    print(f"{'Algorithm':20s} {'Best km':>10s} {'Mean km':>10s} {'SD km':>10s} "
          f"{'Mean sec':>12s} {'Mean evals':>14s}")
    for row in main_summary:
        print(f"{row['algorithm']:20s} {row['best_distance_km']:10.3f} "
              f"{row['mean_distance_km']:10.3f} {row['std_distance_km']:10.3f} "
              f"{row['mean_runtime_seconds']:12.4f} {row['mean_objective_evaluations']:14.1f}")

    print("\nSCALABILITY SUMMARY")
    for n in sorted(summaries_by_size):
        print(f"\nDelivery points: {n}")
        for row in summaries_by_size[n]:
            print(f"  {row['algorithm']:20s} mean={row['mean_distance_km']:.3f} km, "
                  f"SD={row['std_distance_km']:.3f}, "
                  f"time={row['mean_runtime_seconds']:.4f} s")

    print("\nFinished. Files saved in:", os.path.abspath(OUTPUT_DIR))
    print("Do not claim one algorithm is best until you interpret the generated results.")


if __name__ == "__main__":
    main()
