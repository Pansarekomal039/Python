#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS_by_SA-MS2413.py
Traveling Salesman by Simulated Annealing (SA) with path tracing.

Reads city data from a text file named "India_cities.txt" located in the same
directory. The file may be in one of these formats (header optional):

1) CSV (or whitespace) with latitude/longitude (degrees):
   City,Lat,Lon
   Mumbai,19.0760,72.8777
   Pune,18.5204,73.8567
   ...

2) CSV (or whitespace) with planar coordinates:
   City,X,Y
   A,12.3,45.6
   B,78.9,10.1
   ...

The script automatically detects whether the numbers look like Indian lat/lon
(roughly Lat in [5, 37.5], Lon in [68, 97.5]) and uses haversine distance in km;
otherwise it uses Euclidean distance in the same units as provided.

Usage (defaults will "just work"):
    python TS_by_SA-MS24xx.py
Optional arguments:
    --file India_cities.txt      # path to input (default: India_cities.txt)
    --iters 200000               # SA iterations
    --t0 1.0                     # initial temperature (auto-scaled if 0)
    --alpha 0.9995               # cooling rate per iteration
    --seed 42                    # RNG seed
    --plot True                  # whether to show the final plot
    --save_figure tsp_path.png   # filename to save the plot (omit to skip)

The final path is printed (in order) with total distance, and a matplotlib
window shows the route with city names.
"""

import argparse
import math
import os
import random
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt


# ---------------------------- Parsing Utilities ---------------------------- #

def _try_float(s: str):
    try:
        return float(s)
    except Exception:
        return None


def load_cities(path: str) -> Tuple[List[str], List[Tuple[float, float]], bool]:
    """
    Load cities from a text file.
    Returns (names, coords, is_geo) where:
      - names: list of city names (strings)
      - coords: list of (x, y) pairs (floats)
      - is_geo: True if coords look like (lat, lon) in India
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    names: List[str] = []
    coords: List[Tuple[float, float]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # skip blanks/comments
                continue

            # Split on comma first, else whitespace
            parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
            if len(parts) < 3:
                # allow header lines like: City,Lat,Lon
                lower = [p.lower() for p in parts]
                if any(h in lower for h in ("city", "lat", "lon", "x", "y")):
                    continue
                # otherwise ignore malformed
                continue

            name = parts[0]
            x = _try_float(parts[1])
            y = _try_float(parts[2])
            if x is None or y is None:
                continue

            names.append(name)
            coords.append((x, y))

    if len(names) < 3:
        raise ValueError("Need at least 3 valid cities in the input file.")

    # Heuristic: detect India-like lat/lon
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    is_geo = (5.0 <= lat_min <= 37.5) and (5.0 <= lat_max <= 37.5) \
             and (68.0 <= lon_min <= 97.5) and (68.0 <= lon_max <= 97.5)

    return names, coords, is_geo

def load_cities_without_coords(path: str) -> Tuple[List[str], List[Tuple[float, float]], bool]:
    """Load cities from file without coordinates and generate random coordinates."""
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Remove numbering like "1. ", "2. ", etc.
            if line[0].isdigit() and '.' in line:
                line = line.split('.', 1)[1].strip()
            
            names.append(line)
    
    if len(names) < 3:
        raise ValueError("Need at least 3 valid cities in the input file.")
    
    # Generate random coordinates for demonstration
    random.seed(42)  # For reproducible results
    coords = []
    for i in range(len(names)):
        # Generate coordinates roughly in India region
        lat = random.uniform(8.0, 37.0)
        lon = random.uniform(68.0, 97.0)
        coords.append((lat, lon))
    
    return names, coords, True


# ---------------------------- Distance Functions --------------------------- #

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points on Earth in kilometers."""
    R = 6371.0088  # mean Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def build_distance_fn(coords: List[Tuple[float, float]], is_geo: bool):
    if is_geo:
        def dist(i: int, j: int) -> float:
            a = coords[i]; b = coords[j]
            return haversine_km(a[0], a[1], b[0], b[1])
    else:
        def dist(i: int, j: int) -> float:
            a = coords[i]; b = coords[j]
            dx = a[0] - b[0]; dy = a[1] - b[1]
            return math.hypot(dx, dy)
    return dist


# ---------------------------- TSP Utilities -------------------------------- #

def tour_length(tour: List[int], dist) -> float:
    total = 0.0
    n = len(tour)
    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]
        total += dist(i, j)
    return total


def initial_tour(n: int) -> List[int]:
    t = list(range(n))
    random.shuffle(t)
    return t


def two_opt_move(tour: List[int]) -> Tuple[int, int]:
    """Pick two distinct indices i<j and reverse the segment [i, j]."""
    n = len(tour)
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    return i, j


def delta_two_opt(tour: List[int], i: int, j: int, dist) -> float:
    """
    Compute change in tour length if we reverse the segment [i, j].
    Uses 2-opt edge difference without recomputing full length.
    """
    n = len(tour)
    a, b = tour[(i - 1) % n], tour[i]
    c, d = tour[j], tour[(j + 1) % n]

    before = dist(a, b) + dist(c, d)
    after = dist(a, c) + dist(b, d)
    return after - before


def apply_two_opt(tour: List[int], i: int, j: int) -> None:
    """In-place reverse of segment [i, j]."""
    tour[i:j+1] = reversed(tour[i:j+1])


# ---------------------------- Simulated Annealing -------------------------- #

def simulated_annealing(dist, n: int, iters: int = 200000, t0: float = 0.0, alpha: float = 0.9995,
                        seed: int = 42, verbose_every: int = 20000) -> Tuple[List[int], float]:
    random.seed(seed)
    tour = initial_tour(n)
    best = tour[:]
    curr_len = tour_length(tour, dist)
    best_len = curr_len

    # Auto scale initial temperature if not provided:
    if t0 <= 0.0:
        # Sample a few random moves to estimate typical delta
        samples = []
        for _ in range(min(200, n * 5)):
            i, j = two_opt_move(tour)
            d = abs(delta_two_opt(tour, i, j, dist))
            if d > 0:
                samples.append(d)
        typical = (sum(samples) / len(samples)) if samples else (curr_len / n)
        t = typical  # start around typical improving/worsening move
    else:
        t = t0

    for step in range(1, iters + 1):
        i, j = two_opt_move(tour)
        dE = delta_two_opt(tour, i, j, dist)

        if dE < 0 or random.random() < math.exp(-dE / max(t, 1e-12)):
            # accept
            apply_two_opt(tour, i, j)
            curr_len += dE
            if curr_len < best_len:
                best = tour[:]
                best_len = curr_len

        t *= alpha

        if verbose_every and step % verbose_every == 0:
            print(f"[{step}/{iters}] T={t:.6f} curr={curr_len:.3f} best={best_len:.3f}")

    return best, best_len


# ---------------------------- Plotting ------------------------------------- #

def plot_tour(names: List[str], coords: List[Tuple[float, float]], tour: List[int],
              is_geo: bool, save_path: str = None, show: bool = True) -> None:
    xs = [coords[i][1] if is_geo else coords[i][0] for i in tour]  # if geo, use Lon on x-axis
    ys = [coords[i][0] if is_geo else coords[i][1] for i in tour]  # if geo, use Lat on y-axis

    # Close the loop
    xs.append(xs[0])
    ys.append(ys[0])

    plt.figure(figsize=(10, 8))
    plt.plot(xs, ys, marker='o', linewidth=1.5, markersize=6)

    # Annotate each city with its name (slightly offset)
    for idx, city_idx in enumerate(tour):
        x = coords[city_idx][1] if is_geo else coords[city_idx][0]
        y = coords[city_idx][0] if is_geo else coords[city_idx][1]
        plt.text(x, y, names[city_idx], fontsize=9, ha='right', va='bottom')

    plt.title("TSP Route (Simulated Annealing)")
    plt.xlabel("Longitude" if is_geo else "X")
    plt.ylabel("Latitude" if is_geo else "Y")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------- Main ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Traveling Salesman by Simulated Annealing with path tracing.")
    parser.add_argument("--file", type=str, default="India_cities.txt", help="Input file with cities (default: India_cities.txt)")
    parser.add_argument("--iters", type=int, default=200000, help="Number of SA iterations (default: 200000)")
    parser.add_argument("--t0", type=float, default=0.0, help="Initial temperature (0 for auto-scale)")
    parser.add_argument("--alpha", type=float, default=0.9995, help="Cooling rate per iteration (default: 0.9995)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--plot", type=str, default="True", help="Show plot at the end: True/False (default: True)")
    parser.add_argument("--save_figure", type=str, default="", help="If set, save the figure to this file (e.g., tsp_path.png)")

    args = parser.parse_args()

    try:
        # Try regular format first
        names, coords, is_geo = load_cities(args.file)
    except Exception as e:
        print(f"Regular format failed: {e}. Trying alternative format...")
        try:
            names, coords, is_geo = load_cities_without_coords(args.file)
            print("Using generated coordinates for demonstration")
        except Exception as e2:
            print(f"Error loading cities: {e2}")
            sys.exit(1)
            

    dist = build_distance_fn(coords, is_geo)

    print(f"Loaded {len(names)} cities from '{args.file}'. Using {'haversine (km)' if is_geo else 'euclidean'} distance.")
    best, best_len = simulated_annealing(
        dist, n=len(names), iters=args.iters, t0=args.t0, alpha=args.alpha, seed=args.seed
    )

    # Print final path with city names
    print("\nBest tour order (in visiting sequence):")
    for idx in best:
        print(names[idx])
    print(names[best[0]], "(return to start)")

    unit = "km" if is_geo else "units"
    print(f"\nTotal distance ≈ {best_len:.3f} {unit}")

    # Plot
    show_plot = str(args.plot).strip().lower() in ("1", "true", "yes", "y")
    save_path = args.save_figure if args.save_figure else None
    plot_tour(names, coords, best, is_geo, save_path=save_path, show=show_plot)


if __name__ == "__main__":
    main()
