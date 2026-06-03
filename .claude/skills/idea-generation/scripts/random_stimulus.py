#!/usr/bin/env python3
"""Random stimulus generator for idea-generation sessions.

Genuine randomness is the whole point: when you hand-pick a "random" word it tends to
be a near-neighbour of the problem, which kills the remote-association effect the
creativity literature relies on. This script draws from fixed pools with the stdlib RNG
so the stimulus is actually distant.

Pools intentionally lean concrete and far-from-finance (words, analogy domains). The
morphological matrix and primitive pools mirror references/domain_seeds.md so sampled
cells are tradable by construction.

Usage (run with the project interpreter, see CLAUDE.md / AGENTS.md):
    python scripts/random_stimulus.py                 # a mixed seed pack
    python scripts/random_stimulus.py --words 5       # 5 remote stimulus words
    python scripts/random_stimulus.py --analogy       # 1 base domain for structural analogy
    python scripts/random_stimulus.py --matrix        # 1 random strategy morphological cell
    python scripts/random_stimulus.py --triple        # 3 signal primitives to force-combine
    python scripts/random_stimulus.py --seed 42       # reproducible draw

Stdlib only; no dependencies.
"""
from __future__ import annotations

import argparse
import random

WORDS = [
    "tide", "anthill", "origami", "glacier", "lighthouse", "fermentation", "migration",
    "scaffolding", "echo", "mycelium", "thermostat", "avalanche", "tournament", "compost",
    "harbour", "calligraphy", "vaccine", "drought", "orchestra", "ratchet", "estuary",
    "pendulum", "wildfire", "loom", "coral", "quarantine", "flywheel", "monsoon", "relay",
    "sediment", "beehive", "tuning fork", "watershed", "pilgrimage", "trellis", "eclipse",
    "ferry", "crystallisation", "stampede", "aqueduct", "molt", "keystone", "undertow",
    "gearbox", "pollination", "smelting", "checkpoint", "thaw", "spiderweb", "ballast",
    "cascade", "germination", "dam", "metronome", "swarm", "pressure cooker", "tributary",
    "hibernation", "lever", "contagion", "tapestry", "sluice", "rookery", "kiln",
    "circuit breaker", "drift", "spawning", "turnstile", "canopy", "sandbar", "bellows",
    "quench", "foothold", "rip current", "scaffold", "auction", "cairn", "siphon",
    "wavefront", "queue", "resonance", "tinder", "mooring", "feedback howl", "ember",
    "watermark", "switchback", "calving", "headwind", "crosswind", "ricochet", "fulcrum",
    "trade wind", "spillway", "moult", "lattice", "tremor", "valve", "convection", "eddy",
]

ANALOGY_DOMAINS = [
    "biology / evolution", "an ant colony", "the immune system", "a city's traffic system",
    "an electrical grid", "a sports team's tactics", "a jazz ensemble", "ecology / predator-prey",
    "epidemiology / contagion", "military logistics", "a coral reef", "weather / fluid dynamics",
    "a postal/parcel network", "a forest ecosystem", "thermodynamics / heat flow",
    "a beehive's foraging", "geology / erosion", "a power-plant control room",
    "a river delta forming", "an auction house", "a queueing system at a port",
    "neural signalling in the brain", "a flock of starlings (murmuration)",
    "a supply chain under shock", "a dam and reservoir", "plate tectonics",
]

# Mirrors the morphological matrix in references/domain_seeds.md.
MATRIX = {
    "Universe": [
        "US sector ETFs", "Country ETFs", "Crypto perps (BTC/ETH)", "60-40 sleeves",
        "Managed futures (DBMF)", "Commodities (BCOM)", "FX / rates",
    ],
    "Data source": ["yfinance", "fred", "binance_vision", "polymarket", "local"],
    "Signal": [
        "TS momentum / trend", "XS momentum (relative strength)", "short-term reversal",
        "carry (funding / roll / rate diff)", "value / mean-reversion",
        "volatility / vol-target", "seasonality (ToD / DoW)", "cross-asset lead-lag",
        "open-interest / positioning",
    ],
    "Horizon": ["intraday (5m)", "1 day", "1 week", "1-3 months", "6-12 months"],
    "Conditioner": [
        "vol high/low", "trend vs chop", "bull/bear", "rates up/down", "risk-on/off",
        "dispersion high/low", "funding extreme", "credit-spread regime", "time-of-day",
        "day-of-week", "pre/post scheduled event",
    ],
    "Portfolio": [
        "long-only", "dollar-neutral L/S", "top/bottom-N", "inverse-vol", "mean-variance",
        "risk parity", "vol-target", "overlay / sizing",
    ],
    "Timing": [
        "calendar (ME / WE)", "vol-triggered", "event-triggered", "breakout-triggered",
        "always-on (daily)", "turn-of-month",
    ],
}

PRIMITIVES = [
    "TS momentum", "XS momentum", "short-term reversal", "funding carry", "roll-yield carry",
    "value / mean-reversion", "realised vol", "vol risk premium", "time-of-day seasonality",
    "day-of-week seasonality", "cross-asset lead-lag", "open interest / positioning",
    "yield-curve slope", "credit spread", "dispersion", "inverse-vol weighting",
    "prediction-market probability", "gap behaviour", "trend vs chop regime", "turnover",
]


def draw_words(rng: random.Random, n: int) -> list[str]:
    return rng.sample(WORDS, min(n, len(WORDS)))


def draw_matrix_cell(rng: random.Random) -> dict[str, str]:
    return {col: rng.choice(vals) for col, vals in MATRIX.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Random stimulus for idea-generation sessions.")
    p.add_argument("--words", type=int, metavar="N", help="draw N remote stimulus words")
    p.add_argument("--analogy", action="store_true", help="draw one analogy base domain")
    p.add_argument("--matrix", action="store_true", help="draw one strategy morphological cell")
    p.add_argument("--triple", action="store_true", help="draw three primitives to force-combine")
    p.add_argument("--seed", type=int, default=None, help="seed for a reproducible draw")
    args = p.parse_args()

    rng = random.Random(args.seed)  # system entropy when seed is None
    specific = args.words or args.analogy or args.matrix or args.triple

    if args.words:
        print("Remote stimulus words:")
        for w in draw_words(rng, args.words):
            print(f"  - {w}")
    if args.analogy:
        print(f"Analogy base domain:  {rng.choice(ANALOGY_DOMAINS)}")
    if args.matrix:
        print("Morphological cell:")
        for col, val in draw_matrix_cell(rng).items():
            print(f"  {col:12s}: {val}")
    if args.triple:
        print("Force-combine these three primitives (don't justify yet):")
        for prim in rng.sample(PRIMITIVES, 3):
            print(f"  - {prim}")

    if not specific:
        # Default: a mixed seed pack to kick off a session.
        print("=== idea-generation seed pack ===\n")
        print("Remote stimulus words:")
        for w in draw_words(rng, 3):
            print(f"  - {w}")
        print(f"\nAnalogy base domain:  {rng.choice(ANALOGY_DOMAINS)}")
        print("\nMorphological cell:")
        for col, val in draw_matrix_cell(rng).items():
            print(f"  {col:12s}: {val}")
        print("\nForce-combine these three primitives (don't justify yet):")
        for prim in rng.sample(PRIMITIVES, 3):
            print(f"  - {prim}")


if __name__ == "__main__":
    main()
