"""Draw the three demo blueprints into samples/.  uv run python generate_samples.py"""
from pathlib import Path

from blueprint.samples import generate_samples

if __name__ == "__main__":
    for p in generate_samples(Path(__file__).parent / "samples"):
        print(f"wrote {p}")
