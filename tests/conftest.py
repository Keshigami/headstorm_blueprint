from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def samples() -> Path:
    from blueprint.samples import generate_samples

    out = HERE / "samples"
    if not all((out / f"{n}.png").exists() for n in ("floor_plan", "elevation", "site_plan")):
        generate_samples(out)
    return out


@pytest.fixture(scope="session")
def trained_head() -> Path:
    """Train a small head if models/classifier.pt is missing (about a minute on CPU)."""
    out = HERE / "models"
    if not (out / "classifier.pt").exists():
        from train import train

        train(per_class=24, epochs=40, out=out)
    return out


@pytest.fixture(scope="session")
def ocr():
    from blueprint.ocr_module import BlueprintOCR

    o = BlueprintOCR()
    if not o.available:
        pytest.skip(f"EasyOCR model unavailable (download not possible?): {o.unavailable_reason}")
    return o
