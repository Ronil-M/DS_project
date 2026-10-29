import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------------------------
# Fixtures shared by tests/test_module_b.py and tests/test_module_e.py (reduced, CPU-friendly Module C
# configuration, identical to the one used in tests/test_module_c.py). Named b* to avoid clashing with
# the module-scoped fixtures of test_module_c.py.
# ---------------------------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def bcfg():
    from module_c.layout import load_config
    c = load_config(Path(__file__).resolve().parents[1] / "configs" / "module_c.yaml")
    c["run"].update({"resolution": [128, 128], "view_angle": 8, "batch_size": 4, "gpu": None})
    c["layout"]["n_electrodes"] = 150
    c["render"]["n_frames"] = 8
    return c


@pytest.fixture(scope="session")
def bsim(bcfg):
    from module_c.simulator import ProstheticSimulator
    return ProstheticSimulator(bcfg)


@pytest.fixture(scope="session")
def batlas(bsim):
    from module_c.atlas import build_atlas
    return build_atlas(bsim)


@pytest.fixture(scope="session")
def shapes_spec(bsim):
    from datasets.shapes import ShapeSpec
    return ShapeSpec.from_sim(bsim, field_deg=6.0, max_shift_deg=0.5)   # objects must fit the 8 deg test field


@pytest.fixture(scope="session")
def shapes_test(shapes_spec):
    from datasets.shapes import make_dataset
    return make_dataset(6, seed=123, spec=shapes_spec)
