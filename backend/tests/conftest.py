import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="causalx_test_")
os.environ["CAUSALX_DATA_DIR"] = os.path.join(_tmp, "data")
os.environ["CAUSALX_ARTIFACTS_DIR"] = os.path.join(_tmp, "artifacts")
os.environ["CAUSALX_DATABASE_URL"] = "sqlite:///" + os.path.join(_tmp, "test.db").replace("\\", "/")

import pytest  # noqa: E402

from app.data.frames import build_frame  # noqa: E402
from app.data.synthetic import GenConfig, generate  # noqa: E402


@pytest.fixture(scope="session")
def tables_rand():
    return generate(GenConfig(n_customers=20000, seed=7, scenario="randomized"))


@pytest.fixture(scope="session")
def tables_obs():
    return generate(GenConfig(n_customers=20000, seed=7, scenario="observational"))


@pytest.fixture(scope="session")
def tables_hidden():
    return generate(GenConfig(n_customers=20000, seed=7, scenario="observational_hidden"))


@pytest.fixture(scope="session")
def frame_rand(tables_rand):
    return build_frame(tables_rand)


@pytest.fixture(scope="session")
def frame_obs(tables_obs):
    return build_frame(tables_obs)


@pytest.fixture(scope="session")
def frame_hidden(tables_hidden):
    return build_frame(tables_hidden)
