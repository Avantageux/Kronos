import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from tqdm import tqdm

from model import Kronos, KronosPredictor, KronosTokenizer

TEST_DATA_ROOT = Path(__file__).parent / "data"
INPUT_DATA_PATH = TEST_DATA_ROOT / "regression_input.csv"

# Regression test configuration
OUTPUT_DATA_DIR = TEST_DATA_ROOT
TEST_CTX_LEN = [512, 256]
PRED_LEN = 8
REL_TOLERANCE = 1e-5
FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]

# MSE regression test configuration
MSE_SAMPLE_SIZE = 4
MSE_CTX_LEN = [512, 256]
MSE_EXPECTED = [0.008979, 0.003741]
MSE_PRED_LEN = 30
MSE_TOLERANCE = 0.000001
MSE_FEATURE_NAMES = ["open", "high", "low", "close"]

MODEL_REPO = "NeoQuasar/Kronos-small"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
MAX_CTX_LEN = 512
SEED = 123
DEVICE = "cpu"

# Availability guard: ModelHubMixin.from_pretrained resolves exactly these
# artifacts from the Hub (config + a weights file). Probing them through
# try_to_load_from_cache mirrors that resolution with no network access.
_HUB_PROBE_FILES = ("config.json", "model.safetensors", "pytorch_model.bin")

_SKIP_REASON_COLD_CACHE_OFFLINE = (
    "model weights not present in the local HF cache and network is disabled "
    "(offline CI); regression against pretrained weights is recorded "
    "noncoverage on cold-cache runners"
)


def _is_hf_hub_offline() -> bool:
    """True iff huggingface_hub runs in offline mode (mirrors its env parsing)."""
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "").strip().upper() in {"1", "ON", "YES", "TRUE"}:
            return True
    return False


def _hub_artifacts_resolvable_locally(repo_id: str, revision: str) -> bool:
    """True iff from_pretrained(repo_id, revision=revision) can be served
    entirely from the local HuggingFace cache."""
    resolved = {
        filename: try_to_load_from_cache(
            repo_id=repo_id, filename=filename, revision=revision
        )
        for filename in _HUB_PROBE_FILES
    }
    has_config = isinstance(resolved["config.json"], str)
    has_weights = isinstance(resolved["model.safetensors"], str) or isinstance(
        resolved["pytorch_model.bin"], str
    )
    return has_config and has_weights


def _require_pretrained_weights() -> None:
    """Loud availability guard for the pretrained-weight regression tests.

    When the suite runs offline (HF_HUB_OFFLINE) and the pinned pretrained
    artifacts are not present in the local HuggingFace cache, skip loudly:
    these tests exist to pin regression against pretrained weights, and that
    coverage cannot exist on a cold-cache runner. When the artifacts ARE
    cached (dev machine, or a workflow step that pre-caches them), or when
    network access is allowed, the tests run exactly as before.
    """
    if not _is_hf_hub_offline():
        return
    if _hub_artifacts_resolvable_locally(TOKENIZER_REPO, TOKENIZER_REVISION) and _hub_artifacts_resolvable_locally(
        MODEL_REPO, MODEL_REVISION
    ):
        return
    pytest.skip(_SKIP_REASON_COLD_CACHE_OFFLINE)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.mark.parametrize("context_len", TEST_CTX_LEN)
def test_kronos_predictor_regression(context_len):
    _require_pretrained_weights()
    set_seed(SEED)

    expected_output_path = OUTPUT_DATA_DIR / f"regression_output_{context_len}.csv"
    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    expected_df = pd.read_csv(expected_output_path, parse_dates=["timestamps"])

    if df.shape[0] < context_len + len(expected_df):
        raise ValueError("Example data does not contain enough rows for the regression test.")

    context_df = df.iloc[:context_len].copy()
    context_features = context_df[FEATURE_NAMES].reset_index(drop=True)
    x_timestamp = context_df["timestamps"].reset_index(drop=True)
    future_timestamp = df["timestamps"].iloc[context_len:context_len + len(expected_df)].reset_index(drop=True)
    expected = expected_df[FEATURE_NAMES].values.astype(np.float32)

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)

    with torch.no_grad():
        pred_df = predictor.predict(
            df=context_features,
            x_timestamp=x_timestamp,
            y_timestamp=future_timestamp,
            pred_len=expected.shape[0],
            T=1.0,
            top_k=1,
            top_p=1.0,
            verbose=False,
            sample_count=1,
        )

    obtained = pred_df[FEATURE_NAMES].to_numpy(dtype=np.float32)

    abs_diff = np.abs(obtained - expected)
    rel_diff = abs_diff / (np.abs(expected) + 1e-9)
    print(f"Abs diff: {np.max(abs_diff)}, Rel diff: {np.max(rel_diff)}")

    np.testing.assert_allclose(obtained, expected, rtol=REL_TOLERANCE)

@pytest.mark.parametrize("context_len, expected_mse", zip(MSE_CTX_LEN, MSE_EXPECTED))
def test_kronos_predictor_mse(context_len, expected_mse):
    _require_pretrained_weights()
    set_seed(SEED)

    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    if df.shape[0] <= context_len + MSE_PRED_LEN:
        raise ValueError("Example data does not contain enough rows for the random sample regression test.")

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)

    valid_region = df.iloc[context_len : df.shape[0] - MSE_PRED_LEN]
    if valid_region.shape[0] < MSE_SAMPLE_SIZE:
        raise ValueError("Not enough data points to draw the requested random samples.")

    sampled_rows = valid_region.sample(n=MSE_SAMPLE_SIZE, random_state=SEED).sort_index()

    mse_values = []
    sample_indices = sampled_rows.index.to_list()
    with torch.no_grad():
        for row_idx in tqdm(sample_indices):
            context_slice = df.iloc[row_idx - context_len : row_idx].copy()
            future_slice = df.iloc[row_idx : row_idx + MSE_PRED_LEN].copy()

            pred_df = predictor.predict(
                df=context_slice[FEATURE_NAMES].reset_index(drop=True),
                x_timestamp=context_slice["timestamps"].reset_index(drop=True),
                y_timestamp=future_slice["timestamps"].reset_index(drop=True),
                pred_len=MSE_PRED_LEN,
                T=1.0,
                top_k=1,
                top_p=1.0,
                verbose=False,
                sample_count=1,
            )

            obtained = pred_df[MSE_FEATURE_NAMES].to_numpy(dtype=np.float32)
            expected = future_slice[MSE_FEATURE_NAMES].to_numpy(dtype=np.float32)
            mse_values.append(float(np.mean((obtained - expected) ** 2)))

    assert len(mse_values) == MSE_SAMPLE_SIZE, f"Expected {MSE_SAMPLE_SIZE} MSE values, got {len(mse_values)}."

    mse = np.mean(mse_values).item()
    mse_diff = mse - expected_mse
    print(f"Average MSE: {mse} (Diff vs expected: {mse_diff:+})")

    assert abs(mse_diff) <= MSE_TOLERANCE, f"MSE {mse} differs from expected {expected_mse}"
