"""EEG npz-cache contract: tracked caches load with the
expected shapes/label sets, and the class count equals the CUE_NAMES
constant (K == len(CUE_NAMES[dataset]), never a
data-derived max)."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "eeg_cache"


@pytest.mark.parametrize("dataset,K", [("2a", 4), ("2b", 2)])
def test_cache_subject_shapes_and_labels(dataset, K):
    from dataio.eeg import CUE_NAMES, load_subject

    assert len(CUE_NAMES[dataset]) == K
    for subject in (1, 9):
        path = CACHE / dataset / f"subject{subject}.npz"
        if not path.exists():
            pytest.skip(f"cache missing: {path}")
        out = load_subject(dataset=dataset, subject=subject)
        (X_tr, y_tr), (X_st, y_st) = out["train"], out["stream"]
        assert X_tr.ndim == 3 and X_st.ndim == 3
        assert X_tr.shape[1] == X_st.shape[1]
        assert y_tr.ndim == 1 and y_st.ndim == 1
        assert len(X_tr) == len(y_tr) and len(X_st) == len(y_st)
        # the class count is the CUE_NAMES constant, not max(y)+1
        assert int(max(y_tr.max(), y_st.max()) + 1) == len(CUE_NAMES[dataset])
        assert set(np.unique(y_tr)) == set(range(K))
