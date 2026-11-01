"""Chronological splits and streaming-protocol iterators."""

import numpy as np

ETT_BORDERS = {
    "ETTh1": (8640, 2880, 2880),
    "ETTh2": (8640, 2880, 2880),
    "ETTm1": (34560, 11520, 11520),
    "ETTm2": (34560, 11520, 11520),
}


def chronological_split(n: int, ratios=(0.2, 0.05, 0.75)):
    """20/5/75 style chronological train/val/test slices (DSOF convention)."""
    train = int(round(n * ratios[0]))
    val = int(round(n * ratios[1]))
    return slice(0, train), slice(train, train + val), slice(train + val, n)


def ett_split(n: int, name: str):
    """Autoformer border splits for ETT datasets (months 12/4/4 or 60/20/20 min counts)."""
    if name not in ETT_BORDERS:
        raise ValueError(f"no border split for {name}")
    train, val, _ = ETT_BORDERS[name]
    if train + val >= n:
        return chronological_split(n)
    return slice(0, train), slice(train, train + val), slice(train + val, n)


def walk_forward(n: int, train_size: int, test_step: int | None = None, expanding: bool = True):
    """Rolling-origin iterator: yields (train_slice, test_block_slice)."""
    test_step = test_step or max(1, n // 100)
    t = train_size
    while t < n:
        end = min(t + test_step, n)
        train = slice(0, t) if expanding else slice(max(0, t - train_size), t)
        yield train, slice(t, end)
        t = end


def dsof_stream(n: int, lookback: int = 96, horizon: int = 1, block: int = 1, start: int | None = None):
    """Predict-before-update streaming iterator (DSOF protocol).

    Yields (context_slice, target_index): context_slice is the lookback window
    ending at t (exclusive), target_index is the absolute index of y[t+horizon].
    Targets for a block are consumed before ground truth is incorporated.
    """
    start = lookback if start is None else max(start, lookback)
    t = start
    last = n - horizon + 1
    while t < last:
        end = min(t + block, last)
        for k in range(t, end):
            yield slice(k - lookback, k), k + horizon - 1
        t = end


def stream_arrays(n: int, lookback: int = 96, horizon: int = 1, block: int = 1, start: int | None = None) -> np.ndarray:
    """Materialised (n_steps, lookback+1) index table for vectorised streaming."""
    start = lookback if start is None else max(start, lookback)
    last = n - horizon + 1
    steps = np.arange(start, last)
    context = steps[:, None] + np.arange(-lookback, 0)[None, :]
    targets = steps + horizon - 1
    return context, targets
