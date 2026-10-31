"""BCI IV 2a/2b loaders: cross-session splits with epoched arrays via moabb/mne."""

import os
from pathlib import Path

import numpy as np

DATASET_CLASSES = {"2a": "BNCI2014_001", "2b": "BNCI2014_004"}
CUE_NAMES = {"2a": ["left_hand", "right_hand", "feet", "tongue"], "2b": ["left_hand", "right_hand"]}
CUE_CODES = {"2a": [769, 770, 771, 772], "2b": [769, 770]}
TRAIN_SESSIONS = {"2a": ["0train"], "2b": ["0train", "1train", "2train"]}
STREAM_SESSIONS = {"2a": ["1test"], "2b": ["3test", "4test"]}

_CACHE_ROOT = Path(__file__).resolve().parents[1] / "data" / "eeg_cache"


def _data_root() -> Path:
    """MNE data dir: the MNE_DATA env var / mne config, default ~/mne_data
    (same convention as scripts/download_eeg.py)."""
    root = os.environ.get("MNE_DATA")
    if not root:
        import mne

        root = mne.get_config("MNE_DATA")
    return Path(root).expanduser() if root else Path.home() / "mne_data"


def _recover_mangled() -> None:
    """Self-heal moabb's Windows path-sanitization bug: it may move raw files
    into a relative 'C-\\Users\\...' directory inside the working directory."""
    mangled_root = Path.cwd() / "C-"
    src = mangled_root / "Users" / Path.home().name / "mne_data" / "MNE-bnci-data" / "~bci" / "database"
    if not src.exists():
        return
    for folder in src.iterdir():
        if not folder.is_dir():
            continue
        target_dir = _data_root() / folder.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            if item.is_file():
                target = target_dir / item.name
                if not target.exists():
                    item.replace(target)
    import shutil
    shutil.rmtree(mangled_root, ignore_errors=True)


def _ensure_local_files(dataset: str, subject: int) -> None:
    """Guard against moabb auto-downloads (its path sanitizer dumps files into a
    relative 'C-\\' directory on Windows); require pre-downloaded raw files."""
    _recover_mangled()
    folder = _data_root() / ("001-2014" if dataset == "2a" else "004-2014")
    prefix = "A" if dataset == "2a" else "B"
    missing = [
        f"{prefix}{subject:02d}{r}.mat"
        for r in ("T", "E")
        if not (folder / f"{prefix}{subject:02d}{r}.mat").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing local files {missing} in {folder}; run scripts/download_eeg_manual.py"
        )


def _canonical_label(key: str, dataset: str):
    name = key.strip().lower().replace(" ", "_")
    if name in CUE_NAMES[dataset]:
        return CUE_NAMES[dataset].index(name)
    if key.isdigit():
        code = int(key)
        if code in CUE_CODES[dataset]:
            return CUE_CODES[dataset].index(code)
    return None


def _epoch_raw(raw, dataset: str, tmin: float, tmax: float, picks: str):
    import mne

    events, annotations = mne.events_from_annotations(raw, verbose="ERROR")
    event_id = {}
    labels = {}
    for key, code in annotations.items():
        label = _canonical_label(key, dataset)
        if label is not None:
            event_id[key] = code
            labels[code] = label
    if not event_id:
        raise RuntimeError(f"no cue annotations found; available: {list(annotations)[:10]}")
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose="ERROR",
    )
    X = epochs.get_data(copy=True)
    y = np.array([labels[int(code)] for code in epochs.events[:, 2]], dtype=np.int64)
    if picks == "eeg":
        eeg_index = mne.pick_types(epochs.info, meg=False, eeg=True)
        X = X[:, eeg_index, :]
    return X.astype(np.float32), y


def load_subject(dataset: str = "2a", subject: int = 1, tmin: float = 0.5, tmax: float = 2.5, picks: str = "eeg") -> dict:
    """Epoch one subject; returns {'train': (X, y), 'stream': (X, y), 'sessions': {...}}.

    Uses the pre-extracted npz cache when available, otherwise the locally
    downloaded raw files (no moabb auto-downloads).
    """
    if dataset not in DATASET_CLASSES:
        raise ValueError(f"unknown dataset {dataset}")
    cache = _CACHE_ROOT / dataset / f"subject{subject}.npz"
    if cache.exists():
        data = np.load(cache)
        X_train, y_train = data["X_train"], data["y_train"]
        X_stream, y_stream = data["X_stream"], data["y_stream"]
        return {
            "train": (X_train, y_train),
            "stream": (X_stream, y_stream),
            "sessions": {"cached": (X_train.shape, X_stream.shape)},
        }
    _ensure_local_files(dataset, subject)
    import moabb.datasets as moabb_datasets

    cls = getattr(moabb_datasets, DATASET_CLASSES[dataset])
    data = cls().get_data(subjects=[subject])
    sessions = data[subject]
    per_session = {}
    for session, runs in sessions.items():
        Xs, ys = [], []
        for raw in runs.values():
            X, y = _epoch_raw(raw, dataset, tmin, tmax, picks)
            Xs.append(X)
            ys.append(y)
        per_session[session] = (np.concatenate(Xs), np.concatenate(ys))
    train_keys, stream_keys = TRAIN_SESSIONS[dataset], STREAM_SESSIONS[dataset]
    X_train = np.concatenate([per_session[k][0] for k in train_keys])
    y_train = np.concatenate([per_session[k][1] for k in train_keys])
    X_stream = np.concatenate([per_session[k][0] for k in stream_keys])
    y_stream = np.concatenate([per_session[k][1] for k in stream_keys])
    return {
        "train": (X_train, y_train),
        "stream": (X_stream, y_stream),
        "sessions": {k: tuple(v[i].shape for i in (0, 1)) for k, v in per_session.items()},
    }
