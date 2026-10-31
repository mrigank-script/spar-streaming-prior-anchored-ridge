"""Resumable parallel downloader for BCI IV 2b (BNCI2014_004) .mat files.

moabb's single-stream pooch downloads fail/under-report on the slow TU Graz
server, so this writes the 18 files straight into moabb's destination folder;
moabb then trusts local files on later loads.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import time

import requests


def _mne_data_dir() -> Path:
    """MNE data dir: the MNE_DATA env var / mne config, default ~/mne_data
    (same convention as scripts/download_eeg.py)."""
    root = os.environ.get("MNE_DATA")
    if not root:
        try:
            import mne
        except ImportError:
            root = None
        else:
            root = mne.get_config("MNE_DATA")
    return Path(root).expanduser() if root else Path.home() / "mne_data"


DEST = _mne_data_dir() / "MNE-bnci-data" / "~bci" / "database" / "004-2014"
BASE = "https://lampx.tugraz.at/~bci/database/004-2014"
FILES = [f"B{s:02d}{r}.mat" for s in range(1, 10) for r in ("T", "E")]
WORKERS = 3
RETRIES = 10
CHUNK = 1 << 20
TIMEOUT = (30, 180)


def remote_size(url: str) -> int:
    response = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    return int(response.headers.get("Content-Length", 0))


def download_one(name: str):
    url = f"{BASE}/{name}"
    dest = DEST / name
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            total = remote_size(url)
            have = dest.stat().st_size if dest.exists() else 0
            if total and have == total:
                return name, have, "skip"
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as response:
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {response.status_code}")
                mode = "ab" if have and response.status_code == 206 else "wb"
                with open(dest, mode) as handle:
                    for chunk in response.iter_content(CHUNK):
                        if chunk:
                            handle.write(chunk)
            size = dest.stat().st_size
            if total and size != total:
                raise RuntimeError(f"size {size} != expected {total}")
            return name, size, f"ok (attempt {attempt})"
        except Exception as exc:
            last_error = exc
            print(f"[2b] {name} attempt {attempt}/{RETRIES} failed: {exc}", flush=True)
            time.sleep(3)
    size = dest.stat().st_size if dest.exists() else 0
    return name, size, f"FAILED ({last_error})"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, name): name for name in FILES}
        for future in as_completed(futures):
            name, size, status = future.result()
            completed += 1
            print(f"[2b] ({completed}/{len(FILES)}) {name}: {size / 1e6:.1f} MB {status}", flush=True)
    total = sum((DEST / name).stat().st_size for name in FILES if (DEST / name).exists())
    ok = sum(1 for name in FILES if (DEST / name).exists())
    print(f"[2b] done: {ok}/{len(FILES)} files, {total / 1e6:.1f} MB -> {DEST}")


if __name__ == "__main__":
    main()
