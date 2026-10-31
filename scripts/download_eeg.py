"""Download the two small EEG sets via moabb: BCI IV 2a (BNCI2014_001) and 2b (BNCI2014_004).

GB-scale sets (PhysioNet EEGMMIDB, OpenBMI, Sleep-EDF) are skipped by design.
Files land in the MNE data directory (default ~/mne_data).
"""

import time

import mne
from moabb.datasets import BNCI2014_001, BNCI2014_004

MAX_ATTEMPTS = 12


def download_with_retries(dataset, tag: str) -> bool:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            dataset.download()
            print(f"[eeg] BCI IV {tag} done (attempt {attempt})")
            return True
        except Exception as exc:
            print(f"[eeg] {tag} attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}")
            time.sleep(5)
    print(f"[eeg] {tag} INCOMPLETE after {MAX_ATTEMPTS} attempts")
    return False


def main() -> None:
    data_dir = mne.get_config("MNE_DATA") or "~/mne_data"
    print(f"[eeg] MNE data dir: {data_dir}")
    for cls, tag in ((BNCI2014_001, "2a"), (BNCI2014_004, "2b")):
        dataset = cls()
        n_subjects = len(dataset.subject_list)
        print(f"[eeg] downloading BCI IV {tag}: {n_subjects} subjects ...")
        download_with_retries(dataset, tag)


if __name__ == "__main__":
    main()
