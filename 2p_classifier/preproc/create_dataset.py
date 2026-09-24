"""Utility for building baseline spike datasets across 2P mouse sessions."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np


DEFAULT_BASE_DIR = Path("/home/pravuri/Desktop/Data_2P_F_data")


@dataclass
class BaselineSpikeRecord:
    """Dataclass holding baseline session data for a single unit.

    Attributes
    ----------
    mouse_id : str
        Identifier of the mouse (e.g., '5882', '5950').
    unit_id : int
        Unit/ROI index matching the row index in iscell.npy and spks.npy.
    day_id : int
        Day index (e.g., 1, 2, 3) read from processed.npy.
    session_id : int
        Baseline session index (1 to 30) read from processed.npy.
    spks_data : np.ndarray
        2D numpy array of shape (N_frames, 2) where:
        - Column 0: timeframe_id (0, 1, ..., N_frames - 1)
        - Column 1: corresponding spike value from spks.npy for this session
    """

    mouse_id: str
    unit_id: int
    day_id: int
    session_id: int
    spks_data: np.ndarray

    def as_tuple(self) -> Tuple[str, int, int, int, np.ndarray]:
        """Return record as tuple (mouse_id, unit_id, day_id, session_id, spks_data)."""
        return (self.mouse_id, self.unit_id, self.day_id, self.session_id, self.spks_data)


class BaselineSpikeDataset:
    """Dataset container holding baseline session records across mice."""

    def __init__(self, records: List[Tuple[str, int, int, int, np.ndarray]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[str, int, int, int, np.ndarray]:
        return self.records[idx]

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved dataset ({len(self.records)} records) to {path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> BaselineSpikeDataset:
        path = Path(file_path).expanduser().resolve()
        with open(path, "rb") as f:
            records = pickle.load(f)
        print(f"Loaded dataset ({len(records)} records) from {path}")
        return cls(records)


def create_baseline_dataset(
    mouse_folders: Union[str, Path, Sequence[Union[str, Path]]],
    max_baseline_session_id: int = 30,
) -> BaselineSpikeDataset:
    """Build a dataset of baseline session records for given mouse folder(s).

    Parameters
    ----------
    mouse_folders:
        Path or sequence of paths to mouse folders in Data_2P_F_data.
    max_baseline_session_id:
        Maximum session_id / run_id to include as baseline (default: 30).

    Returns
    -------
    BaselineSpikeDataset
        Dataset containing baseline spike records.
    """
    if isinstance(mouse_folders, (str, Path)):
        folders = [Path(mouse_folders)]
    else:
        folders = [Path(p) for p in mouse_folders]

    records: List[Tuple[str, int, int, int, np.ndarray]] = []

    for folder in folders:
        folder = folder.expanduser().resolve()
        proc_path = folder / "processed.npy"
        spks_path = folder / "spks.npy"
        iscell_path = folder / "iscell.npy"

        if not proc_path.exists():
            raise FileNotFoundError(f"Missing processed.npy in {folder}")
        if not spks_path.exists():
            raise FileNotFoundError(f"Missing spks.npy in {folder}")

        proc = np.load(proc_path)
        spks = np.load(spks_path, mmap_mode="r")
        num_units = spks.shape[0]

        if iscell_path.exists():
            iscell = np.load(iscell_path)
            if iscell.shape[0] != num_units:
                print(
                    f"Warning in {folder.name}: iscell shape {iscell.shape} vs spks shape {spks.shape}"
                )

        # Filter baseline sessions (run_id 1 to max_baseline_session_id)
        baseline_mask = (
            (proc["is_baseline"] == True)
            & (proc["run_id"] >= 1)
            & (proc["run_id"] <= max_baseline_session_id)
        )
        baseline_sessions = proc[baseline_mask]

        print(
            f"[{folder.name}] Units: {num_units} | Baseline sessions: {len(baseline_sessions)}"
        )

        for sess in baseline_sessions:
            sess_name = str(sess["session_name"])
            parts = sess_name.split("_")
            mouse_id = parts[1] if len(parts) > 1 else folder.name
            day_id = int(sess["day"])
            session_id = int(sess["run_id"])
            f_start = int(sess["frame_start"])
            f_end = int(sess["frame_end"])
            n_frames = int(sess["n_frames"])

            timeframe_ids = np.arange(n_frames, dtype=np.int64)

            for unit_id in range(num_units):
                spk_vals = np.array(spks[unit_id, f_start:f_end], dtype=np.float32)
                spks_2d = np.column_stack((timeframe_ids, spk_vals))

                records.append((mouse_id, unit_id, day_id, session_id, spks_2d))

    print(f"\nTotal dataset records created: {len(records)}")
    return BaselineSpikeDataset(records)


    print(f"\nTotal dataset records created: {len(records)}")
    return BaselineSpikeDataset(records)


def process_all_baseline_datasets(
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
) -> BaselineSpikeDataset:
    """Process all subfolders in base_dir containing processed.npy and spks.npy."""
    base_path = Path(base_dir).expanduser().resolve()
    folders = [
        f for f in sorted(base_path.iterdir())
        if f.is_dir() and (f / "processed.npy").exists() and (f / "spks.npy").exists()
    ]
    dataset = create_baseline_dataset(folders)
    out_file = base_path / "baseline_spike_dataset.pkl"
    dataset.save(out_file)
    return dataset


if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = [Path(t).expanduser().resolve() for t in sys.argv[1:]]
        # If a single directory is provided and it doesn't have processed.npy, assume it's a base dir
        if len(targets) == 1 and targets[0].is_dir() and not (targets[0] / "processed.npy").exists():
            process_all_baseline_datasets(targets[0])
        else:
            ds = create_baseline_dataset(targets)
            out_file = targets[0].parent / "combined_baseline_spike_dataset.pkl"
            ds.save(out_file)
    else:
        process_all_baseline_datasets(DEFAULT_BASE_DIR)
