"""Script to generate processed.npy metadata mapping for all 2P sessions in Data_2P_F_data."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np


DEFAULT_BASE_DIR = Path("/home/pravuri/Desktop/Data_2P_F_data")


def create_processed_for_folder(folder_path: Union[str, Path]) -> np.ndarray:
    """Generate processed.npy for a single Suite2p output directory containing ops.npy.

    Parameters
    ----------
    folder_path:
        Path to the Suite2p output folder containing ops.npy.

    Returns
    -------
    np.ndarray
        Structured array containing session metadata, flags, and frame slice mapping.
    """
    folder = Path(folder_path).expanduser().resolve()
    ops_path = folder / "ops.npy"

    if not ops_path.exists():
        raise FileNotFoundError(f"ops.npy not found in folder: {folder}")

    ops = np.load(ops_path, allow_pickle=True).item()

    filelist_raw = ops.get("filelist", [])
    filelist = [str(f).replace("\\", "/") for f in filelist_raw]
    stems = [Path(f).stem for f in filelist]
    nframes_per_folder = ops.get("nframes_per_folder", [])

    if len(stems) != len(nframes_per_folder):
        raise ValueError(
            f"Mismatch in {folder.name}: {len(stems)} files vs {len(nframes_per_folder)} frame counts"
        )

    # Parse dates and run IDs from session filenames
    parsed = []
    for idx, name in enumerate(stems):
        parts = name.split("_")
        date_str = parts[0]
        mouse_id = parts[1] if len(parts) > 1 else ""
        run_id = int(parts[2]) if len(parts) > 2 else idx + 1
        parsed.append({
            "idx": idx,
            "session_name": name,
            "date": date_str,
            "mouse": mouse_id,
            "run_id": run_id,
            "n_frames": int(nframes_per_folder[idx]),
        })

    # Sort dates chronologically to assign Day 1, Day 2, Day 3
    unique_dates = sorted(list(set(p["date"] for p in parsed)))
    date_to_day = {d: day_idx for day_idx, d in enumerate(unique_dates, start=1)}

    sessions: List[dict] = []
    for p in parsed:
        day_idx = date_to_day[p["date"]]
        r_id = p["run_id"]
        is_baseline = r_id <= 40

        if is_baseline:
            treatment = "baseline"
            treatment_code = 0
        else:
            if day_idx % 2 == 1:  # Day 1, Day 3, etc.
                treatment = "PBS"
                treatment_code = 1
            else:  # Day 2, etc.
                treatment = "clozapine"
                treatment_code = 2

        sessions.append({
            "session_name": p["session_name"],
            "date": p["date"],
            "day": day_idx,
            "run_id": r_id,
            "is_baseline": is_baseline,
            "treatment": treatment,
            "treatment_code": treatment_code,
            "n_frames": p["n_frames"],
        })

    running_sum = 0
    dtype = [
        ("session_name", "U40"),
        ("date", "U10"),
        ("day", "i4"),
        ("run_id", "i4"),
        ("is_baseline", "bool"),
        ("treatment", "U20"),
        ("treatment_code", "i4"),
        ("n_frames", "i8"),
        ("frame_start", "i8"),
        ("frame_end", "i8"),
        ("running_sum_frames", "i8"),
    ]

    rec_array = np.zeros(len(sessions), dtype=dtype)

    for i, s in enumerate(sessions):
        nf = s["n_frames"]
        f_start = running_sum
        f_end = running_sum + nf
        running_sum = f_end

        rec_array[i] = (
            s["session_name"],
            s["date"],
            s["day"],
            s["run_id"],
            s["is_baseline"],
            s["treatment"],
            s["treatment_code"],
            nf,
            f_start,
            f_end,
            running_sum,
        )

    out_file = folder / "processed.npy"
    np.save(out_file, rec_array)
    print(f"[{folder.name}] Saved processed.npy: {len(sessions)} sessions, {running_sum} total frames -> {out_file}")

    return rec_array


def process_all_folders(base_dir: Union[str, Path] = DEFAULT_BASE_DIR) -> Dict[str, np.ndarray]:
    """Process all subdirectories in base_dir containing ops.npy and create processed.npy.

    Parameters
    ----------
    base_dir:
        Path to the root folder containing mouse session directories (default: Data_2P_F_data).

    Returns
    -------
    dict
        Dictionary mapping folder names to generated structured arrays.
    """
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    results = {}
    print(f"Scanning for Suite2p folders in: {base_path}\n")

    for folder in sorted(base_path.iterdir()):
        if folder.is_dir() and (folder / "ops.npy").exists():
            try:
                arr = create_processed_for_folder(folder)
                results[folder.name] = arr
            except Exception as err:
                print(f"[{folder.name}] ERROR: {err}")

    print(f"\nSuccessfully generated processed.npy for {len(results)} folders.")
    return results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_DIR
    target_path = Path(target).expanduser().resolve()

    if target_path.is_dir() and (target_path / "ops.npy").exists():
        create_processed_for_folder(target_path)
    else:
        process_all_folders(target_path)
