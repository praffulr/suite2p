"""
Suite2p Output Loader & Valid Cell Counter for Mouse Dataset Folders (Read-Only)

Reads ONLY iscell.npy (and optional redcell.npy) across mouse folders structured
under a root path (e.g., <root_path>/<mouse_id>/LT/suite2p/plane0/iscell.npy) and
prints exact file paths and cell counts per mouse.

Filtering Criteria:
  • Matches session paths containing 'LT' (e.g. *LT* subfolders)
  • Explicitly ignores any 'ST' subfolders
  • Reads ONLY files directly inside 'plane0' (reporting exactly 1 plane per mouse)

Supported Modes:
  • custom: Process only specified mouse IDs (default: ['Ent3Dq_5952'])
  • all   : Process all discovered mouse folders under root_path

NOTE: This script performs STRICTLY READ-ONLY operations. No files are created,
written, modified, or deleted at any point.
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Union, Dict, List, Optional
import numpy as np

# Default root directory for dataset processing
DEFAULT_ROOT_PATH = "/run/user/1000/gvfs/afp-volume:host=invivo.local,user=Pravuri,volume=shared2/2-PhotonData/JDP/2pData_processed/"
# Default mouse ID list
DEFAULT_MICE = ["Ent3Dq_5952"]


def find_mouse_folders(root_path: Union[str, Path]) -> Dict[str, List[Path]]:
    """
    Find mouse folders under root_path matching *LT* session paths and plane0 directly.
    Explicitly ignores ST subfolders, ensuring exactly 1 plane (plane0) per mouse.

    Strictly READ-ONLY operation.

    Parameters
    ----------
    root_path : str or Path
        Root directory containing mouse folders (e.g. containing Ent3Dq_5952/LT/suite2p/plane0).

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping mouse_id -> list with single plane0 Path directory.
    """
    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    # Check if root itself is a plane0 directory containing iscell.npy directly
    if root.name == "plane0" and (root / "iscell.npy").exists():
        mouse_id = root.parent.parent.name if root.parent.name == "suite2p" else root.parent.name
        return {mouse_id: [root]}

    mouse_map: Dict[str, List[Path]] = {}

    # Iterate over subdirectories in root_path (read-only directory traversal)
    try:
        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    except Exception as e:
        print(f"Warning: Could not list directory {root}: {e}")
        subdirs = []

    for subdir in subdirs:
        # Search specifically for LT session folders containing plane0/iscell.npy inside each mouse folder
        try:
            candidate_files = sorted(list(subdir.glob("*LT*/**/plane0/iscell.npy")))
            if not candidate_files:
                candidate_files = sorted(list(subdir.glob("**/plane0/iscell.npy")))

            valid_plane_dirs = []
            for f in candidate_files:
                plane_dir = f.parent
                path_parts = plane_dir.parts

                # Must be 'plane0', contain 'LT' in path, and NOT contain 'ST' in any path part
                has_lt = any("LT" in part for part in path_parts)
                has_st = any("ST" in part for part in path_parts)

                if plane_dir.name == "plane0" and has_lt and not has_st:
                    valid_plane_dirs.append(plane_dir)

            # Pick exactly 1 plane (plane0 under LT) per mouse
            if valid_plane_dirs:
                mouse_id = subdir.name
                mouse_map[mouse_id] = [valid_plane_dirs[0]]
        except Exception as e:
            print(f"Warning scanning subfolder {subdir}: {e}")

    # Fallback scan if root structure was directly passed
    if not mouse_map:
        try:
            all_iscell = sorted(list(root.glob("**/plane0/iscell.npy")))
            for f in all_iscell:
                plane_dir = f.parent
                path_parts = plane_dir.parts
                has_lt = any("LT" in part for part in path_parts)
                has_st = any("ST" in part for part in path_parts)

                if plane_dir.name == "plane0" and has_lt and not has_st:
                    try:
                        rel_parts = plane_dir.relative_to(root).parts
                        mouse_id = rel_parts[0] if len(rel_parts) > 0 else root.name
                    except ValueError:
                        mouse_id = root.name
                    if mouse_id not in mouse_map:
                        mouse_map[mouse_id] = [plane_dir]
        except Exception as e:
            print(f"Warning in fallback scan of root {root}: {e}")

    return mouse_map


def load_suite2p_data(plane_dir: Union[str, Path]) -> dict:
    """
    Load ONLY iscell.npy (and redcell.npy if present) from a single plane directory.
    Checks presence of other files on disk without loading large arrays for speed.

    Strictly READ-ONLY operation.

    Parameters
    ----------
    plane_dir : str or Path
        Path to plane folder (e.g., suite2p/plane0).

    Returns
    -------
    dict
        Dictionary containing loaded iscell/redcell arrays, existence flags, and exact file paths.
    """
    plane_path = Path(plane_dir).expanduser().resolve()
    if not plane_path.exists():
        raise FileNotFoundError(f"Plane directory not found: {plane_path}")

    data = {
        "plane_path": plane_path,
        "file_paths": {},
        "exists": {}
    }

    npy_files = {
        "iscell": "iscell.npy",
        "redcell": "redcell.npy",
        "stat": "stat.npy",
        "F": "F.npy",
        "Fneu": "Fneu.npy",
        "spks": "spks.npy",
        "ops": "ops.npy",
        "zcorr": "zcorr.npy"
    }

    # Only load small classification files into memory (iscell, redcell)
    load_files = {"iscell", "redcell"}

    for key, filename in npy_files.items():
        file_path = plane_path / filename
        data["file_paths"][key] = file_path
        file_exists = file_path.exists()
        data["exists"][key] = file_exists

        if file_exists and key in load_files:
            try:
                data[key] = np.load(file_path, allow_pickle=True)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                data[key] = None
        else:
            data[key] = None

    return data


def get_cell_counts(iscell: np.ndarray, redcell: Optional[np.ndarray] = None, prob_threshold: float = 0.5) -> dict:
    """
    Compute valid cell counts, non-cell counts, and probabilities from iscell array.

    Parameters
    ----------
    iscell : numpy.ndarray
        Array of shape (N_rois, 2) where column 0 is binary classification and column 1 is probability.
    redcell : numpy.ndarray or None, optional
        Array of shape (N_rois, 2) or (N_rois,) for channel 2 (red cell) classifications.
    prob_threshold : float, optional
        Probability threshold for classifying valid cells. Default is 0.5.

    Returns
    -------
    dict
        Dictionary containing counts and statistics.
    """
    if iscell is None or len(iscell) == 0:
        return {
            "total_rois": 0,
            "valid_cells": 0,
            "non_cells": 0,
            "valid_cell_pct": 0.0,
            "mean_prob_valid": 0.0,
            "mean_prob_all": 0.0,
            "red_cells": None,
        }

    total_rois = iscell.shape[0]
    is_cell_binary = iscell[:, 0].astype(bool)
    probabilities = iscell[:, 1]

    # Valid cells by binary classification flag
    valid_cells = int(np.sum(is_cell_binary))
    non_cells = total_rois - valid_cells
    valid_cell_pct = (valid_cells / total_rois * 100.0) if total_rois > 0 else 0.0

    # Probability stats
    mean_prob_valid = float(np.mean(probabilities[is_cell_binary])) if valid_cells > 0 else 0.0
    mean_prob_all = float(np.mean(probabilities)) if total_rois > 0 else 0.0

    # Count by custom probability threshold
    above_thresh = int(np.sum(probabilities >= prob_threshold))

    counts = {
        "total_rois": total_rois,
        "valid_cells": valid_cells,
        "non_cells": non_cells,
        "valid_cell_pct": valid_cell_pct,
        "above_thresh_count": above_thresh,
        "mean_prob_valid": mean_prob_valid,
        "mean_prob_all": mean_prob_all,
        "red_cells": None,
    }

    # Red cell analysis if present
    if redcell is not None and len(redcell) == total_rois:
        if redcell.ndim == 2:
            red_binary = redcell[:, 0].astype(bool)
        else:
            red_binary = redcell.astype(bool)
        
        red_cells_count = int(np.sum(red_binary))
        # Valid cells that are also red positive
        valid_red_cells = int(np.sum(is_cell_binary & red_binary))
        
        counts["red_cells"] = {
            "total_red_rois": red_cells_count,
            "valid_red_cells": valid_red_cells,
        }

    return counts


def print_mouse_summary(mouse_id: str, plane_data_list: List[dict], prob_threshold: float = 0.5) -> dict:
    """
    Print statistics for a single mouse folder, listing the exact file paths data was read from.

    Parameters
    ----------
    mouse_id : str
        ID or folder name of the mouse.
    plane_data_list : list of dict
        List of dictionaries loaded from load_suite2p_data for each plane in this mouse folder.
    prob_threshold : float, optional
        Threshold used for probability counts.

    Returns
    -------
    dict
        Summary counts for this mouse.
    """
    print("=" * 80)
    print(f" MOUSE ID: {mouse_id} (1 plane directory found: plane0 under LT)")
    print("=" * 80)

    mouse_total_rois = 0
    mouse_valid_cells = 0

    for idx, data in enumerate(plane_data_list, 1):
        plane_path = data["plane_path"]
        iscell = data["iscell"]
        file_paths = data.get("file_paths", {})
        exists_dict = data.get("exists", {})

        print(f"\n --- Target Plane ({plane_path.name}): {plane_path} ---")
        print(" Exact File Paths Read From:")
        for key in ["iscell", "redcell", "stat", "F", "Fneu", "spks", "ops", "zcorr"]:
            fp = file_paths.get(key)
            arr = data.get(key)
            file_exists = exists_dict.get(key, False)

            if arr is not None:
                shape_str = str(arr.shape) if hasattr(arr, "shape") else f"len={len(arr)}"
                print(f"  • {key:<10} -> {fp} [Loaded, Shape: {shape_str}]")
            elif file_exists:
                print(f"  • {key:<10} -> {fp} [Exists on disk (skipped load for speed)]")
            else:
                print(f"  • {key:<10} -> {fp} [NOT FOUND]")

        if iscell is None:
            print("  [ERROR] iscell.npy could not be loaded!")
            continue

        counts = get_cell_counts(iscell, redcell=data.get("redcell"), prob_threshold=prob_threshold)
        mouse_total_rois += counts["total_rois"]
        mouse_valid_cells += counts["valid_cells"]

        print("\n ROI & Cell Counts:")
        print(f"  • Total Detected ROIs     : {counts['total_rois']}")
        print(f"  • Valid Cells (iscell=1)  : {counts['valid_cells']} ({counts['valid_cell_pct']:.2f}%)")
        print(f"  • Non-Cells (iscell=0)    : {counts['non_cells']} ({100.0 - counts['valid_cell_pct']:.2f}%)")
        print(f"  • ROIs with Prob >= {prob_threshold:.2f} : {counts['above_thresh_count']}")
        print(f"  • Mean Prob (Valid Cells) : {counts['mean_prob_valid']:.4f}")
        print(f"  • Mean Prob (All ROIs)    : {counts['mean_prob_all']:.4f}")

        if counts["red_cells"] is not None:
            rc = counts["red_cells"]
            print(f"  • Red Channel ROIs        : {rc['total_red_rois']}")
            print(f"  • Valid Red Cells         : {rc['valid_red_cells']}")

    print("\n" + "-" * 80)
    print(f" MOUSE SUMMARY ({mouse_id}):")
    print(f"  • Total ROIs Detected     : {mouse_total_rois}")
    print(f"  • Total Valid Cells       : {mouse_valid_cells}")
    if mouse_total_rois > 0:
        pct = (mouse_valid_cells / mouse_total_rois) * 100.0
        print(f"  • Valid Cell Percentage   : {pct:.2f}%")
    print("=" * 80)

    return {
        "mouse_id": mouse_id,
        "total_rois": mouse_total_rois,
        "valid_cells": mouse_valid_cells
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read ONLY iscell.npy for plane0 under LT session folders per mouse and print exact file paths & cell statistics."
    )
    parser.add_argument(
        "root_path",
        nargs="?",
        default=DEFAULT_ROOT_PATH,
        help=f"Root directory containing mouse folders (default: {DEFAULT_ROOT_PATH})."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "custom"],
        default="custom",
        help="Processing mode: 'all' to process all discovered mouse folders, or 'custom' to filter by specified mouse IDs (default: 'custom')."
    )
    parser.add_argument(
        "--mice",
        "-m",
        nargs="+",
        default=DEFAULT_MICE,
        help="List of Mouse IDs to process when mode is 'custom' (default: ['Ent3Dq_5952']). Examples: --mice Ent3Dq_5952 or --mice Ent3Dq_5952 Ent3Dq_5953"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Probability threshold for valid cell classification (default: 0.5)."
    )

    args = parser.parse_args()

    # Parse requested mouse IDs (handling space or comma separated inputs)
    raw_mice = []
    for item in args.mice:
        raw_mice.extend(item.split(","))
    requested_mice = [m.strip() for m in raw_mice if m.strip()]

    try:
        mouse_map = find_mouse_folders(args.root_path)

        if not mouse_map:
            print(f"No mouse folders containing LT/suite2p/plane0/iscell.npy found under: {args.root_path}")
            sys.exit(1)

        total_discovered = len(mouse_map)
        all_discovered_keys = list(mouse_map.keys())

        # Filter mouse_map based on --mode
        if args.mode == "custom":
            target_mice_lower = set(m.lower() for m in requested_mice)
            filtered_map = {
                k: v for k, v in mouse_map.items() if k.lower() in target_mice_lower
            }

            missing_mice = [m for m in requested_mice if m.lower() not in set(k.lower() for k in mouse_map.keys())]
            if missing_mice:
                print(f"Warning: Requested mouse ID(s) not found under root_path: {missing_mice}")
                print(f"Available mouse IDs under root_path ({total_discovered}): {', '.join(all_discovered_keys)}\n")

            if not filtered_map:
                print(f"Error: None of the requested mouse IDs {requested_mice} were found under {args.root_path}.")
                sys.exit(1)

            mouse_map = filtered_map

        print("\n" + "=" * 80)
        print(f" ROOT PATH ANALYSIS: {Path(args.root_path).expanduser().resolve()}")
        print(f" Mode: {args.mode.upper()}")
        print(f" Filter: Matching *LT* sessions, ignoring ST, reading plane0 exclusively")
        print(f" Discovered Mouse Folders Total : {total_discovered}")
        print(f" Selected Mouse Folders Count   : {len(mouse_map)}")
        print(f" Selected Mouse IDs             : {', '.join(mouse_map.keys())}")
        print("=" * 80 + "\n")

        grand_total_rois = 0
        grand_total_valid_cells = 0

        for mouse_id, plane_dirs in mouse_map.items():
            plane_data_list = [load_suite2p_data(pdir) for pdir in plane_dirs]
            summary = print_mouse_summary(mouse_id, plane_data_list, prob_threshold=args.threshold)
            grand_total_rois += summary["total_rois"]
            grand_total_valid_cells += summary["valid_cells"]
            print("\n")

        print("=" * 80)
        print(f" OVERALL SUMMARY ({args.mode.upper()} MODE - 1 PLANE0 PER MOUSE)")
        print(f"  • Total Mouse Folders Processed  : {len(mouse_map)}")
        print(f"  • Grand Total Detected ROIs      : {grand_total_rois}")
        print(f"  • Grand Total Valid Cells        : {grand_total_valid_cells}")
        if grand_total_rois > 0:
            pct = (grand_total_valid_cells / grand_total_rois) * 100.0
            print(f"  • Grand Total Valid Percentage   : {pct:.2f}%")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"Error processing mouse data under root path: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
