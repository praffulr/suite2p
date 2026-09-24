"""Utilities for inspecting Suite2p F.npy fluorescence arrays and MATLAB .mat metadata files."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import scipy.io as sio


DEFAULT_F_FOLDER = Path("/home/pravuri/Desktop/Data_2P_F_data/5882_ST_40/")
DEFAULT_MAT_PATH = (
    "smb://invivo.local/shared2/2-PhotonData/JDP/2pData_TSM/Ent3Dq_5882/ST/"
    "ShortTerm_30-runs-per-session/25_01_29-30-31/250129_5882_001.mat"
)


def resolve_path(path_input: Union[str, Path]) -> Path:
    """Resolve local paths or smb:// network share URIs to accessible filesystem paths."""
    path_str = str(path_input).strip()
    if path_str.startswith("smb://"):
        no_smb = path_str[len("smb://") :]
        parts = no_smb.split("/", 2)
        server = parts[0]
        share = parts[1] if len(parts) > 1 else ""
        rest = parts[2] if len(parts) > 2 else ""
        uid = os.getuid()
        gvfs_path = Path(f"/run/user/{uid}/gvfs/smb-share:server={server},share={share}") / rest
        return gvfs_path
    return Path(path_str).expanduser().resolve()


def _mat_to_dict(obj: Any) -> Any:
    """Recursively convert MATLAB structs and ndarrays to Python dicts / lists / scalars."""
    if hasattr(obj, "_fieldnames"):
        return {field: _mat_to_dict(getattr(obj, field)) for field in obj._fieldnames}
    elif isinstance(obj, np.ndarray):
        if obj.dtype.names is not None:
            return {name: _mat_to_dict(obj[name]) for name in obj.dtype.names}
        elif obj.size == 0:
            return []
        elif obj.dtype.kind in ["i", "u", "f", "b", "U", "S"]:
            if obj.size == 1:
                val = obj.item()
                if isinstance(val, (np.integer, np.floating, np.bool_)):
                    return val.item()
                return val
            elif obj.ndim == 1 and obj.size < 20:
                return obj.tolist()
            else:
                return {
                    "type": "ndarray",
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                    "min": float(np.min(obj)) if obj.size > 0 else None,
                    "max": float(np.max(obj)) if obj.size > 0 else None,
                    "mean": float(np.mean(obj)) if obj.size > 0 else None,
                    "std": float(np.std(obj)) if obj.size > 0 else None,
                }
        elif obj.dtype == object:
            return [_mat_to_dict(x) for x in obj.flat]
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def report_mat_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a MATLAB .mat file and report its structure and statistics.

    Parameters
    ----------
    file_path:
        Path or smb:// URI pointing to the .mat file.

    Returns
    -------
    dict
        Dictionary containing extracted metadata and array summary statistics.
    """
    resolved_path = resolve_path(file_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"MAT file not found at {resolved_path} (original: {file_path})")

    mat_raw = sio.loadmat(resolved_path, squeeze_me=True, struct_as_record=False)
    parsed_contents: Dict[str, Any] = {}
    for key, value in mat_raw.items():
        if not key.startswith("__"):
            parsed_contents[key] = _mat_to_dict(value)

    stats = {
        "file_path": str(file_path),
        "resolved_path": str(resolved_path),
        "contents": parsed_contents,
    }

    print(f"Loaded MAT File: {resolved_path}")
    print(f"Top-level keys: {list(parsed_contents.keys())}")
    print("\n--- Parsed Contents ---")
    print(json.dumps(parsed_contents, indent=2))

    return stats


def report_ops_stats(path_input: Union[str, Path]) -> Dict[str, Any]:
    """Load Suite2p ops.npy and report pipeline configuration, session files, and frame distributions.

    Parameters
    ----------
    path_input:
        Path to ops.npy or directory containing ops.npy.

    Returns
    -------
    dict
        Dictionary of parsed ops.npy configuration and session metadata.
    """
    path = resolve_path(path_input)
    ops_path = path if (path.is_file() and path.name == "ops.npy") else path / "ops.npy"

    if not ops_path.exists():
        raise FileNotFoundError(f"ops.npy not found at {ops_path}")

    ops = np.load(ops_path, allow_pickle=True).item()

    filelist_raw = ops.get("filelist", [])
    filelist = [str(f).replace("\\", "/") for f in filelist_raw]
    stems = [Path(f).stem for f in filelist]

    nframes_per_folder = ops.get("nframes_per_folder", np.array([]))

    # Parse dates and run IDs from session filenames
    parsed_sessions = []
    for s in stems:
        parts = s.split("_")
        if len(parts) >= 3:
            parsed_sessions.append({"date": parts[0], "mouse": parts[1], "run_id": int(parts[2])})

    dates = sorted(list(set(p["date"] for p in parsed_sessions))) if parsed_sessions else []

    pipeline_cfg = {
        "Lx": ops.get("Lx"),
        "Ly": ops.get("Ly"),
        "nplanes": ops.get("nplanes"),
        "nchannels": ops.get("nchannels"),
        "tau": ops.get("tau"),
        "fs": ops.get("fs"),
        "do_registration": ops.get("do_registration"),
        "nonrigid": ops.get("nonrigid"),
        "functional_chan": ops.get("functional_chan"),
    }

    stats = {
        "ops_path": str(ops_path),
        "data_path": ops.get("data_path"),
        "total_nframes": int(ops.get("nframes", 0)),
        "num_files": len(filelist),
        "dates": dates,
        "nframes_per_folder": {
            "data": nframes_per_folder,
            "count": len(nframes_per_folder),
            "min": int(np.min(nframes_per_folder)) if len(nframes_per_folder) > 0 else 0,
            "max": int(np.max(nframes_per_folder)) if len(nframes_per_folder) > 0 else 0,
            "mean": float(np.mean(nframes_per_folder)) if len(nframes_per_folder) > 0 else 0.0,
            "sum": int(np.sum(nframes_per_folder)) if len(nframes_per_folder) > 0 else 0,
        },
        "pipeline_config": pipeline_cfg,
    }

    print(f"Loaded ops.npy from: {ops_path}")
    print(f"  Data Path: {stats['data_path']}")
    print(f"  Total Frames: {stats['total_nframes']}")
    print(f"  Total Sessions/Files: {stats['num_files']}")
    print(f"  Dates ({len(dates)}): {dates}")
    print(
        "  Frames per session: "
        f"count={stats['nframes_per_folder']['count']}, "
        f"min={stats['nframes_per_folder']['min']}, "
        f"max={stats['nframes_per_folder']['max']}, "
        f"mean={stats['nframes_per_folder']['mean']:.1f}, "
        f"sum={stats['nframes_per_folder']['sum']}"
    )
    print("  Pipeline Config:", json.dumps(pipeline_cfg, indent=4))
    print ("nframes_per_folder:", stats['nframes_per_folder'])

    return stats


def report_f_stats(path_input: Union[str, Path] = DEFAULT_F_FOLDER) -> Dict[str, Any]:
    """Load F.npy, ops.npy, or a .mat file and report basic statistics.

    Parameters
    ----------
    path_input:
        Path or smb:// URI to a folder containing F.npy/ops.npy, direct file, or a .mat file.

    Returns
    -------
    dict
        Dictionary with shape and summary statistics or parsed MAT/ops contents.
    """
    path_str = str(path_input)
    if path_str.endswith(".mat") or (isinstance(path_input, Path) and path_input.suffix == ".mat"):
        return report_mat_stats(path_input)

    path = resolve_path(path_input)
    if path.is_file():
        if path.name == "ops.npy":
            return report_ops_stats(path)
        elif path.suffix == ".mat":
            return report_mat_stats(path)

    # Check if ops.npy exists in directory
    ops_file = path / "ops.npy" if path.is_dir() else None
    if ops_file and ops_file.exists():
        print("--- Suite2p ops.npy Details ---")
        report_ops_stats(ops_file)
        print("\n--- Suite2p F.npy Details ---")

    f_path = path if path.is_file() else path / "F.npy"

    if not f_path.exists():
        raise FileNotFoundError(f"F.npy not found at {f_path}")

    f = np.load(f_path, allow_pickle=False)

    stats = {
        "path": str(f_path),
        "shape": tuple(f.shape),
        "dtype": str(f.dtype),
        "size": int(f.size),
        "min": float(np.min(f)),
        "max": float(np.max(f)),
        "mean": float(np.mean(f)),
        "std": float(np.std(f)),
        "median": float(np.median(f)),
    }

    print(f"Loaded: {f_path}")
    print(f"Shape: {stats['shape']}")
    print(f"Dtype: {stats['dtype']}")
    print(f"Size: {stats['size']}")
    print(
        "Stats: "
        f"min={stats['min']:.6g}, max={stats['max']:.6g}, "
        f"mean={stats['mean']:.6g}, std={stats['std']:.6g}, median={stats['median']:.6g}"
    )

    return stats


report_stats = report_f_stats


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_F_FOLDER
    report_ops_stats(target)