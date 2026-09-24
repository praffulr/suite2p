import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def print_stat_summary(stat):
    """
    Print header-level summary details of what stat.npy captures.

    Parameters
    ----------
    stat : numpy.ndarray or list of dict
        The loaded stat.npy array containing ROI metadata dictionaries.
    """
    print("\n" + "=" * 80)
    print(f" STAT.NPY HEADER-LEVEL SUMMARY (Total ROIs: {len(stat)})")
    print("=" * 80)

    if len(stat) == 0:
        print(" [WARN] stat file contains 0 ROIs.")
        print("=" * 80 + "\n")
        return

    first_roi = stat[0]
    keys = list(first_roi.keys())

    print(f" • Total ROI Entries : {len(stat)}")
    print(f" • Number of Keys    : {len(keys)}")
    print(f" • Captured Keys     : {', '.join(keys)}\n")

    header = f" {'Key Name':<18} {'Data Type':<16} {'Structure / Shape':<20} {'Summary / Sample Value':<24}"
    print(header)
    print("-" * 80)

    for k in keys:
        val = first_roi[k]
        val_type = type(val).__name__

        if isinstance(val, np.ndarray):
            dtype_str = f"ndarray ({val.dtype})"
            shape_str = str(val.shape)
            if np.issubdtype(val.dtype, np.number) and val.size > 0:
                sample_str = f"min={val.min():.2f}, max={val.max():.2f}, mean={val.mean():.2f}"
            else:
                sample_str = f"len={len(val)}"
        elif isinstance(val, (list, tuple)):
            dtype_str = val_type
            shape_str = f"len={len(val)}"
            sample_str = str(val[:3]) if len(val) > 3 else str(val)
        elif isinstance(val, (int, float, np.integer, np.floating)):
            dtype_str = val_type
            shape_str = "scalar"
            # Aggregate statistics across all ROIs for numeric scalar attributes
            all_vals = [s[k] for s in stat if k in s and isinstance(s[k], (int, float, np.integer, np.floating))]
            if len(all_vals) > 0:
                sample_str = f"mean={np.mean(all_vals):.2f} [{np.min(all_vals):.2f} - {np.max(all_vals):.2f}]"
            else:
                sample_str = str(val)
        else:
            dtype_str = val_type
            shape_str = "scalar/object"
            sample_str = str(val)

        print(f" {k:<18} {dtype_str:<16} {shape_str:<20} {sample_str:<24}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Suite2p ROIs from stat.npy and iscell.npy in a folder."
    )
    parser.add_argument(
        "folder_path",
        nargs="?",
        default=".",
        help="Path to folder containing stat.npy and iscell.npy (default: current directory)."
    )
    args = parser.parse_args()

    folder = Path(args.folder_path).expanduser().resolve()
    stat_path = folder / "stat.npy"
    iscell_path = folder / "iscell.npy"

    if not stat_path.exists():
        raise FileNotFoundError(f"stat.npy not found in: {folder}")
    if not iscell_path.exists():
        raise FileNotFoundError(f"iscell.npy not found in: {folder}")

    print(f"Loading data from: {folder}")

    # 1. Load data
    stat = np.load(stat_path, allow_pickle=True)
    iscell = np.load(iscell_path, allow_pickle=True)

    # Print header-level details captured in stat.npy
    print_stat_summary(stat)

    # 2. Determine image dimensions from pixel coordinates
    Ly = max(s['ypix'].max() for s in stat) + 1
    Lx = max(s['xpix'].max() for s in stat) + 1

    # 3. Build 2D ROI spatial maps for instance segmentation & footprints
    cell_mask = np.zeros((Ly, Lx), dtype=np.int32)
    non_cell_mask = np.zeros((Ly, Lx), dtype=np.int32)
    footprint_mask = np.full((Ly, Lx), np.nan, dtype=np.float32)

    for i, s in enumerate(stat):
        ypix, xpix = s['ypix'], s['xpix']
        fp = s.get('footprint', 0)
        if iscell[i, 0] == 1:
            cell_mask[ypix, xpix] = i + 1  # Unique ROI instance ID
        else:
            non_cell_mask[ypix, xpix] = i + 1
        footprint_mask[ypix, xpix] = fp

    # Shuffle non-zero labels so neighboring ROIs have distinct contrasting colors
    np.random.seed(42)
    rand_lut = np.random.permutation(len(stat) + 1)
    rand_lut[0] = 0  # Background stays 0

    cell_instances = rand_lut[cell_mask]
    non_cell_instances = rand_lut[non_cell_mask]

    # Mask background (0) to display as solid black
    cell_instances_masked = np.ma.masked_where(cell_instances == 0, cell_instances)
    non_cell_instances_masked = np.ma.masked_where(non_cell_instances == 0, non_cell_instances)

    cmap_instance = plt.cm.nipy_spectral.copy()
    cmap_instance.set_bad(color='black')

    cmap_footprint = plt.cm.viridis.copy()
    cmap_footprint.set_bad(color='black')

    # 4. Plot ROIs (Instance Segmentation & Footprint Scale Map)
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    axes[0].imshow(cell_instances_masked, cmap=cmap_instance, interpolation='nearest')
    axes[0].set_title(f"Valid Cells Instance Map (iscell=1 | N={int(iscell[:, 0].sum())})")
    axes[0].axis('off')

    axes[1].imshow(non_cell_instances_masked, cmap=cmap_instance, interpolation='nearest')
    axes[1].set_title(f"Non-Cells Instance Map (iscell=0 | N={int((iscell[:, 0] == 0).sum())})")
    axes[1].axis('off')

    # im2 = axes[2].imshow(footprint_mask, cmap=cmap_footprint, vmin=0, vmax=4, interpolation='nearest')
    # axes[2].set_title("ROI Spatial Footprint Map (Scale Index: 0 to 4)")
    # axes[2].axis('off')

    # cbar = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3, 4])
    # cbar.ax.set_yticklabels(['0 (3px)', '1 (6px)', '2 (12px)', '3 (24px)', '4 (48px)'])
    # cbar.set_label('Detection Scale Level', rotation=270, labelpad=15)

    plt.suptitle(f"Folder: {folder}", fontsize=11)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
