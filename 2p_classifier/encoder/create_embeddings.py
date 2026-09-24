import sys
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Add preproc directory to sys.path to import BaselineSpikeDataset
sys.path.append(str(Path(__file__).resolve().parent.parent / "preproc"))
try:
    from create_dataset import BaselineSpikeDataset
except ImportError as e:
    print(f"Warning: Could not import BaselineSpikeDataset ({e}).")

try:
    from momentfm import MOMENTPipeline
except ImportError:
    print("Error: momentfm not installed.")
    print("Please run this script in a Python 3.10+ environment where momentfm is installed.")
    print("Example: conda activate moment_env")
    sys.exit(1)


def get_embeddings_for_dataset(dataset_path: str, batch_size: int = 64):
    """
    Loads a BaselineSpikeDataset, formats the time series data for MOMENT,
    extracts embeddings for each unit, and saves the results.
    """
    dataset_path = Path(dataset_path).expanduser().resolve()
    print(f"Loading dataset from {dataset_path}")
    ds = BaselineSpikeDataset.load(dataset_path)
    
    print("Initializing MOMENT model...")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"}
    )
    model.init()
    model.eval()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    all_embeddings = []
    total_records = len(ds)
    
    from tqdm import tqdm
    print(f"Extracting embeddings for {total_records} records in batches of {batch_size}...")
    
    for i in tqdm(range(0, total_records, batch_size), desc="Extracting Embeddings"):
        batch_records = [ds[j] for j in range(i, min(i + batch_size, total_records))]
        
        batch_tensors = []
        for rec in batch_records:
            # rec is (mouse_id, unit_id, day_id, session_id, spks_2d)
            spks_2d = rec[4]
            spk_vals = spks_2d[:, 1] # 1D array of spike values
            
            # Convert to tensor: shape [1, seq_len]
            t = torch.tensor(spk_vals, dtype=torch.float32).unsqueeze(0)
            
            # MOMENT expects exactly 512 length sequences.
            # We use linear interpolation to resize the sequence to 512.
            # F.interpolate expects [batch, channels, length] -> [1, 1, seq_len]
            t = t.unsqueeze(0)
            t_resized = F.interpolate(t, size=512, mode='linear', align_corners=False)
            batch_tensors.append(t_resized.squeeze(0)) # Back to [1, 512]
            
        # batch_x shape: [batch_size, 1, 512]
        batch_x = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            outputs = model(x_enc=batch_x)
            # embeddings shape: [batch_size, 1024] (for MOMENT-1-large)
            emb = outputs.embeddings.detach().cpu().numpy()
            
        for k, rec in enumerate(batch_records):
            all_embeddings.append({
                "mouse_id": rec[0],
                "unit_id": rec[1],
                "day_id": rec[2],
                "session_id": rec[3],
                "embedding": emb[k]
            })
            
    out_path = dataset_path.parent / (dataset_path.stem + "_embeddings.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(all_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Successfully saved embeddings to {out_path}")


if __name__ == "__main__":
    # Default to the combined dataset if no path is provided
    default_ds = "/home/pravuri/Desktop/Data_2P_F_data/combined_baseline_spike_dataset.pkl"
    target = sys.argv[1] if len(sys.argv) > 1 else default_ds
    
    if not Path(target).exists():
        print(f"Error: Dataset not found at {target}")
        sys.exit(1)
        
    get_embeddings_for_dataset(target)