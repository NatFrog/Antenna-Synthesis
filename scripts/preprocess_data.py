"""
One-time preprocessing: convert 200 CSV files (MATLAB + HFSS) into a single HDF5 file.

Usage:
    python -m scripts.preprocess_data

This reads all 100 CSV files from each dataset, extracts patterns and metadata,
and writes them to processed/antenna_data.h5.

Expected runtime: 20-40 minutes for ~7 GB of CSV data.
"""

import sys
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    HFSS_DIR, MATLAB_DIR, PROCESSED_DIR, HDF5_PATH,
    N_FILES, N_CONFIGS_PER_FILE, N_TOTAL_CONFIGS,
    N_THETA, N_PHI, N_SPATIAL_POINTS,
)
from src.data.loader import load_single_csv, get_file_path, get_config_columns


def preprocess_all():
    """Convert all CSV files to HDF5."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Create HDF5 file with pre-allocated arrays
    print(f"Creating HDF5 file at: {HDF5_PATH}")
    print(f"Total configurations: {N_TOTAL_CONFIGS}")
    print(f"Pattern shape per config: ({N_THETA}, {N_PHI})")

    with h5py.File(str(HDF5_PATH), 'w') as f:
        # Pre-allocate datasets
        matlab_ds = f.create_dataset(
            'matlab_patterns',
            shape=(N_TOTAL_CONFIGS, N_THETA, N_PHI),
            dtype=np.float16,
            chunks=(1, N_THETA, N_PHI),
            compression='gzip',
            compression_opts=4,
        )
        hfss_ds = f.create_dataset(
            'hfss_patterns',
            shape=(N_TOTAL_CONFIGS, N_THETA, N_PHI),
            dtype=np.float16,
            chunks=(1, N_THETA, N_PHI),
            compression='gzip',
            compression_opts=4,
        )
        meta_ds = f.create_dataset(
            'metadata',
            shape=(N_TOTAL_CONFIGS, 6),
            dtype=np.float64,
        )
        theta_ds = f.create_dataset('theta_grid', shape=(N_THETA,), dtype=np.float32)
        phi_ds = f.create_dataset('phi_grid', shape=(N_PHI,), dtype=np.float32)

        grids_written = False

        print("\nProcessing files...")
        for file_idx in tqdm(range(1, N_FILES + 1), desc="Files"):
            matlab_path = get_file_path(MATLAB_DIR, file_idx)
            hfss_path = get_file_path(HFSS_DIR, file_idx)

            if not matlab_path.exists():
                print(f"\n  WARNING: MATLAB file not found: {matlab_path}")
                continue
            if not hfss_path.exists():
                print(f"\n  WARNING: HFSS file not found: {hfss_path}")
                continue

            # Load both files with file index for correct column names
            matlab_data = load_single_csv(matlab_path, file_index=file_idx)
            hfss_data = load_single_csv(hfss_path, file_index=file_idx)

            # Write grids once
            if not grids_written:
                theta_ds[:] = matlab_data['theta_grid']
                phi_ds[:] = matlab_data['phi_grid']
                grids_written = True

            # Use the actual column names from this file
            file_config_cols = matlab_data['config_cols']

            # Write patterns and metadata for each config
            for config_idx, col in enumerate(file_config_cols):
                global_idx = (file_idx - 1) * N_CONFIGS_PER_FILE + config_idx

                # Patterns (convert to float16 for storage)
                matlab_ds[global_idx] = matlab_data['patterns'][config_idx].astype(np.float16)
                hfss_ds[global_idx] = hfss_data['patterns'][config_idx].astype(np.float16)

                # Metadata: [dphase_x, dphase_y, matlab_phi_peak, matlab_theta_peak,
                #             hfss_phi_peak, hfss_theta_peak]
                matlab_meta = matlab_data['metadata'][col]
                hfss_meta = hfss_data['metadata'][col]
                meta_ds[global_idx] = [
                    matlab_meta['dphase_x'],     # same for both
                    matlab_meta['dphase_y'],     # same for both
                    matlab_meta['phi_peak'],
                    matlab_meta['theta_peak'],
                    hfss_meta['phi_peak'],
                    hfss_meta['theta_peak'],
                ]

    print(f"\nDone! HDF5 file saved to: {HDF5_PATH}")
    print(f"File size: {HDF5_PATH.stat().st_size / (1024**3):.2f} GB")

    # -- Verification --
    verify_data()


def verify_data():
    """Spot-check a few values against the original CSVs."""
    print("\n-- Verification --")
    rng = np.random.RandomState(42)
    check_indices = rng.choice(N_TOTAL_CONFIGS, size=5, replace=False)

    with h5py.File(str(HDF5_PATH), 'r') as f:
        for gi in check_indices:
            file_idx = gi // N_CONFIGS_PER_FILE + 1
            config_idx = gi % N_CONFIGS_PER_FILE

            # Load original with correct file index for column names
            matlab_data = load_single_csv(get_file_path(MATLAB_DIR, file_idx), file_index=file_idx)
            hfss_data = load_single_csv(get_file_path(HFSS_DIR, file_idx), file_index=file_idx)
            col = matlab_data['config_cols'][config_idx]

            # Compare a random spatial point
            ti, pi = rng.randint(0, N_THETA), rng.randint(0, N_PHI)
            h5_matlab = float(f['matlab_patterns'][gi, ti, pi])
            csv_matlab = float(matlab_data['patterns'][config_idx, ti, pi])
            h5_hfss = float(f['hfss_patterns'][gi, ti, pi])
            csv_hfss = float(hfss_data['patterns'][config_idx, ti, pi])

            # float16 introduces small rounding errors, allow ~0.1 dB tolerance
            mat_diff = abs(h5_matlab - csv_matlab)
            hfss_diff = abs(h5_hfss - csv_hfss)

            status = "OK" if (mat_diff < 0.5 and hfss_diff < 0.5) else "MISMATCH"
            print(f"  Config {gi} (file {file_idx}, {col}), "
                  f"point ({ti},{pi}): "
                  f"MATLAB diff={mat_diff:.4f}, HFSS diff={hfss_diff:.4f} [{status}]")

    print("Verification complete.")


if __name__ == '__main__':
    preprocess_all()
