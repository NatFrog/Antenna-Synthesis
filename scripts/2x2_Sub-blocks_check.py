import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

path_4x4 = PROJECT_ROOT / "datasets_4x4_hfss" / "datasets_4x4_hfss"
path_sub = PROJECT_ROOT / "datasets_2x2Sub-blocks_hfss"

# ==========================================================
# helper functions
# ==========================================================

def normalize(x):

    x=np.array(x,dtype=float)

    return x-np.max(x)


def compute_metrics(x,y):

    rmse=np.sqrt(
        np.mean((x-y)**2)
    )

    corr=np.corrcoef(
        x,y
    )[0,1]

    maxdiff=np.max(
        np.abs(x-y)
    )

    return rmse,corr,maxdiff


# ==========================================================
# find files
# ==========================================================

files4=sorted(
    glob.glob(
        os.path.join(
            path_4x4,
            "patterns_global_*.csv"
        )
    )
)

filessub=sorted(
    glob.glob(
        os.path.join(
            path_sub,
            "patterns_global_*.csv"
        )
    )
)

print(
    f"4x4 files: {len(files4)}"
)

print(
    f"Sub-block files: {len(filessub)}"
)

# ==========================================================
# aggregate statistics
# ==========================================================

all_rmse=[]
all_corr=[]
all_max=[]

missing=[]

# ==========================================================
# process all files
# ==========================================================

for f4,fsub in zip(files4,filessub):

    print(
        "\nProcessing:",
        os.path.basename(f4)
    )

    parent=pd.read_csv(
        f4,
        low_memory=False
    )

    sub=pd.read_csv(
        fsub,
        low_memory=False
    )

    parent_samples=parent.columns[2:]
    sub_samples=sub.columns[2:]

    # ------------------------------------
    # verify sub-block existence
    # ------------------------------------

    for s in parent_samples:

        expected=[

            f"{s}_b0",
            f"{s}_b1",
            f"{s}_b2",
            f"{s}_b3"
        ]

        for e in expected:

            if e not in sub_samples:

                missing.append(e)

    # ------------------------------------
    # visualize first sample
    # ------------------------------------

    sample=parent_samples[0]

    plt.figure(figsize=(10,5))

    theta=np.array(
        sub.iloc[4:,0],
        dtype=float
    )

    for b in range(4):

        name=f"{sample}_b{b}"

        if name not in sub.columns:
            continue

        pattern=np.array(
            sub[name].iloc[4:],
            dtype=float
        )

        pattern=normalize(pattern)

        plt.plot(
            theta,
            pattern,
            label=f"b{b}"
        )

    plt.xlabel(
        "Theta (deg)"
    )

    plt.ylabel(
        "Normalized Gain (dB)"
    )

    plt.title(
        f"{sample} sub-block comparison"
    )

    plt.grid()

    plt.legend()

    plt.show()

    # ------------------------------------
    # compute pairwise similarity
    # ------------------------------------

    blocks=[]

    for b in range(4):

        name=f"{sample}_b{b}"

        if name in sub.columns:

            p=np.array(
                sub[name].iloc[4:],
                dtype=float
            )

            p=normalize(p)

            blocks.append(p)

    for i in range(len(blocks)):

        for j in range(i+1,len(blocks)):

            rmse,corr,maxd=compute_metrics(
                blocks[i],
                blocks[j]
            )

            all_rmse.append(
                rmse
            )

            all_corr.append(
                corr
            )

            all_max.append(
                maxd
            )


# ==========================================================
# summary
# ==========================================================

print("\n")
print("="*50)
print("VALIDATION SUMMARY")
print("="*50)

if len(missing)==0:

    print(
        "All parent samples have b0-b3."
    )

else:

    print(
        "Missing sub-blocks:"
    )

    for m in missing:

        print(m)

print(
    f"\nMean RMSE: {np.mean(all_rmse):.3f}"
)

print(
    f"Mean Correlation: {np.mean(all_corr):.4f}"
)

print(
    f"Mean Max Difference: {np.mean(all_max):.3f}"
)