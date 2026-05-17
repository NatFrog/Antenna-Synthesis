import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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

    x=np.asarray(x,dtype=float)

    if len(x)==0:
        return x

    return x-np.nanmax(x)


def clean_pattern(theta,pattern):

    theta=np.asarray(theta,dtype=float)
    pattern=np.asarray(pattern,dtype=float)

    # remove NaNs
    valid=(
        np.isfinite(theta)
        &
        np.isfinite(pattern)
    )

    theta=theta[valid]
    pattern=pattern[valid]

    if len(theta)<3:
        return None,None

    # sort by theta
    idx=np.argsort(theta)

    theta=theta[idx]
    pattern=pattern[idx]

    # remove duplicate theta values
    unique_theta=np.unique(theta)

    averaged=[]

    for t in unique_theta:

        averaged.append(
            np.mean(
                pattern[
                    theta==t
                ]
            )
        )

    averaged=np.array(averaged)

    if len(unique_theta)<3:
        return None,None

    return unique_theta,averaged


def interpolate_pattern(theta,
                        pattern,
                        theta_common):

    theta,pattern=clean_pattern(
        theta,
        pattern
    )

    if theta is None:
        return None

    try:

        f=interp1d(
            theta,
            pattern,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        y=f(theta_common)

        if np.any(np.isnan(y)):
            return None

        return y

    except Exception:

        return None


def metrics(a,b,theta):

    rmse=np.sqrt(
        np.mean((a-b)**2)
    )

    corr=np.corrcoef(
        a,b
    )[0,1]

    maxdiff=np.max(
        np.abs(a-b)
    )

    peak_a=theta[np.argmax(a)]
    peak_b=theta[np.argmax(b)]

    peak_shift=np.abs(
        peak_a-peak_b
    )

    return rmse,corr,maxdiff,peak_shift


# ==========================================================
# file lists
# ==========================================================

files4=sorted(
    glob.glob(
        os.path.join(
            path_4x4,
            "patterns_global_*.csv"
        )
    )
)

filesSub=sorted(
    glob.glob(
        os.path.join(
            path_sub,
            "patterns_global_*.csv"
        )
    )
)

theta_common=np.linspace(
    -90,
    90,
    1000
)

RMSE=[]
CORR=[]
MAXD=[]
PEAK=[]

skipped=0

# ==========================================================
# main loop
# ==========================================================

for file4,fileSub in zip(files4,filesSub):

    print(
        "Processing:",
        os.path.basename(file4)
    )

    df4=pd.read_csv(
        file4,
        low_memory=False
    )

    dfSub=pd.read_csv(
        fileSub,
        low_memory=False
    )

    parent_samples=df4.columns[2:]

    for sample in parent_samples:

        try:

            theta4=df4.iloc[
                4:,
                0
            ]

            parent=df4[
                sample
            ].iloc[4:]

            parent=normalize(
                parent
            )

            parent_interp=interpolate_pattern(
                theta4,
                parent,
                theta_common
            )

            if parent_interp is None:

                skipped+=1
                continue

            # --------------------
            # average sub-blocks
            # --------------------

            subpatterns=[]

            for b in range(4):

                name=f"{sample}_b{b}"

                if name not in dfSub.columns:
                    continue

                thetaSub=dfSub.iloc[
                    4:,
                    0
                ]

                p=dfSub[
                    name
                ].iloc[4:]

                p=normalize(p)

                p_interp=interpolate_pattern(
                    thetaSub,
                    p,
                    theta_common
                )

                if p_interp is not None:

                    subpatterns.append(
                        p_interp
                    )

            if len(subpatterns)==0:

                skipped+=1
                continue

            avg_sub=np.mean(
                subpatterns,
                axis=0
            )

            rmse,corr,maxd,peak=metrics(
                parent_interp,
                avg_sub,
                theta_common
            )

            RMSE.append(rmse)
            CORR.append(corr)
            MAXD.append(maxd)
            PEAK.append(peak)

        except Exception as e:

            print(
                "Skipped:",
                sample,
                e
            )

            skipped+=1


# ==========================================================
# summary
# ==========================================================

print("\n")
print("="*60)
print("SIMPLIFIED G0b RESULTS")
print("="*60)

print(
    f"Valid comparisons: {len(RMSE)}"
)

print(
    f"Skipped samples: {skipped}"
)

print(
    f"Mean RMSE: {np.mean(RMSE):.3f}"
)

print(
    f"Mean Correlation: {np.mean(CORR):.4f}"
)

print(
    f"Mean Max Difference: {np.mean(MAXD):.3f}"
)

print(
    f"Mean Peak Shift: {np.mean(PEAK):.3f}"
)