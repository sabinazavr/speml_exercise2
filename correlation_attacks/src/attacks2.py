# correlation_attacks/src/attacks2.py


import pandas as pd
import numpy as np
from collections import Counter

def compute_joint_distribution(df: pd.DataFrame, col_p: str, col_q: str) -> pd.DataFrame:
    """
    Returns the joint frequency table J_pq(a, b) = P(col_p=a, col_q=b).
    """
    # count occurrences of each (a,b)
    joint = df.groupby([col_p, col_q]).size().unstack(fill_value=0)
    return joint / joint.values.sum()

def column_wise_attack(df_fp, true_joint: dict, threshold: float = 0.01, pbar=None) -> pd.DataFrame:
    """
    Alg.1: For each pair (p,q) and each cell (a,b),
    if |empirical - true| >= threshold, flip that LSB back.
    """
    if hasattr(df_fp, 'dataframe'):
        df = df_fp.dataframe.copy()
    else:
        df = df_fp.copy()

    iterator = true_joint.items()
    if pbar:
        iterator = pbar(iterator, desc="Col-wise attack")

    # list of suspicious attributes
    P = {}

    for (p, q), J_true in iterator:
        J_emp = compute_joint_distribution(df, p, q)
        # align indices
        J_true = J_true.reindex(index=J_emp.index,
                                columns=J_emp.columns,
                                fill_value=0)
        # find all (a,b) cells that deviate
        for a in J_emp.index:
            for b in J_emp.columns:
                if abs(J_emp.at[a, b] - J_true.at[a, b]) >= threshold:
                    # all rows where df[p]==a AND df[q]==b
                    mask = (df[p] == a) & (df[q] == b)
                    for i in df.index[mask]:
                        P.setdefault(i, []).append(p)
                        P[i].append(q)

    H = []
    for i, attrs in P.items():
        if not attrs:
            continue
        # most common attribute in attrs
        mode_attr, _ = Counter(attrs).most_common(1)[0]
        H.append((i, mode_attr))

    # perform flips
    for i, attr in H:
        val = int(df.at[i, attr])
        df.at[i, attr] = val ^ 1

    if hasattr(df_fp, 'dataframe'):
        df_fp.dataframe = df
        return df_fp
    return df


def compute_row_similarity(df: pd.DataFrame, i: int, j: int) -> float:
    """
    Simple Hamming-based similarity: fraction of equal attributes.
    """
    dist = (df.loc[i] != df.loc[j]).sum()
    return np.exp(-dist)

def row_wise_attack(df_fp, clusters: dict, true_sims: dict, threshold: float = 0.1, pbar=None) -> pd.DataFrame:
    """
    Alg.2: For each row i, compare sum of deviations in its cluster.
    If > threshold, flip all its bits back.
    """
    if hasattr(df_fp, 'dataframe'):
        df = df_fp.dataframe.copy()
    else:
        df = df_fp.copy()

    iterator = clusters.items()
    if pbar:
        iterator = pbar(iterator, desc="Row-wise attack")

    for i, members in iterator:
        dev = 0.0
        for j in members:
            if j == i:
                continue
            dev += abs(compute_row_similarity(df, i, j) - true_sims[(i, j)])
        if dev >= threshold:
            # revert every column's LSB for row i
            for col in df.columns:
                val = int(df.at[i, col])
                df.at[i, col] = val ^ 1

    if hasattr(df_fp, 'dataframe'):
        df_fp.dataframe = df
        return df_fp
    return df
