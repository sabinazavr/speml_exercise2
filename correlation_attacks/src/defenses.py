# correlation_attacks/src/defenses.py

import numpy as np
import pandas as pd
from tqdm.auto import tqdm 
from scipy.spatial.distance import cdist
from .attacks2 import compute_joint_distribution

def _sinkhorn(a, b, C, reg, num_iters=1000, tol=1e-9):
    K = np.exp(-C / reg)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(num_iters):
        u_prev = u.copy()
        u = a / (K @ v)
        v = b / (K.T @ u)
        if np.linalg.norm(u - u_prev, 1) < tol:
            break
    return np.diag(u) @ K @ np.diag(v)


def defend_column_wise(
    df_fp: pd.DataFrame,
    reference_joints: dict,
    lambda_p: float = 500,
    tau_col: float = 1e-4,
) -> pd.DataFrame:
    """
    Algorithm 3 (Dfscol) with reindexing and progress bar.
    """
    df = df_fp.copy()
    attrs = df.columns.tolist()

    # 1) detect which attributes need smoothing
    Q = set()
    for (p, q), J_ref in tqdm(reference_joints.items(),
                              desc="Checking joint deltas", total=len(reference_joints)):
        # recompute empirical joint on the current (possibly attacked) df
        J_emp = compute_joint_distribution(df, p, q)
        # reindex to exactly match J_ref shape
        J_emp = J_emp.reindex(index=J_ref.index,
                              columns=J_ref.columns,
                              fill_value=0)
        delta = np.linalg.norm(J_emp.values - J_ref.values, ord='fro')
        if delta >= tau_col:
            Q.add(p)
            Q.add(q)

    # 2) for each flagged attribute p, transport its marginal back toward the reference
    for p in tqdm(Q, desc="Smoothing columns", total=len(Q)):
        # pick some q' so that (p,q') in reference_joints
        q_prime = next(q for (x, q) in reference_joints if x == p)
        J_ref   = reference_joints[(p, q_prime)]

        # empirical marginal a, reference marginal b
        J_emp   = compute_joint_distribution(df, p, q_prime)
        a       = J_emp.sum(axis=1).reindex(J_ref.index, fill_value=0).values
        b       = J_ref.sum(axis=1).values

        # cost matrix on the integer codes
        vals     = J_ref.index.to_numpy()  # assumed sorted codes
        C        = cdist(vals.reshape(-1,1), vals.reshape(-1,1), metric='cityblock')

        # solve Sinkhorn
        G = _sinkhorn(a, b, C, reg=1.0/lambda_p)

        # randomly reassign frac=G[i,j] of non-fingerprinted tuples value i→j
        for i, orig_val in enumerate(vals):
            idxs = df.index[df[p] == orig_val]
            n    = len(idxs)
            if n == 0:
                continue
            for j, new_val in enumerate(vals):
                frac = G[i, j]
                m    = int(np.floor(frac * n))
                if m > 0 and new_val != orig_val:
                    chosen = np.random.choice(idxs, size=m, replace=False)
                    df.loc[chosen, p] = new_val

    return df


def defend_row_wise(
    df_fp: pd.DataFrame,
    clusters: dict,
    true_sims: dict,
    tau_row: float = 0.1,
    gamma: float = None
) -> pd.DataFrame:
    """
    Algorithm 4 (Dfsrow): for each community c, select up to γ·|c| non‐fingerprinted
    records whose empirical s_ij deviate most from true_sims, then re‐project them
    to the cluster centroid to confuse the attacker
    """
    df = df_fp.copy()
    M = len(df)
    if gamma is None:
        # recover γ from insertion parameters
        # we assume 1/γ = fraction of marked tuples
        fingerprinted = df_fp.ne(df).any(axis=1).sum()
        gamma = fingerprinted / M

    for members in clusters.values():
        # compute per‐row deviation from true_sims
        devs = []
        for i in members:
            total_dev = 0.0
            for j in members:
                if i == j:
                    continue
                # empirical similarity on df_fp
                dist_emp = (df_fp.iloc[i] != df_fp.iloc[j]).sum()
                s_emp   = np.exp(-dist_emp)
                total_dev += abs(s_emp - true_sims[(i, j)])
            devs.append((total_dev, i))

        devs.sort(reverse=True)
        k = int(np.floor(gamma * len(members)))
        to_fix = {i for _, i in devs[:k]}

        sub = df.loc[members]
        centroid = sub.mean().round().astype(int)
        for i in to_fix:
            df.loc[i] = centroid

    return df
