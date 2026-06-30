# glp_model_matrix.py
import numpy as np

def _construct_tensor_prod(dimen_input):
    """
    Internal helper to construct the index combinations for GLP model matrix.
    Equivalent to 'construct.tensor.prod' in R.
    """
    dimen = np.sort(dimen_input)[::-1]  # sort descending
    k = len(dimen)

    nd1 = np.ones(dimen[0], dtype=int)
    nd1[dimen[0]-1] = 0
    ncol_bs = dimen[0]

    sets = np.arange(1, dimen[0] + 1).reshape(-1, 1)
    d2p = 0

    def two_dimen(d1, d2, d2p, nd1, pd12, d1sets):
        d12 = d2
        if d2 == 1:
            return {
                "d12": pd12,
                "nd1": nd1,
                "d2p": d2,
                "sets": np.hstack((d1sets, np.zeros((d1sets.shape[0], 1), dtype=int)))
            }

        d2sets = np.hstack((np.zeros((min(d1sets.shape[0], d2), d1sets.shape[1]), dtype=int),
                            np.arange(1, d2 + 1).reshape(-1, 1)))

        if d1 - d2 > 0:
            for i in range(1, d1 - d2 + 1):
                d12 += d2 * nd1[i - 1]
                one_set = d1sets.copy()
                if d1sets.shape[1] > 1 and d2p > 0:
                    one_set = d1sets[d1sets[:, -1] != d2p]
                row_sum_mask = np.sum(one_set, axis=1) == i
                expanded = np.repeat(one_set[row_sum_mask], d2, axis=0)
                appended = np.tile(np.arange(d2), len(one_set[row_sum_mask]))
                d2sets = np.vstack((d2sets, np.hstack((expanded, appended.reshape(-1, 1)))))

        nd2 = nd1.copy()
        if d1 > 1:
            for j in range(1, d1):
                nd2[j - 1] = 0
                for i in range(j, max(0, j - d2 + 1) - 1, -1):
                    nd2[j - 1] += nd1[i - 1] if i > 0 else 1

        if d2 > 1:
            nd2[d1 - 1] = nd1[d1 - 1]
            for i in range(d1 - d2 + 1, d1):
                nd2[d1 - 1] += nd1[i - 1]
        else:
            nd2[d1 - 1] = nd1[d1 - 1]

        return {"d12": d12, "nd1": nd2, "d2p": d2, "sets": d2sets}

    if k > 1:
        for i in range(1, k):
            dim_rt = two_dimen(dimen[0], dimen[i], d2p, nd1, ncol_bs, sets)
            nd1 = dim_rt["nd1"]
            ncol_bs = dim_rt["d12"]
            sets = dim_rt["sets"]
            d2p = dim_rt["d2p"]
        ncol_bs = dim_rt["d12"] + k - 1
        for i in range(1, k):
            one_row = np.zeros(sets.shape[1], dtype=int)
            one_row[i - 1] = dimen[i - 1]
            sets = np.vstack((sets, one_row))

    # Reorder to match R’s order
    index_order = np.argsort(dimen_input)[::-1]
    z = sets[:, index_order]
    return z


def glp_model_matrix(X):
    """
    Generalized Local Polynomial model matrix (Python/NumPy port).
    """
    if not isinstance(X, (list, tuple)) or len(X) == 0:
        raise ValueError("X must be a non-empty list of 2D numpy arrays.")

    n = X[0].shape[0]
    for i, Xi in enumerate(X):
        if Xi.ndim != 2:
            raise ValueError(f"X[{i}] must be 2D.")
        if Xi.shape[0] != n:
            raise ValueError("All X[j] must have the same number of rows.")

    d = np.array([Xi.shape[1] for Xi in X], dtype=int)
    k = len(X)
    z = _construct_tensor_prod(d)

    if z.shape[1] > 1:
        sort_keys = tuple(z[:, j] for j in range(z.shape[1] - 1, -1, -1))
        ord_rows = np.lexsort(sort_keys)
        z = z[ord_rows, :]

    n_terms = z.shape[0]
    B = np.ones((n, n_terms), dtype=float)

    for j in range(k):
        zj = z[:, j]
        cols_j = np.where(zj <= 0, 0, zj - 1)
        block = X[j][:, cols_j]
        B *= block

    return B
