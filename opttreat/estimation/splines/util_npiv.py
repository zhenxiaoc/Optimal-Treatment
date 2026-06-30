import numpy as np

def is_fullrank(x: np.ndarray) -> bool:
    """
    Check if a matrix is full rank.
    Equivalent to R's is.fullrank().
    """
    x = np.asarray(x, dtype=float)
    xtx = x.T @ x  # crossprod in R
    eigvals = np.linalg.eigvalsh(xtx)
    eigvals = np.sort(eigvals)[::-1]  # descending order

    if eigvals[0] <= 0:
        return False

    tol = max(x.shape) * np.max(np.sqrt(np.abs(eigvals))) * np.finfo(float).eps
    return abs(eigvals[-1] / eigvals[0]) > tol


def RSQfunc(y: np.ndarray, y_pred: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Weighted R-squared function.
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if weights is not None:
        w = np.sqrt(weights)
        y = y * w
        y_pred = y_pred * w

    y_mean = np.mean(y)
    num = np.sum((y - y_mean) * (y_pred - y_mean)) ** 2
    den = np.sum((y - y_mean) ** 2) * np.sum((y_pred - y_mean) ** 2)

    return num / den if den != 0 else np.nan


def sqrtm2(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable positive definite matrix square root.
    """
    eigvals, eigvecs = np.linalg.eigh(x)
    lambda_sqrt = np.sqrt(np.maximum(eigvals, 0))
    return eigvecs @ np.diag(lambda_sqrt) @ eigvecs.T


def NZD(a: np.ndarray) -> np.ndarray:
    """
    Avoid division by zero (equivalent to R's NZD()).
    """
    a = np.asarray(a, dtype=float)
    eps = np.finfo(float).eps
    return np.where(a < 0, np.minimum(-eps, a), np.maximum(eps, a))


def dimbs(basis: str = "additive", degree=None, segments=None) -> int:
    """
    Compute the dimension of a multivariate spline basis
    without explicitly constructing it.
    Equivalent to R's dimbs().
    """
    if basis not in ["additive", "glp", "tensor"]:
        raise ValueError("basis must be one of {'additive', 'glp', 'tensor'}")

    K = np.column_stack([degree, segments])

    def two_dimen(d1, d2, nd1, pd12):
        if d2 == 1:
            return {'d12': pd12, 'nd1': nd1}

        d12 = d2
        for i in range(1, d1 - d2 + 1):
            d12 += d2 * nd1[i - 1]

        for i in range(2, d2 + 1):
            d12 += i * nd1[d1 - i]

        d12 += nd1[d1 - 1]  # max number

        nd2 = nd1.copy()
        if d1 > 1:
            for j in range(1, d1):
                nd2[j - 1] = 0
                for i in range(j, max(0, j - d2 + 1) - 1, -1):
                    if i > 0:
                        nd2[j - 1] += nd1[i - 1]
                    else:
                        nd2[j - 1] += 1

        if d2 > 1:
            nd2[d1 - 1] = nd1[d1 - 1]
            for i in range(d1 - d2 + 1, d1):
                nd2[d1 - 1] += nd1[i - 1]
        else:
            nd2[d1 - 1] = nd1[d1 - 1]

        return {'d12': d12, 'nd1': nd2}

    ncol_bs = 0

    if basis == "additive":
        if np.any(K[:, 0] > 0):
            ncol_bs = np.sum(np.sum(K[K[:, 0] != 0, :], axis=1) - 1)

    elif basis == "glp":
        dimen = np.sum(K[K[:, 0] != 0, :], axis=1) - 1
        dimen = dimen[dimen > 0]
        dimen = np.sort(dimen)[::-1]
        k = len(dimen)
        if k == 0:
            ncol_bs = 0
        else:
            nd1 = np.ones(dimen[0])
            nd1[int(dimen[0]) - 1] = 0
            ncol_bs = dimen[0]
            if k > 1:
                for i in range(1, k):
                    dim_rt = two_dimen(int(dimen[0]), int(dimen[i]), nd1, ncol_bs)
                    nd1 = dim_rt['nd1']
                    ncol_bs = dim_rt['d12']
                ncol_bs = dim_rt['d12'] + k - 1

    elif basis == "tensor":
        if np.any(K[:, 0] > 0):
            ncol_bs = np.prod(np.sum(K[K[:, 0] != 0, :], axis=1))

    return int(ncol_bs)
