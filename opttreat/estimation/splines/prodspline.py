import numpy as np
from .gsl_bspline import gsl_bspline, predict_gsl_bspline
from .mgcv_tensor import tensor_prod_model_matrix
from .glp_model_matrix import glp_model_matrix
from .util_npiv import dimbs


def prodspline(
    x,
    z=None,
    K=None,
    I=None,
    xeval=None,
    zeval=None,
    knots="quantiles",
    basis="additive",
    x_min=None,
    x_max=None,
    deriv_index=1,
    deriv=0,
    extrapolate=False
):
    """
    Python version of the R function `prodspline`.
    Builds multivariate spline bases (additive, tensor, or GLP).

    Parameters
    ----------
    x : ndarray
        Predictor matrix.
    z : ndarray, optional
        Discrete/categorical variables.
    K : ndarray
        2-column matrix: spline degree and segments per variable.
    I : ndarray, optional
        Binary indicator for inclusion of z variables.
    xeval : ndarray, optional
        Evaluation grid (like R's xeval).
    zeval : ndarray, optional
        Evaluation data for z variables.
    knots : str
        "uniform" or "quantiles"
    basis : str
        "additive", "tensor", or "glp"
    x_min, x_max : list or ndarray, optional
        Domain boundaries for each variable.
    deriv_index : int
        Variable index for derivative (1-based).
    deriv : int
        Order of derivative (0 = none).
    extrapolate : bool
        If True, evaluation points outside each variable's training range are
        extended polynomially from the boundary spline pieces (matching R's
        ``crs::gsl.bs``). If False (default), such points yield ``nan``.

    Returns
    -------
    P : ndarray
        Spline basis matrix.
    dim_P_no_tensor : int
        Number of non-tensor basis columns.
    """

    # --- Validate inputs ---
    if x is None or K is None:
        raise ValueError("Must provide both x and K.")
    if not isinstance(K, np.ndarray) or K.shape[1] != 2:
        raise ValueError("K must be a two-column matrix.")

    x = np.asarray(x, dtype=float)
    n, num_x = x.shape
    num_K = K.shape[0]

    if deriv < 0:
        raise ValueError("deriv must be non-negative.")
    if deriv_index < 1 or deriv_index > num_x:
        raise ValueError("deriv_index is invalid.")
    if num_K != num_x:
        raise ValueError("Dimension mismatch between x and K.")

    # --- Initialize list for spline components ---
    tp_list = []

    for i in range(num_x):
        if K[i, 0] > 0:
            nbreak = int(K[i, 1]) + 1
            degree = int(K[i, 0])

            # --- Knot selection ---
            if isinstance(knots, str) and knots == "uniform":
                knot_vec = None
            else:
                probs = np.linspace(0, 1, nbreak)
                knot_vec = np.quantile(x[:, i], probs)
                delta = 1e-10 * (x[:, i].max() - x[:, i].min())
                knot_vec = knot_vec + np.linspace(0, delta, len(knot_vec))

            # --- Handle x_min/x_max properly (scalar, list, or vector) ---
            if x_min is None:
                xmin_i = np.min(x[:, i])
            elif np.ndim(x_min) == 0 or len(np.atleast_1d(x_min)) == 1:
                xmin_i = float(np.atleast_1d(x_min)[0])
            else:
                xmin_i = float(x_min[i])

            if x_max is None:
                xmax_i = np.max(x[:, i])
            elif np.ndim(x_max) == 0 or len(np.atleast_1d(x_max)) == 1:
                xmax_i = float(np.atleast_1d(x_max)[0])
            else:
                xmax_i = float(x_max[i])

            # --- Build the B-spline base ---
            B_base, B_attr = gsl_bspline(
                x[:, [i]],
                degree=degree,
                nbreak=nbreak,
                knots=knot_vec,
                deriv=deriv if (i + 1 == deriv_index and deriv != 0) else 0,
                x_min=xmin_i,
                x_max=xmax_i,
                intercept=(basis not in ["additive", "glp"]),
                extrapolate=extrapolate
            )

            # --- Evaluate the spline basis at xeval (if provided) ---
            B = predict_gsl_bspline(
                B_attr,
                newx=xeval[:, [i]] if xeval is not None else x[:, [i]]
            )

            tp_list.append(B)

    # --- Combine bases depending on model type ---
    if len(tp_list) == 0:
        P = np.ones((n, 1))
        dim_P_no_tensor = 0
    elif len(tp_list) == 1:
        P = tp_list[0]
        dim_P_no_tensor = P.shape[1]
    else:
        P = tp_list[0]
        for i in range(1, len(tp_list)):
            P = np.column_stack((P, tp_list[i]))
        dim_P_no_tensor = P.shape[1]

        if basis == "tensor":
            P = tensor_prod_model_matrix(tp_list)
        elif basis == "glp":
            P = glp_model_matrix(tp_list)

    P = np.asarray(P, dtype=float)
    return P
