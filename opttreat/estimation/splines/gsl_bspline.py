# gsl_bspline.py
import numpy as np
from scipy.interpolate import BSpline

def bs_des(x, degree=3, nbreak=2, deriv=0, x_min=None, x_max=None, knots=None,
           extrapolate=False):
    """
    Construct B-spline basis matrix (and its derivatives if requested).
    Equivalent to bs.des() in R npiv package.

    When ``extrapolate=True``, evaluation points outside ``[x_min, x_max]`` are
    extended polynomially from the boundary pieces, matching the behaviour of
    R's ``crs::gsl.bs`` / ``predict.gsl.bs``. With ``extrapolate=False`` (the
    default) such points return ``nan``.
    """

    x = np.asarray(x).ravel()
    n = len(x)

    # --- Ensure domain range is fixed (avoid truncation) ---
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    if x_min >= x_max:
        raise ValueError("x_min must be less than x_max")

    # --- Use provided knots or compute evenly spaced ones ---
    if knots is None:
        knots = np.linspace(x_min, x_max, nbreak)
    knots = np.asarray(knots)

    # --- Extend the knot vector by repeating endpoints (for boundary smoothness) ---
    t = np.concatenate((
        np.repeat(knots[0], degree),
        knots,
        np.repeat(knots[-1], degree)
    ))

    n_basis = len(t) - degree - 1
    B = np.zeros((n, n_basis))

    # --- Build spline columns (extrapolation optional) ---
    for i in range(n_basis):
        coeff = np.zeros(n_basis)
        coeff[i] = 1.0
        spline = BSpline(t, coeff, degree, extrapolate=extrapolate)
        if deriv > 0:
            B[:, i] = spline(x, nu=deriv)
        else:
            B[:, i] = spline(x)

    return B


def gsl_bspline(x,
           degree=3,
           nbreak=2,
           deriv=0,
           x_min=None,
           x_max=None,
           intercept=False,
           knots=None,
           extrapolate=False):
    """
    Generalized B-spline generator (R's gsl.bs).
    Builds spline basis with given degree and number of segments.
    """

    x = np.asarray(x).ravel()

    if degree <= 0:
        raise ValueError("degree must be positive")
    if deriv < 0:
        raise ValueError("deriv must be non-negative")
    if nbreak <= 1:
        raise ValueError("nbreak must be at least 2")

    # --- Build spline basis with fixed domain ---
    B = bs_des(x, degree, nbreak, deriv, x_min, x_max, knots, extrapolate)

# --- Keep intercept by default (match R npiv) ---
# if not intercept:
#     B = B[:, 1:]
    attr = {
        "degree": degree,
        "nbreak": nbreak,
        "deriv": deriv,
        "x_min": x_min,
        "x_max": x_max,
        "intercept": intercept,
        "knots": knots,
        "extrapolate": extrapolate
    }

    return B, attr


def predict_gsl_bspline(B_attr, newx):
    """
    Evaluate a previously created spline basis on new data.
    Equivalent to predict.gsl.bs() in R npiv.
    Allows extrapolation beyond training range.
    """

    degree = B_attr["degree"]
    nbreak = B_attr["nbreak"]
    deriv = B_attr["deriv"]
    x_min = B_attr["x_min"]
    x_max = B_attr["x_max"]
    intercept = B_attr["intercept"]
    knots = B_attr["knots"]
    extrapolate = B_attr.get("extrapolate", False)

    # --- Avoid clipping and allow smooth extrapolation ---
    newx = np.asarray(newx).ravel()

    B_new = bs_des(
        newx,
        degree=degree,
        nbreak=nbreak,
        deriv=deriv,
        x_min=x_min,
        x_max=x_max,
        knots=knots,
        extrapolate=extrapolate
    )

    if not intercept:
        B_new = B_new[:, 1:]

    return B_new
