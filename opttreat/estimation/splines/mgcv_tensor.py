import numpy as np

def tensor_prod_model_matrix(X_list):
    """
    Equivalent to mgcv::tensor.prod.model.matrix in R.

    Parameters
    ----------
    X_list : list of ndarray
        Each element is a (n x k_i) basis matrix (e.g. from gsl_bs).

    Returns
    -------
    ndarray
        Tensor-product model matrix of shape (n, prod(k_i)).
        Each row corresponds to the Kronecker product of the basis rows.
    """
    if not isinstance(X_list, (list, tuple)):
        raise TypeError("X_list must be a list of numpy arrays")

    k = len(X_list)
    if k == 0:
        raise ValueError("Empty list provided")

    # All must have the same number of rows
    n_rows = X_list[0].shape[0]
    for X in X_list:
        if X.shape[0] != n_rows:
            raise ValueError("All matrices in X_list must have the same number of rows")

    # Initialize with first matrix
    result = X_list[0]

    # Multiply column-wise with successive matrices
    for i in range(1, k):
        Xi = X_list[i]
        # Kronecker product of columns, not rows
        result = np.einsum('ij,ik->ijk', result, Xi).reshape(n_rows, -1)

    return result

