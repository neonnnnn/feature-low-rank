from sklearn.utils.extmath import safe_sparse_dot
import numpy as np
from scipy.sparse import issparse
from .kernel_fast import _canova_alt_predict, _cpair_predict


def safe_power(X, degree=2):
    """Element-wise power supporting both sparse and dense data.
    Parameters
    ----------
    X : ndarray or sparse
        The array whose entries to raise to the power.
    degree : int, default: 2
        The power to which to raise the elements.
    Returns
    -------
    X_ret : ndarray or sparse
        Same shape as X, but (x_ret)_ij = (x)_ij ^ degree
    """
    if issparse(X):
        if hasattr(X, 'power'):
            return X.power(degree)
        else:
            # old scipy
            X = X.copy()
            X.data **= degree
            return X
    else:
        return X ** degree


def _D(X, P, degree=2):
    """The "replacement" part of the homogeneous polynomial kernel.
    D[i, j] = sum_k [(X_ik * P_jk) ** degree]
    """
    return safe_sparse_dot(safe_power(X, degree),
                           safe_power(P, degree).T,
                           dense_output=True)


def anova_alt(degree, dptable_anova, dptable_poly):
    dptable_anova[0] = 1
    dptable_poly[0] = 1

    for m in range(1, degree+1):
        temp = 0.
        sign = 1.
        for t in range(1, m+1):
            temp += sign * dptable_anova[m-t]*dptable_poly[t]
            sign *= -1
        dptable_anova[m] = temp / m

    return dptable_anova


def _anova_predict(X, P, kernel, degree=2):
    N = X.shape[0]
    if kernel == 'anova':
        Ds = [np.ones((N, P.shape[0]))]
        Ds += [_D(X, P, i) for i in range(1, degree+1)]
        # K = anova_alt(degree, np.zeros((degree+1, N, P.shape[0])), Ds)
        K = np.asarray(_canova_alt_predict(np.asarray(Ds), degree))
        print(K.shape)
    else:
        raise ValueError("Unsuppported kernel: Use 'anova'".format(kernel))
    return K


def _pair_predict(A, B, U, V, degree=2):

    N = A.shape[0]
    Ds1 = [np.ones((N, U.shape[0]))]
    Ds1 += [_D(A, U, i) for i in range(1, degree)]
    Ds2 = [np.ones((N, V.shape[0]))]
    Ds2 += [_D(B, V, i) for i in range(1, degree)]
    # K = anova_alt(degree, np.zeros((degree+1, N, P.shape[0])), Ds)
    K = np.asarray(_cpair_predict(np.atleast_2d(np.asarray(Ds1)),
                                  np.atleast_2d(np.asarray(Ds2)),
                                  degree))
    return K
