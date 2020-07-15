# Author: Kyohei Atarashi
import numpy as np
from .kernels import _pair_predict
from .adagrad_fast_dist import _adagrad
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils import check_array
from .base import _BasePair, _PairClassifierMixin, _PairRegressorMixin
import warnings
from abc import ABCMeta, abstractmethod
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot, row_norms


try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    class NotFittedError(ValueError, AttributeError):
        pass
from lightning.impl.dataset_fast import get_dataset


class _BaseFeatureLowRankDistance(_BasePair, metaclass=ABCMeta):
    def __init__(self, degree=2, loss='squared', n_components=30, eta0=1.0,
                 alpha=1e-5, beta=1e-5, eps=1e-6, tol=1e-6,
                 fit_linear=True, warm_start=False, max_iter=1000, verbose=False,
                 random_state=None, var=0.01, callback=None, n_calls=10):
        self.degree = degree
        self.loss = loss
        self.n_components = n_components
        self.eta0 = eta0
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.tol = tol
        self.fit_linear = fit_linear
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.var = var
        self.callback = callback
        self.n_calls = n_calls

    def _augment(self, X):
        return X

    def fit(self, X, y):
        """Fit feature low rank distance model to training data.

        Parameters
        ----------
        X : array-like or sparse, shape = [n_samples, 2*n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features of each object.

        y : array-like, shape = [n_samples]
            Target values.
            Let A = X[:, :n_features] and B = X[:, n_features:], then,
            y[i] represents how much stronger A[i] is than B[i].
        
        Returns
        -------
        self : Estimator
            Returns self.
        """
        X = check_array(array=X, accept_sparse=True)
        n_features = int(X.shape[1]/2)
        A = X[:, :n_features]
        B = X[:, n_features:]
        A, B, y = self._check_A_B_y(A, B, y)
        A = self._augment(A)
        B = self._augment(B)
        row_sum_A = np.array(A.sum(axis=1)).ravel()
        row_sum_B = np.array(B.sum(axis=1)).ravel()
        dataset_A, dataset_B, loss_obj, y_pred = self._preliminary(A, B)
        if self.fit_linear:
            Linear = get_dataset(A-B, "c")
        else:
            Linear = None
        converged, self.n_iter_ = _adagrad(
            self, self.U_, self.V_, self.w_, dataset_A, dataset_B, Linear,
            row_sum_A, row_sum_B, y, self.acc_grad_U_, self.acc_grad_norm_U_,
            self.acc_grad_V_, self.acc_grad_norm_V_, self.acc_grad_w_,
            self.acc_grad_norm_w_, self.eta0, self.alpha, self.beta, self.eps,
            self.fit_linear, loss_obj, self.max_iter, self.n_iter_, self.tol,
            self.verbose, self.callback, self.n_calls)

        if not converged:
            warnings.warn("Objective did not converge. Increase max_iter.")

        return self

    def _get_output(self, A, B):
        row_sum_A = np.array(A.sum(axis=1)).ravel()
        row_sum_B = np.array(B.sum(axis=1)).ravel()

        y_pred = -self._feature_low_rank_dist(A, B, row_sum_A, row_sum_B)
        y_pred += self._feature_low_rank_dist(B, A, row_sum_B, row_sum_A)
        y_pred += safe_sparse_dot(A, self.w_)
        y_pred -= safe_sparse_dot(B, self.w_)
        return y_pred

    def _feature_low_rank_dist(self, A, B, row_sum_A, row_sum_B):
        y_pred = _pair_predict(A, B, self.U_[0], self.V_[0], self.degree)
        col_norm_sq_U = row_norms(self.U_[0].T, squared=True).ravel()
        col_norm_sq_V = row_norms(self.V_[0].T, squared=True).ravel()
        y_pred = -2*np.sum(y_pred, axis=1)
        y_pred += safe_sparse_dot(A, col_norm_sq_U) * row_sum_B
        y_pred += safe_sparse_dot(B, col_norm_sq_V) * row_sum_A
        return 0.5*y_pred

    def _predict(self, A, B):
        A = check_array(A, accept_sparse='csr', dtype=np.double)
        B = check_array(B, accept_sparse='csr', dtype=np.double)
        A = self._augment(A)
        B = self._augment(B)
        return self._get_output(A, B)

    def _preliminary(self, A, B):

        if self.degree >2:
            raise ValueError("degree > 2 (now supports only degree=2).")
        n_features = A.shape[1]
        if n_features != B.shape[1]:
            raise ValueError("A.shape != B.shape.")
        rng = check_random_state(self.random_state)
        """
        if self.fit_lower == 'explicit':
            n_orders = self.degree - 1
        else:
            n_orders = 1
        """
        n_orders = 1
        if not (self.warm_start and hasattr(self, 'U_')):
            self.U_ = rng.randn(n_orders, self.n_components, n_features)
            self.U_ *= self.var

        if not (self.warm_start and hasattr(self, 'V_')):
            self.V_ = rng.randn(n_orders, self.n_components, n_features)
            self.V_ *= self.var

        if not (self.warm_start and hasattr(self, 'w_')):
            self.w_ = np.zeros(n_features)

        if not (self.warm_start and hasattr(self, 'acc_grad_U_')):
            self.acc_grad_U_ = np.zeros(self.U_.shape)
            self.acc_grad_norm_U_ = np.zeros(self.U_.shape)

        if not (self.warm_start and hasattr(self, 'acc_grad_V_')):
            self.acc_grad_V_ = np.zeros(self.V_.shape)
            self.acc_grad_norm_V_ = np.zeros(self.V_.shape)

        if not (self.warm_start and hasattr(self, 'acc_grad_w_')):
            self.acc_grad_w_ = np.zeros(n_features)
            self.acc_grad_norm_w_ = np.zeros(n_features)

        if not (self.warm_start and hasattr(self, 'n_iter_')):
            self.n_iter_ = 1

        loss_obj = self._get_loss(self.loss)
        y_pred = self._predict(A, B)
        return get_dataset(A, "c"), get_dataset(B, "c"), loss_obj, y_pred


class FeatureLowRankDistanceRegressor(
    _BaseFeatureLowRankDistance, _PairRegressorMixin
):
    """Feature low rank distance model for regression.

    Parameters
    ----------
    degree : int >= 2 (default=2)
        Degree of the polynomial. Corresponds to the order of feature
        interactions captured by the model.
        Now only degree=2 is supported.

    n_components : int (default=30)
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    alpha : float (default=1e-5)
        Regularization amount for linear weights.

    beta : float (default=1e-5)
        Regularization amount for higher-order weights.

    eps : float (default=1e-6)
        A positive number to avoid zero-division in AdaGrad.

    tol : float (default=1e-6)
        Tolerance for the stopping condition.

    fit_linear: bool (default=True)
        Whether fit linear term or not.

    warm_start : boolean, optional (default=False)
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    max_iter : int, optional (default=1000)
        Maximum number of passes over the dataset to perform.

    verbose : boolean, optional (default=False)
        Whether to print debugging information.

    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use for
        initialization of U_ and V_.

    var : float (default=0.01)
        Variance for initialization.

    callback: callable (default=None)
        Callback function.

    n_calls: int (default=10)
        Frequency with which 'callback' is called in optimizing.

    Attributes
    ----------
    self.U_ : array, shape [n_orders, n_components, n_features]
        The learned basis functions.
        ``self.U_[0, :, :]`` is always available, and corresponds to
        interactions of order ``self.degree``.
        ``self.U_[i, :, :]`` for i > 0 corresponds to interactions of order

    self.V_ : array, shape [n_orders, n_components, n_features]
        The learned basis functions.
        ``self.V_[0, :, :]`` is always available, and corresponds to
        interactions of order ``self.degree``.
        ``self.V_[i, :, :]`` for i > 0 corresponds to interactions of order

    self.w_ : array, shape [n_features]
        The learned linear coefficients.
        It is learned only when self.fit_linear = True

    self.acc_grad_U_ : array, shape [n_orders, n_components, n_features]
        The sum of gradients for U.

    self.acc_grad_norm_U_ : array, shape [n_orders, n_components, n_features]
        The sum squares of gradients for U.

    self.acc_grad_V_ : array, shape [n_orders, n_components, n_features]
        The sum of gradients for V.

    self.acc_grad_norm_V_ : array, shape [n_orders, n_components, n_features]
        The sum of squares of gradiens for V.

    self.acc_grad_w_ : array, shape [n_features]
        The sum of gradients for w.

    self.acc_grad_norm_w_ : array, shape [n_features]
        The sum of squares of gradients for w.

    References
    ----------
    
    """

    def __init__(self, degree=2, n_components=30, eta0=1.0, alpha=1e-5,
                 beta=1e-5, eps=1e-6, tol=1e-6, fit_linear=True,
                 warm_start=False, max_iter=1000, verbose=False,
                 random_state=None, var=0.01, callback=None, n_calls=10):
        super(FeatureLowRankDistanceRegressor, self).__init__(
            degree, 'squared', n_components, eta0, alpha, beta, eps, tol,
            fit_linear, warm_start, max_iter, verbose,
            random_state, var, callback, n_calls)


class FeatureLowRankDistanceClassifier(
    _BaseFeatureLowRankDistance, _PairClassifierMixin
):
    """Feature low rank distance model for classification.

    Parameters
    ----------
    degree : int >= 2 (default=2)
        Degree of the polynomial. Corresponds to the order of feature
        interactions captured by the model.
        Now only degree=2 is supported.

    loss : {'logistic'|'squared_hinge'|'squared'|'soft_cross_entropy'|}, 
           default: 'squared_hinge'
        Which loss function to use.

        - logistic: L(y, p) = log(1 + exp(-yp))
        
        - squared hinge: L(y, p) = max(1 - yp, 0)²
        
        - squared: L(y, p) = 0.5 * (y - p)²

        - soft cross entropy L(y, p) = -y*log(sigm(p)) - (1-y)*log(1-sigm(p))

    n_components : int (default=30)
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    alpha : float (default=1e-5)
        Regularization amount for linear weights.

    beta : float (default=1e-5)
        Regularization amount for higher-order weights.

    eps : float (default=1e-6)
        A positive number to avoid zero-division in AdaGrad.

    tol : float (default=1e-6)
        Tolerance for the stopping condition.

    fit_linear: bool (default=True)
        Whether fit linear term or not.

    warm_start : boolean, optional (default=False)
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    max_iter : int, optional (default=1000)
        Maximum number of passes over the dataset to perform.

    verbose : boolean, optional (default=False)
        Whether to print debugging information.

    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use for
        initialization of U_ and V_.

    var : float (default=0.01)
        Variance for initialization.

    callback: callable (default=None)
        Callback function.

    n_calls: int (default=10)
        Frequency with which 'callback' is called in optimizing.

    Attributes
    ----------
    self.U_ : array, shape [n_orders, n_components, n_features]
        The learned basis functions.
        ``self.U_[0, :, :]`` is always available, and corresponds to
        interactions of order ``self.degree``.
        ``self.U_[i, :, :]`` for i > 0 corresponds to interactions of order

    self.V_ : array, shape [n_orders, n_components, n_features]
        The learned basis functions.
        ``self.V_[0, :, :]`` is always available, and corresponds to
        interactions of order ``self.degree``.
        ``self.V_[i, :, :]`` for i > 0 corresponds to interactions of order

    self.w_ : array, shape [n_features]
        The learned linear coefficients.
        It is learned only when self.fit_linear = True

    self.acc_grad_U_ : array, shape [n_orders, n_components, n_features]
        The sum of gradients for U.

    self.acc_grad_norm_U_ : array, shape [n_orders, n_components, n_features]
        The sum squares of gradients for U.

    self.acc_grad_V_ : array, shape [n_orders, n_components, n_features]
        The sum of gradients for V.

    self.acc_grad_norm_V_ : array, shape [n_orders, n_components, n_features]
        The sum of squares of gradiens for V.

    self.acc_grad_w_ : array, shape [n_features]
        The sum of gradients for w.

    self.acc_grad_norm_w_ : array, shape [n_features]
        The sum of squares of gradients for w.

    References
    ----------

    """

    def __init__(self, degree=2, loss='squared_hinge', n_components=30,
                 eta0=1.0, alpha=1e-5, beta=1e-5, eps=1e-6, tol=1e-6,
                 fit_linear=True, warm_start=False, max_iter=1000,
                 verbose=False, random_state=None,
                 var=0.01, callback=None, n_calls=10):
        super(FeatureLowRankDistanceClassifier, self).__init__(
            degree, loss, n_components, eta0, alpha, beta, eps, tol,
            fit_linear, warm_start, max_iter, verbose, random_state,
            var, callback, n_calls)
