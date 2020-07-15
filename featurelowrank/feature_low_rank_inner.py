# Author: Kyohei Atarashi
import numpy as np
from .kernels import _pair_predict
from .cd_fast_inner import _cd
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


class _BaseFeatureLowRankInner(_BasePair, metaclass=ABCMeta):
    def __init__(self, degree=2, loss='squared', n_components=30, 
                 alpha=1e-5, beta=1e-5, tol=1e-6, fit_lower='explicit',
                 fit_linear=True, warm_start=False, max_iter=1000, verbose=False,
                 random_state=None, mean=True, var=0.01, callback=None, 
                 n_calls=10):
        self.degree = degree
        self.loss = loss
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.fit_lower = fit_lower
        self.fit_linear = fit_linear
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.mean = mean
        self.var = var
        self.callback = callback
        self.n_calls = n_calls

    def _augment(self, X):
        if self.fit_lower == 'augment':
            for _ in range(self.degree - 2):
                X = add_dummy_feature(X, value=1)
        return X

    def fit(self, X, y):
        """Fit feature low rank inner model to training data.

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
        col_norm_sq = row_norms((A-B).T, squared=True)

        dataset_A, dataset_B, loss_obj, y_pred = self._preliminary(A, B)
        converged, self.n_iter_ = _cd(
            self, self.U_, self.V_, self.w_, dataset_A, dataset_B,
            y, y_pred, col_norm_sq, self.degree, self.alpha, self.beta,
            self.fit_lower == 'explicit', self.fit_linear, self.mean,
            loss_obj, self.max_iter, self.tol,
            self.verbose, self.callback, self.n_calls)

        if not converged:
            warnings.warn("Objective did not converge. Increase max_iter.")

        return self

    def _get_output(self, A, B):
        y_pred = self._feature_low_rank_inner(A, B)
        y_pred -= self._feature_low_rank_inner(B, A)
        y_pred += safe_sparse_dot(A, self.w_)
        y_pred -= safe_sparse_dot(B, self.w_)
        return y_pred

    def _feature_low_rank_inner(self, A, B):
        y_pred = _pair_predict(A, B, self.U_[0], self.V_[0], self.degree)
        y_pred = np.sum(y_pred, axis=1)
        if self.fit_lower == 'explicit':
            for i in range(2, self.degree):
                pred = _pair_predict(A, B, self.U_[self.degree-i],
                                     self.V_[self.degree-i], i)
                y_pred += np.sum(pred, axis=1)
        return y_pred

    def _predict(self, A, B):
        row_sum_A = A.sum(axis=1)
        row_sum_B = B.sum(axis=1)
        A = check_array(A, accept_sparse='csc', dtype=np.double)
        B = check_array(B, accept_sparse='csc', dtype=np.double)
        A = self._augment(A)
        B = self._augment(B)
        return self._get_output(A, B)

    def _preliminary(self, A, B):
        n_features = A.shape[1]  # augmented
        if n_features != B.shape[1]:
            raise ValueError("A.shape != B.shape.")
        rng = check_random_state(self.random_state)

        if self.fit_lower == 'explicit':
            n_orders = self.degree - 1
        else:
            n_orders = 1
        if not (self.warm_start and hasattr(self, 'U_')):
            self.U_ = rng.randn(n_orders, self.n_components, n_features)
            self.U_ *= self.var

        if not (self.warm_start and hasattr(self, 'V_')):
            self.V_ = rng.randn(n_orders, self.n_components, n_features)
            self.V_ *= self.var

        if not (self.warm_start and hasattr(self, 'w_')):
            self.w_ = np.zeros(n_features)

        loss_obj = self._get_loss(self.loss)
        y_pred = self._predict(A, B)
        return get_dataset(A, "fortran"), get_dataset(B,"fortran"), loss_obj, y_pred


class FeatureLowRankInnerRegressor(
    _BaseFeatureLowRankInner, _PairRegressorMixin
):
    """Feature low rank inner model for regression.

    Parameters
    ----------
    degree : int >= 2 (default=2)
        Degree of the polynomial. Corresponds to the order of feature
        interactions captured by the model.

    n_components : int (default=30)
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    alpha : float (default=1e-5)
        Regularization amount for linear weights.
 
    beta : float (default=1e-5)
        Regularization amount for higher-order weights.

    tol : float (default=1e-6)
        Tolerance for the stopping condition.

    fit_lower : {'explicit'|'augment'|None}, (default='explicit')
        Whether and how to fit lower-order, non-homogeneous terms.
        - 'explicit': fits a separate P directly for each lower order.
        - 'augment': adds the required number of dummy columns (columns
           that are 1 everywhere) in order to capture lower-order terms.
           Adds ``degree - 2`` columns if ``fit_linear`` is true, or
           ``degree - 1`` columns otherwise, to account for the linear term.
        - None: only learns weights for the degree given.  If ``degree == 3``,
          for example, the model will only have weights for third-order
          feature interactions.

    fit_linear: bool (default=True)
        Whether fit linear term or not.

    warm_start : boolean, optional, (default=False)
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    max_iter : int, optional, (default=1000)
        Maximum number of passes over the dataset to perform.

    verbose : boolean, optional (default=False)
        Whether to print debugging information.

    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use for
        initialization of B_ and C_.

    mean : bool (default=True)
        Whether loss is mean or sum of all data.

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
        ``self.degree - i``, available only if ``self.fit_lower='explicit'``.

    self.V_ : array, shape [n_orders, n_components, n_features]
        The learned basis functions.
        ``self.V_[0, :, :]`` is always available, and corresponds to
        interactions of order ``self.degree``.
        ``self.V_[i, :, :]`` for i > 0 corresponds to interactions of order
        ``self.degree - i``, available only if ``self.fit_lower='explicit'``.

    self.w_ : array, shape [n_features]
        The learned linear coefficients.
        It is learned only when self.fit_linear = True

    References
    ----------
    Inductive Pairwise Ranking: Going Beyond the $n log(n)$ Barrier.
    U.N. Niranjan and Arun Rajkumar.
    In AAAI 2017.
    https://arxiv.org/pdf/1702.02661.pdf

    """

    def __init__(self, degree=2, n_components=30, beta=1e-5, tol=1e-6,
                 fit_lower='explicit', fit_linear=True, warm_start=False,
                 max_iter=1000, verbose=False, random_state=None,
                 mean=True, var=0.01, callback=None, n_calls=10):
        super(FeatureLowRankInnerRegressor, self).__init__(
            degree, 'squared', n_components, beta, tol, fit_lower, fit_linear,
            warm_start, max_iter, verbose, random_state, mean, var, callback,
            n_calls)


class FeatureLowRankInnerClassifier(
    _BaseFeatureLowRankInner, _PairClassifierMixin
):
    """Feature low rank inner model for classification.

    Parameters
    ----------
    degree : int >= 2 (default=2)
        Degree of the polynomial. Corresponds to the order of feature
        interactions captured by the model.

    loss : {'logistic'|'squared_hinge'|'squared'|'soft_cross_entropy'|}, 
           (default='squared_hinge')
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

    tol : float (default=1e-6)
        Tolerance for the stopping condition.

    fit_lower : {'explicit'|'augment'|None} (default='explicit')
        Whether and how to fit lower-order, non-homogeneous terms.
        - 'explicit': fits a separate P directly for each lower order.
        - 'augment': adds the required number of dummy columns (columns
           that are 1 everywhere) in order to capture lower-order terms.
           Adds ``degree - 2`` columns if ``fit_linear`` is true, or
           ``degree - 1`` columns otherwise, to account for the linear term.
        - None: only learns weights for the degree given.  If ``degree == 3``,
          for example, the model will only have weights for third-order
          feature interactions.

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
        initialization of B_ and C_.

    mean : bool (default=True)
        Whether loss is mean or sum of all data.

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
        ``self.degree - i``, available only if ``self.fit_lower='explicit'``.

    self.V_ : array, shape [n_orders, n_components, n_features]
        The learned basis functions.
        ``self.V_[0, :, :]`` is always available, and corresponds to
        interactions of order ``self.degree``.
        ``self.V_[i, :, :]`` for i > 0 corresponds to interactions of order
        ``self.degree - i``, available only if ``self.fit_lower='explicit'``.

    self.w_ : array, shape [n_features]
        The learned linear coefficients.
        It is learned only when self.fit_linear = True

    References
    ----------
    Inductive Pairwise Ranking: Going Beyond the $n log(n)$ Barrier.
    U.N. Niranjan and Arun Rajkumar.
    In AAAI 2017.
    https://arxiv.org/pdf/1702.02661.pdf

    """

    def __init__(self, degree=2, loss='squared_hinge', n_components=30, 
                 alpha=1e-5, beta=1e-5, tol=1e-6, fit_lower='explicit', 
                 fit_linear=True, warm_start=False, max_iter=1000, 
                 verbose=False, random_state=None, mean=True, var=0.01,
                 callback=None, n_calls=10):
        super(FeatureLowRankInnerClassifier, self).__init__(
            degree, loss, n_components, alpha, beta, tol, fit_lower, fit_linear,
            warm_start, max_iter, verbose, random_state, mean, var, callback,
            n_calls)
