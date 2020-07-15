# Author: Kyohei Atarashi
# License: Simplified BSD

from abc import ABCMeta
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import type_of_target
from .loss import CLASSIFICATION_LOSSES, REGRESSION_LOSSES

EPS = 1e-10

class _BasePair(BaseEstimator, metaclass=ABCMeta):

    def _get_loss(self, loss):
        # classification losses
        if loss not in self._LOSSES:
            raise ValueError(
                'Loss function "{}" not supported. The available options '
                'are: "{}".'.format(loss, '", "'.join(self._LOSSES)))
        return self._LOSSES[loss]


class _PairRegressorMixin(RegressorMixin):

    _LOSSES = REGRESSION_LOSSES

    def _check_A_B_y(self, A, B, y):
        A, y = check_X_y(A, y, accept_sparse='csc', multi_output=False,
                         dtype=np.double, y_numeric=True)
        B, _ = check_X_y(B, y, accept_sparse='csc', multi_output=False,
                         dtype=np.double, y_numeric=True)
        y = y.astype(np.double).ravel()
        return A, B, y

    def predict(self, X):
        """Predict regression output for the samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, 2*n_features]
            Samples.

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Returns predicted values.
        """
        n_features = int(X.shape[1] / 2)
        A = X[:, :n_features]
        B = X[:, n_features:]
        return self._predict(A, B)


class _PairClassifierMixin(ClassifierMixin):

    _LOSSES = CLASSIFICATION_LOSSES

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy or log-likelihood on the given test data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, 2*n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        if self.loss != 'soft_cross_entropy':
            return super(_PairClassifierMixin, self).score(X, y, sample_weight)
        else:
            y_pred = self.predict_proba(X)
            log_likelihood = y*np.log(y_pred+EPS) + (1-y)*np.log(1-y_pred+EPS)
            if sample_weight is not None:
                return np.dot(log_likelihood, sample_weight)
            else:
                return np.mean(log_likelihood)

    def decision_function(self, X):
        """Compute the output before thresholding.

        Parameters
        ----------
        A : {array-like, sparse matrix}, shape = [n_samples, 2*n_features]
            Samples.
    
        Returns
        -------
        y_scores : array, shape = [n_samples]
            Returns predicted values.
        """
        n_features = int(X.shape[1] / 2)
        A = X[:, :n_features]
        B = X[:, n_features:]
 
        return self._predict(A, B)

    def predict(self, X):
        """Prediction of the given data.
        If loss='soft_cross_entropy', returns {0, 0.5, 1}.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, 2*n_features]
            Samples.

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Returns predicted values.
        """
        if self.loss != 'soft_cross_entropy':
            y_pred = self.decision_function(X) > 0
            return self.label_binarizer_.inverse_transform(y_pred)
        else:
            y_pred = self.decision_function(X)
            y_pred[y_pred>0] = 1
            y_pred[y_pred==0] = 0.5 
            y_pred[y_pred<0] = 0
            return y_pred

    def predict_proba(self, X):
        """Compute probability estimates for the test samples.
        Only available if `loss='logistic'` or `loss='soft_cross_entropy'`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, 2*n_features]
            Samples.

        Returns
        -------
        y_scores : array, shape = [n_samples]
            Probability estimates that the samples are from the positive class.
        """
 
        if self.loss in ['logistic', 'soft_cross_entropy'] :
            return 1 / (1 + np.exp(-self.decision_function(X)))
        else:
            raise ValueError("Probability estimates only available for "
                             "loss='logistic' or 'soft_cross_entropy'. "
                             "You may use probability "
                             "calibration methods from scikit-learn instead.")

    def _check_A_B_y(self, A, B, y):
        # helpful error message for sklearn < 1.17
        is_2d = hasattr(y, 'shape') and len(y.shape) > 1 and y.shape[1] >= 2

        if self.loss != "soft_cross_entropy":
            if is_2d or type_of_target(y) != 'binary':
                raise TypeError("Only binary targets supported. For training "
                                "multiclass or multilabel models, you may use the "
                                "OneVsRest or OneVsAll metaestimators in "
                                "scikit-learn.")
        else:
            if is_2d or np.min(y) < 0 or np.max(y) > 1:
                raise ValueError("If loss='soft_cross_entropy', each element "
                                 "in y must be in [0, 1].")

        A, Y = check_X_y(A, y, dtype=np.double, accept_sparse='csc',
                         multi_output=False)
        B, _ = check_X_y(B, y, dtype=np.double, accept_sparse='csc',
                         multi_output=False)
        if self.loss != "soft_cross_entropy":
            self.label_binarizer_ = LabelBinarizer(pos_label=1, neg_label=-1)
            y = self.label_binarizer_.fit_transform(Y).ravel().astype(np.double)
            return A, B, y
        else:
            return A, B, Y.ravel()