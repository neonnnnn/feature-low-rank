# Author: Kyohei Atarashi
# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


from cython.view cimport array
from lightning.impl.dataset_fast cimport ColumnDataset
from libc.math cimport fabs
from .loss_fast cimport LossFunction


cdef inline void _synchronize(int* indices,
                              double* data,
                              int n_nz,
                              double p_old,
                              double[:] y_pred,
                              unsigned int degree,
                              double update,
                              double[:, ::1] A_pa,
                              double[:, ::1] A_qb,
                              double[:] dA_a,
                              int sign): # coeff = 1/2 if symmetric
    cdef Py_ssize_t t, ii, i
    for ii in range(n_nz):
        i = indices[ii]
        dA_a[0] = data[ii]

        for t in range(1, degree-1):
            dA_a[t] = data[ii] * (A_pa[t, i] - p_old * dA_a[t-1])
            A_pa[t, i] -= update * dA_a[t-1]
            y_pred[i] -= sign * update * dA_a[t-1] * A_qb[degree-t, i]


        A_pa[degree-1, i] -= update * dA_a[degree-1-1]
        y_pred[i] -= sign * update * dA_a[degree-1-1] * A_qb[1, i]


cdef double _cd_linear_epoch(double[:] w,
                             ColumnDataset A,
                             ColumnDataset B,
                             double[:] y,
                             double[:] y_pred,
                             double[:] col_norm_sq,
                             double alpha,
                             LossFunction loss,
                             unsigned int denominator):
    cdef Py_ssize_t i, j, ii
    cdef double sum_viol = 0
    cdef Py_ssize_t n_features = w.shape[0]
    cdef double update
    cdef double inv_step_size

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for j in range(n_features):
        update = 0
        A.get_column_ptr(j, &indices, &data, &n_nz)
        # compute gradient with respect to w_j
        for ii in range(n_nz):
            i = indices[ii]
            update += loss.dloss(y_pred[i], y[i]) * data[ii]
        B.get_column_ptr(j, &indices, &data, &n_nz)
        for ii in range(n_nz):
            i = indices[ii]
            update -= loss.dloss(y_pred[i], y[i]) * data[ii]

        update /= denominator
        update += alpha * w[j]
        # compute second derivative upper bound
        inv_step_size = loss.mu * col_norm_sq[j] + alpha
        update /= inv_step_size

        w[j] -= update
        sum_viol += fabs(update)

        # update predictions
        A.get_column_ptr(j, &indices, &data, &n_nz)
        for ii in range(n_nz):
            i = indices[ii]
            y_pred[i] -= update * data[ii]

        B.get_column_ptr(j, &indices, &data, &n_nz)
        for ii in range(n_nz):
            i = indices[ii]
            y_pred[i] += update * data[ii]

    return sum_viol


cdef inline double _grad_pair(double[:] dA,
                              double[:, ::1] B,
                              Py_ssize_t degree,
                              Py_ssize_t i):
    cdef Py_ssize_t t
    cdef double ret = 0
    for t in range(1, degree):
        ret += B[t, i] * dA[degree-t-1]
    return ret


cdef inline void _grad_anova(double[:] dA,
                             double[:, ::1] A,
                             double x_ij,
                             double p_js,
                             unsigned int degree,
                             Py_ssize_t i):
    cdef Py_ssize_t t

    dA[0] = x_ij
    for t in range(1, degree):
        dA[t] = x_ij * (A[t, i] - p_js * dA[t-1])


cdef inline void _compute_pair_kernel(double[:, ::1] A_pa,
                                      double[:, ::1] A_qb,
                                      double[:] pair,
                                      unsigned int degree):
    cdef Py_ssize_t i, t
    cdef Py_ssize_t n_samples = A_pa.shape[1]
    for i in range(n_samples):
        for t in range(1, degree):
            pair[i] += A_pa[t, i] * A_qb[degree-t, i]


cdef void _precompute_A_all_degree(ColumnDataset X,
                                   double[:] p,
                                   double[:, ::1] A,
                                   unsigned int degree):
    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef Py_ssize_t i, j, ii, t
    cdef unsigned int deg

    for t in range(1, degree+1):
        for i in range(n_samples):
            A[t, i] = 0

    # calc {1, \ldots, degree}-order anova kernels for all data
    # A[m, i] = m-order anova kernel for i-th data
    for j in range(n_features):
        X.get_column_ptr(j, &indices, &data, &n_nz)
        for t in range(degree):
            for ii in range(n_nz):
                i = indices[ii]
                A[degree-t, i] += A[degree-t-1, i] * p[j] * data[ii]


cdef inline double _update_sym(int* indices1,
                               double* data1,
                               int n_nz1,
                               int* indices2,
                               double* data2,
                               int n_nz2,
                               double p_js,
                               double[:] y,
                               double[:] y_pred,
                               LossFunction loss,
                               unsigned int degree,
                               double beta,
                               double[:, ::1] A_ua,
                               double[:, ::1] A_vb,
                               double[:, ::1] A_ub,
                               double[:, ::1] A_va,
                               double[:] dA_a,
                               double[:] dA_b,
                               unsigned int denominator):
    cdef double l1_reg = 2 * beta
    cdef Py_ssize_t i, ii1, ii2
    cdef double inv_step_size = 0
    cdef double update = 0
    cdef double grad = 0

    ii1 = 0
    ii2 = 0
    # when _update_sym is called, n_nz1 or n_nz2 is always non-zero.
    while ii1 < n_nz1 or ii2 < n_nz2:
        if (ii1 != n_nz1) and (ii2 == n_nz2 or indices1[ii1] < indices2[ii2]):
            i = indices1[ii1]
            _grad_anova(dA_a, A_ua, data1[ii1], p_js, degree-1, i)
            grad = _grad_pair(dA_a, A_vb, degree, i)
            ii1 += 1
        elif (ii2 != n_nz2) and (ii1 == n_nz1 or indices2[ii2] < indices1[ii1]):
            i = indices2[ii2]
            _grad_anova(dA_b, A_ub, data2[ii2], p_js, degree-1, i)
            grad = -_grad_pair(dA_b, A_va, degree, i)
            ii2 += 1
        else:
            i = indices1[ii1] # = indices2[ii2]
            _grad_anova(dA_a, A_ua, data1[ii1], p_js, degree-1, i)
            _grad_anova(dA_b, A_ub, data2[ii2], p_js, degree-1, i)
            grad = _grad_pair(dA_a, A_vb, degree, i)
            grad -= _grad_pair(dA_b, A_va, degree, i)
            ii1 += 1
            ii2 += 1
        update += grad * loss.dloss(y_pred[i], y[i])
        inv_step_size += grad**2

    inv_step_size *= loss.mu / denominator
    inv_step_size += l1_reg

    update /= denominator
    update += l1_reg * p_js
    update /= inv_step_size

    return update


cdef inline double _cd_epoch(double[:, ::1] U,
                             double[:, ::1] V,
                             ColumnDataset A,
                             ColumnDataset B,
                             double[:] y,
                             double[:] y_pred,
                             unsigned int degree,
                             double beta,
                             LossFunction loss,
                             double[:, ::1] A_ua,
                             double[:, ::1] A_vb,
                             double[:, ::1] A_ub,
                             double[:, ::1] A_va,
                             double[:] dA_a,
                             double[:] dA_b,
                             unsigned int denominator):

    cdef Py_ssize_t s, j, ii, i
    cdef double old, update
    cdef double sum_viol = 0
    cdef Py_ssize_t n_components = U.shape[0]
    cdef Py_ssize_t n_features = U.shape[1]
    cdef Py_ssize_t n_samples = A.get_n_samples()
    # Data pointers
    cdef double* data1
    cdef double* data2
    cdef int* indices1
    cdef int* indices2
    cdef int n_nz1, n_nz2

    # update U_{s} \forall s \in [n_components] for A^{degree}
    # U_{s} for A^{degree} = U[s]
    for s in range(n_components):
        # initialize anova kernels
        _precompute_A_all_degree(A, U[s, :], A_ua, degree-1)
        _precompute_A_all_degree(B, V[s, :], A_vb, degree-1)
        _precompute_A_all_degree(B, U[s, :], A_ub, degree-1)
        _precompute_A_all_degree(A, V[s, :], A_va, degree-1)

        for j in range(n_features):
            A.get_column_ptr(j, &indices1, &data1, &n_nz1)
            B.get_column_ptr(j, &indices2, &data2, &n_nz2)
            if n_nz1 == 0 and n_nz2 == 0:
                continue

            # update U_{s, j}
            old = U[s, j]
            # compute coordinate update
            update = _update_sym(indices1, data1, n_nz1, indices2, data2,
                                 n_nz2, old, y, y_pred, loss, degree,
                                 beta, A_ua, A_vb, A_ub, A_va, dA_a, dA_b,
                                 denominator)
            sum_viol += fabs(update)
            U[s, j] -= update
            # Synchronize predictions, A_ua and A_ub
            if n_nz1 != 0:
                _synchronize(indices1, data1, n_nz1, old, y_pred,
                             degree, update, A_ua, A_vb, dA_a, 1)
            if n_nz2 != 0:
                _synchronize(indices2, data2, n_nz2, old, y_pred,
                             degree, update, A_ub, A_va, dA_b, -1)

            # update V_{s, j}
            old = V[s, j]
            # compute coordinate update
            update = _update_sym(indices2, data2, n_nz2, indices1, data1,
                                 n_nz1, old, y, y_pred, loss, degree,
                                 beta, A_vb, A_ua, A_va, A_ub, dA_b, dA_a,
                                 denominator)
            sum_viol += fabs(update)
            V[s, j] -= update
            # Synchronize predictions, A_vb and A_va
            if n_nz1 != 0:
                _synchronize(indices1, data1, n_nz1, old, y_pred,
                             degree, update, A_va, A_ub, dA_a, -1)
            if n_nz2 != 0:
                _synchronize(indices2, data2, n_nz2, old, y_pred,
                             degree, update, A_vb, A_ua, dA_b, 1)

    return sum_viol


def _cd(self,
        double[:, :, ::1] U not None,
        double[:, :, ::1] V not None,
        double[:] w,
        ColumnDataset A,
        ColumnDataset B,
        double[:] y not None,
        double[:] y_pred not None,
        double[:] col_norm_sq,
        unsigned int degree,
        double alpha,
        double beta,
        bint fit_lower,
        bint fit_linear,
        bint mean,
        LossFunction loss,
        Py_ssize_t max_iter,
        double tol,
        int verbose,
        callback,
        int n_calls):
    cdef Py_ssize_t n_samples, it, i, t
    cdef double viol
    cdef bint converged = False
    cdef bint has_callback = callback is not None
    cdef unsigned int denominator = 1
    cdef double[:, ::1] A_ua, A_vb, A_ub, A_va
    cdef double[:] dA_a = array((degree-1, ), sizeof(double), 'd')
    cdef double[:] dA_b = array((degree-1, ), sizeof(double), 'd')

    n_samples = A.get_n_samples()
    if mean:
        denominator = n_samples
    # precomputed values
    # A[m, i] = A^{m}(p, X_i)
    A_ua = array((degree, n_samples),  sizeof(double), 'd')
    A_vb = array((degree, n_samples),  sizeof(double), 'd')
    A_ub = array((degree, n_samples),  sizeof(double), 'd')
    A_va = array((degree, n_samples),  sizeof(double), 'd')

    # init anova kernels
    for i in range(n_samples):
        A_ua[0, i] = 1
        A_vb[0, i] = 1
    for i in range(n_samples):
        A_ub[0, i] = 1
        A_va[0, i] = 1
    # start epoch
    it = 0
    for it in range(max_iter):
        viol = 0

        viol += _cd_epoch(U[0], V[0], A, B, y, y_pred, degree, beta,
                          loss, A_ua, A_vb, A_ub, A_va, dA_a, dA_b,
                          denominator)
        if fit_lower:
            for t in range(2, degree):
                viol += _cd_epoch(U[degree-t], V[degree-t], A, B, y, y_pred,
                                  t, beta, loss, A_ua, A_vb, A_ub, A_va, dA_a,
                                  dA_b, denominator)

        if fit_linear:
            viol += _cd_linear_epoch(w, A, B, y, y_pred, col_norm_sq, alpha,
                                     loss, denominator)

        if has_callback and it % n_calls == 0:
            ret = callback(self)
            if ret is not None:
                break

        if verbose:
            print("Iteration {} violation sum {}".format(it + 1, viol))

        if viol < tol:
            if verbose:
                print("Converged at iteration {}".format(it + 1))
            converged = True
            break

    return converged, it
