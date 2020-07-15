# Author: Kyohei Atarashi
# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


from cython.view cimport array
from lightning.impl.dataset_fast cimport RowDataset
from libc.math cimport fabs, sqrt
from .loss_fast cimport LossFunction


cdef inline void _lazy_update_all(double* param,
                                  double* acc_grad,
                                  double* acc_grad_norm,
                                  int n_features,
                                  double eta0,
                                  double eta_t,
                                  int n_iter,
                                  double lam,
                                  double eps):
    cdef Py_ssize_t j
    for j in range(n_features):
        param[j] = - eta0 * acc_grad[j]
        param[j] /= (eps + sqrt(acc_grad_norm[j]) + eta_t*lam)


cdef inline double _lazy_update(double* param,
                                double* acc_grad,
                                double* acc_grad_norm,
                                int* indices,
                                int n_nz,
                                double eta0,
                                double eta_t,
                                int n_iter,
                                double lam,
                                double eps):
    cdef Py_ssize_t j, jj
    cdef double old, viol
    viol = 0
    for jj in range(n_nz):
        j = indices[jj]
        old = param[j]
        param[j] = - eta0 * acc_grad[j]
        param[j] /= (eps + sqrt(acc_grad_norm[j]) + eta_t*lam)
        viol += fabs(param[j] - old)
    return viol


cdef inline void _compute_grads(double[:, ::1] U,
                                double[:, ::1] V,
                                int* indices,
                                double* data,
                                int n_nz,
                                double Vx_s,
                                double Ux_s,
                                double row_sum_x,
                                double[:] grad_U,
                                double[:] grad_V,
                                Py_ssize_t s):
    cdef Py_ssize_t j
    for jj in range(n_nz):
        j = indices[jj]
        grad_U[j] -= row_sum_x * data[jj] * U[s, j]
        grad_U[j] += data[jj] * Vx_s
        grad_V[j] += row_sum_x * data[jj] * V[s, j]
        grad_V[j] -= data[jj] * Ux_s


cdef inline void _compute_acc_grads(double[:, ::1] U,
                                    double[:, ::1] V,
                                    int* indices,
                                    int n_nz,
                                    double dloss,
                                    double[:] grad_U,
                                    double[:, ::1] acc_grad_U,
                                    double[:, ::1] acc_grad_norm_U,
                                    double[:] grad_V,
                                    double[:, ::1] acc_grad_V,
                                    double[:, ::1] acc_grad_norm_V,
                                    Py_ssize_t s):
    cdef Py_ssize_t j
    for jj in range(n_nz):
        j = indices[jj]
        acc_grad_U[s, j] += dloss * grad_U[j]
        acc_grad_norm_U[s, j] += (dloss * grad_U[j])**2
        grad_U[j] = 0

        acc_grad_V[s, j] += dloss * grad_V[j]
        acc_grad_norm_V[s, j] += (dloss * grad_V[j])**2
        grad_V[j] = 0


cdef inline double _adagrad_epoch(double[:, ::1] U,
                                  double[:, ::1] V,
                                  double[:] w,
                                  RowDataset A,
                                  RowDataset B,
                                  RowDataset Linear,
                                  double[:] row_sum_A,
                                  double[:] row_sum_B,
                                  double[:] y,
                                  double[:, ::1] acc_grad_U,
                                  double[:, ::1] acc_grad_norm_U,
                                  double[:, ::1] acc_grad_V,
                                  double[:, ::1] acc_grad_norm_V,
                                  double[:] acc_grad_w,
                                  double[:] acc_grad_norm_w,
                                  double eta0,
                                  double alpha,
                                  double beta,
                                  double eps,
                                  bint fit_linear,
                                  LossFunction loss,
                                  int* n_iter,
                                  double[:] Ua,
                                  double[:] Vb,
                                  double[:] Ub,
                                  double[:] Va,
                                  double[:] grad_U,
                                  double[:] grad_V):
    cdef Py_ssize_t s, j, jj, j1, jj1, j2, jj2, i, ii
    cdef double y_pred, eta_t, grad_w_j
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
    for i in range(n_samples):
        eta_t = eta0 * n_iter[0]
        y_pred = 0

        if fit_linear and (n_iter[0] > 1):
            Linear.get_row_ptr(i, &indices1, &data1, &n_nz1)
            sum_viol += _lazy_update(&w[0], &acc_grad_w[0], &acc_grad_norm_w[0],
                                     indices1, n_nz1, eta0, eta_t/eta0,
                                     n_iter[0], alpha, eps)

        A.get_row_ptr(i, &indices1, &data1, &n_nz1)
        B.get_row_ptr(i, &indices2, &data2, &n_nz2)
        # compute prediction
        for s in range(n_components):
            # lazy update
            if n_iter[0] > 1:
                sum_viol += _lazy_update(&U[s, 0], &acc_grad_U[s, 0],
                                         &acc_grad_norm_U[s, 0], indices1,
                                         n_nz1, eta0, eta_t, n_iter[0], beta, eps)
                sum_viol += _lazy_update(&V[s, 0], &acc_grad_V[s, 0],
                                         &acc_grad_norm_V[s, 0], indices2,
                                         n_nz2, eta0, eta_t, n_iter[0], beta, eps)
            Ua[s] = 0
            Ub[s] = 0
            Va[s] = 0
            Vb[s] = 0
            for jj in range(n_nz1):
                j = indices1[jj]
                Ua[s] += data1[jj] * U[s, j]
                Va[s] += data1[jj] * V[s, j]
                y_pred -= 0.5 * row_sum_B[i] * data1[jj] * U[s, j]**2
                y_pred += 0.5 * row_sum_B[i] * data1[jj] * V[s, j]**2

            for jj in range(n_nz2):
                j = indices2[jj]
                Ub[s] += data2[jj] * U[s, j]
                Vb[s] += data2[jj] * V[s, j]
                y_pred -= 0.5 * row_sum_A[i] * data2[jj] * V[s, j]**2
                y_pred += 0.5 * row_sum_A[i] * data2[jj] * U[s, j]**2
            y_pred += Ua[s] * Vb[s] - Ub[s] * Va[s]

        if fit_linear:
            for jj in range(n_nz1):
                j = indices1[jj]
                y_pred += w[j] * data1[jj]
            for jj in range(n_nz2):
                j = indices2[jj]
                y_pred -= w[j] * data2[j]
    
        dloss = loss.dloss(y_pred, y[i])
        if dloss != 0:
            for s in range(n_components):
                _compute_grads(U, V, indices1, data1, n_nz1, Vb[s], Ub[s],
                               row_sum_B[i], grad_U, grad_V, s)

                _compute_grads(V, U, indices2, data2, n_nz2, Ua[s], Va[s],
                               row_sum_A[i], grad_V, grad_U, s)

                _compute_acc_grads(U, V, indices1, n_nz1, dloss,
                                   grad_U, acc_grad_U, acc_grad_norm_U,
                                   grad_V, acc_grad_V, acc_grad_norm_V, s)

                _compute_acc_grads(U, V, indices2, n_nz2, dloss,
                                   grad_U, acc_grad_U, acc_grad_norm_U,
                                   grad_V, acc_grad_V, acc_grad_norm_V, s)

            if fit_linear:
                Linear.get_row_ptr(i, &indices1, &data1, &n_nz1)
                for jj in range(n_nz1):
                    j = indices1[jj]
                    grad_w_j = dloss * data1[jj]
                    acc_grad_w[j] += grad_w_j
                    acc_grad_norm_w[j] += grad_w_j**2
        n_iter[0] += 1
    return sum_viol


def _adagrad(self,
             double[:, :, ::1] U not None,
             double[:, :, ::1] V not None,
             double[:] w,
             RowDataset A,
             RowDataset B,
             RowDataset Linear,
             double[:] row_sum_A,
             double[:] row_sum_B,
             double[:] y not None,
             double[:, :, ::1] acc_grad_U,
             double[:, :, ::1] acc_grad_norm_U,
             double[:, :, ::1] acc_grad_V,
             double[:, :, ::1] acc_grad_norm_V,
             double[:] acc_grad_w,
             double[:] acc_grad_norm_w,
             double eta0,
             double alpha,
             double beta,
             double eps,
             bint fit_linear,
             LossFunction loss,
             Py_ssize_t max_iter,
             int n_iter,
             double tol,
             int verbose,
             callback,
             int n_calls):
    cdef Py_ssize_t n_samples, n_features, n_components, it, s, j
    cdef double viol
    cdef bint converged = False
    cdef bint has_callback = callback is not None
    cdef double[:] Ua, Vb, Ub, Va, grad_U, grad_V, grad_w
    n_samples = A.get_n_samples()
    n_features = A.get_n_features()
    n_components = U.shape[1]

    # precomputed values
    Ua = array((n_components, ),  sizeof(double), 'd')
    Vb = array((n_components, ),  sizeof(double), 'd')
    Ub = array((n_components, ),  sizeof(double), 'd')
    Va = array((n_components, ),  sizeof(double), 'd')

    # cache for gradient
    grad_U = array((n_features, ), sizeof(double), 'd')
    grad_V = array((n_features, ), sizeof(double), 'd')
    for j in range(n_features):
        grad_U[j] = 0
        grad_V[j] = 0

    # start epoch
    for it in range(max_iter):
        viol = _adagrad_epoch(U[0], V[0], w, A, B, Linear, row_sum_A, row_sum_B, y,
                              acc_grad_U[0], acc_grad_norm_U[0], acc_grad_V[0],
                              acc_grad_norm_V[0], acc_grad_w, acc_grad_norm_w,
                              eta0, alpha, beta, eps, fit_linear, loss, &n_iter,
                              Ua, Vb, Ub, Va, grad_U, grad_V)
        if has_callback and it % n_calls == 0:
            ret = callback(self)
            if ret is not None:
                break
        if verbose:
            print("Epoch {} violation sum {}".format(it + 1, viol))

        if viol < tol:
            if verbose:
                print("Converged at iteration {}".format(it + 1))
            converged = True
            break

    # update all parameters lazily
    for s in range(n_components):
        _lazy_update_all(&U[0, s, 0], &acc_grad_U[0, s, 0],
                         &acc_grad_norm_U[0, s, 0], n_features, eta0,
                         eta0*n_iter, n_iter, beta, eps)
        _lazy_update_all(&V[0, s, 0], &acc_grad_V[0, s, 0],
                         &acc_grad_norm_V[0, s, 0], n_features, eta0,
                         eta0*n_iter, n_iter, beta, eps)
    if fit_linear:
        _lazy_update_all(&w[0], &acc_grad_w[0],
                         &acc_grad_norm_w[0], n_features, eta0,
                         eta0*n_iter, n_iter, beta, eps)

    return converged, n_iter
