from cython.view cimport array


cdef void _canova(double[:] p,
                  double* x,
                  int degree,
                  int* indices,
                  int n_nz,
                  double[::1, :] a):
    cdef Py_ssize_t t, jj, j, j_prev

    a[0] = 1
    for t in range(1, degree+1):
        j_prev = t - 2
        for jj in range(t-1, n_nz):
            j = indices[jj]
            a[t, j+1] = a[t, j_prev+1] + p[j]*x[jj]*a[t-1, j_prev+1]
            j_prev = indices[jj]


cdef void _cgrad_anova(double[:] p,
                       double* x,
                       int degree,
                       int* indices,
                       int n_nz,
                       double[::1, :] a,
                       double[:, ::1] grad_table,
                       double[:] grad):

    cdef Py_ssize_t t, jj, j, j_prev
    cdef Py_ssize_t n_features = p.shape[0]

    # init gradient dp table
    for t in range(degree+1):
        for j in range(n_features):
            grad_table[t, j] = 0
    grad_table[degree-1, indices[n_nz-1]] = 1

    # compute dp table for gradient
    for t in range(degree-1, -1, -1):
        for jj in range(n_nz-2, t-1, -1):
            j_plus_1 = indices[jj+1]
            j = indices[jj]
            grad_table[t, j] = (grad_table[t, j_plus_1]
                                + grad_table[t+1, j_plus_1]*p[j_plus_1]*x[jj+1])

    # compute gradient
    for j in range(n_features):
        grad[j] = 0
    for jj in range(n_nz):
        j = indices[jj]
        for t in range(degree):
            grad[j] += grad_table[t, j] * a[t, j] * x[jj]


cdef void _canova_saving_memory(double[:] p,
                                double* x,
                                int degree,
                                int* indices,
                                int n_nz,
                                double[:] a):
    cdef Py_ssize_t t, j, jj

    a[0] = 1
    for t in range(1, degree+1):
        a[t] = 0
    for jj in range(degree):
        for t in range(jj+1):
            a[jj+1-t] += a[jj-t]*p[jj]*x[jj]

    for jj in range(degree, n_nz):
        j = indices[jj]
        for t in range(degree):
            a[degree-t] += a[degree-t-1]*p[j]*x[jj]


cdef void _canova_alt(int degree,
                      double[:] a,
                      double[:] poly):
    cdef unsigned int m, t
    cdef int sign
    a[0] = 1
    for m in range(1, degree+1):
        sign = 1
        a[m] = 0
        for t in range(1, m+1):
            a[m] += sign * a[m-t] * poly[t]
            sign *= -1
        a[m] /= m


cdef void _cgrad_anova_alt(double p_js,
                           double x_iij,
                           int degree,
                           double[:] a,
                           double[:] poly,
                           double[:] grad):
    cdef unsigned int m, t
    cdef int sign
    grad[:] = 0
    grad[1] = x_iij
    for m in range(2, degree+1):
        sign = 1
        for t in range(1, m+1):
            grad[m] += sign * (grad[m-t]*poly[t]
                               + a[m-t]*t*(p_js**(t-1))*(x_iij**t))
            sign *= -1
        grad[m] /= m


cdef void _cgrad_anova_coordinate_wise(double p_js,
                                       double x_iij,
                                       int degree,
                                       double[:] a,
                                       double[:] grad):
    cdef Py_ssize_t t
    grad[1] = x_iij
    for t in range(2, degree+1):
        grad[t] = x_iij * (a[t-1] - p_js * grad[t-1])


def canova(double[:] p,
           double[:] x,
           int degree,
           int[:] indices,
           int n_nz,
           double[::1, :] a):
    _canova(p, &x[0], degree, &indices[0], n_nz, a)


def cgrad_anova(double[:] p,
                double[:] x,
                int degree,
                int[:] indices,
                int n_nz,
                double[::1, :] a,
                double[:, ::1] grad_table,
                double[:] grad):
    _cgrad_anova(p, &x[0], degree, &indices[0], n_nz, a, grad_table, grad)


def canova_saving_memory(double[:] p,
                         double[:] x,
                         int degree,
                         int[:] indices,
                         int n_nz,
                         double[:] a):
    _canova_saving_memory(p, &x[0], degree, &indices[0], n_nz, a)


def canova_alt(int degree,
               double[:] a,
               double[:] poly):
    _canova_alt(degree, a, poly)


def cgrad_anova_alt(double p_js,
                    double x_iij,
                    int degree,
                    double[:] a,
                    double[:] poly,
                    double[:] grad):
    _cgrad_anova_alt(p_js, x_iij, degree, a, poly, grad)


def cgrad_anova_coordinate_wise(double p_js,
                                double x_iij,
                                int degree,
                                double[:] a,
                                double[:] grad):
    _cgrad_anova_coordinate_wise(p_js, x_iij, degree, a, grad)


def _canova_alt_predict(double[:, :, ::1] D, int degree):
    cdef Py_ssize_t i, s, t, n_samples, n_components
    n_samples = D.shape[1]
    n_components = D.shape[2]

    cdef double[:] A = array((degree+1, ), sizeof(double), 'd')
    cdef double[:, ::1] out = array((n_samples, degree+1), sizeof(double), 'd')

    for i in range(n_samples):
        for t in range(0, degree+1):
            out[i, t] = 0
        for s in range(n_components):
            _canova_alt(degree, A, D[:, i, s])
            for t in range(1, degree+1):
                out[i, t] += A[t]
    return out

def _cpair_predict(double[:, :, ::1] D1, double[:, :, ::1] D2, int degree):
    cdef Py_ssize_t i, s, t, n_samples, n_components
    n_samples = D1.shape[1]
    n_components = D1.shape[2]

    cdef double[:] A1 = array((degree, ), sizeof(double), 'd')
    cdef double[:] A2 = array((degree, ), sizeof(double), 'd')

    cdef double[:, ::1] out = array((n_samples, n_components),
                                    sizeof(double), 'd')

    for i in range(n_samples):
        for s in range(n_components):
            out[i, s] = 0
            _canova_alt(degree-1, A1, D1[:, i, s])
            _canova_alt(degree-1, A2, D2[:, i, s])
            for t in range(1, degree):
                out[i, s] += A1[t] * A2[degree-t]
    return out