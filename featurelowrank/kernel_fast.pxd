cdef void _canova(double[:] p,
                  double* x,
                  int degree,
                  int* indices,
                  int n_nz,
                  double[::1, :] a)


cdef void _cgrad_anova(double[:] p,
                       double* x,
                       int degree,
                       int* indices,
                       int n_nz,
                       double[::1, :] a,
                       double[:, ::1] grad_table,
                       double[:] grad)


cdef void _canova_saving_memory(double[:] p,
                                double* x,
                                int degree,
                                int* indices,
                                int n_nz,
                                double[:] a)


cdef void _canova_alt(int degree,
                      double[:] a,
                      double[:] poly)


cdef void _cgrad_anova_alt(double p_js,
                           double x_iij,
                           int degree,
                           double[:] a,
                           double[:] poly,
                           double[:] grad)


cdef void _cgrad_anova_coordinate_wise(double p_js,
                                       double x_iij,
                                       int degree,
                                       double[:] a,
                                       double[:] grad)