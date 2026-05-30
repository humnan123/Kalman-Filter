/* test_vector.c – Unit test for vectorised matrix-vector multiply */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Prototype matches lkf_vector.s matvec_NxM(out, mat, vec, rows, cols) */
extern void matvec_NxM(double *out, double *mat, double *vec,
                        int rows, int cols);

/* Reference scalar implementation for comparison */
static void ref_matvec(double *out, const double *mat, const double *vec,
                        int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        double s = 0.0;
        for (int j = 0; j < cols; j++)
            s += mat[i * cols + j] * vec[j];
        out[i] = s;
    }
}

int main(void)
{
    const int N = 276;

    size_t size_A   = (size_t)N * N * sizeof(double);
    size_t size_vec = (size_t)N     * sizeof(double);

    /* 64-byte aligned allocations */
    double *A    = aligned_alloc(64, (size_A   + 63) & ~(size_t)63);
    double *x    = aligned_alloc(64, (size_vec + 63) & ~(size_t)63);
    double *y    = aligned_alloc(64, (size_vec + 63) & ~(size_t)63);
    double *yref = aligned_alloc(64, (size_vec + 63) & ~(size_t)63);

    if (!A || !x || !y || !yref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Fill: A = all 1.0,  x = all 2.0  → y[i] should equal N*2 = 552 */
    for (int i = 0; i < N * N; i++) A[i] = 1.0;
    for (int i = 0; i < N;     i++) x[i] = 2.0;

    printf("Starting Vectorised Matrix-Vector Multiplication (N=%d)...\n", N);

    matvec_NxM(y,    A, x, N, N);
    ref_matvec(yref, A, x, N, N);

    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        double e = fabs(y[i] - yref[i]);
        if (e > max_err) max_err = e;
    }

    printf("Result y[0]   : %f  (Expected: %f)\n", y[0], (double)N * 2.0);
    printf("Result y[N-1] : %f  (Expected: %f)\n", y[N-1], (double)N * 2.0);
    printf("Max error vs reference: %.2e\n", max_err);
    printf("TEST %s\n", max_err < 1e-12 ? "PASSED" : "FAILED");

    free(A); free(x); free(y); free(yref);
    return max_err < 1e-12 ? 0 : 1;
}
