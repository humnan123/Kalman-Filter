

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JOINTS          23
#define STATE_PER_JOINT 12
#define MEAS_PER_JOINT   3
#define TOTAL_STATE     276  
#define TOTAL_MEAS       69   
#define MAX_FRAMES      4096
#define MAX_LINE_LENGTH 10000
#define MAX_COLS        200
#define MAX_JOINT_NAME  50

extern void   mat_add_asm(double* C, const double* A, const double* B, int n);
extern void   mat_sub_asm(double* C, const double* A, const double* B, int n);
extern void   mat_scale_asm(double* A, double scalar, int n);
extern double mat_dot_asm(const double* A, const double* B, int n);
extern void   vec_axpy_asm(double* y, double alpha, const double* x, int n);
extern void   predict_x_asm(double* x12, double dt);


static inline double mget(const double* m, int cols, int i, int j) {
    return m[(long)i * cols + j];
}
static inline void mset(double* m, int cols, int i, int j, double v) {
    m[(long)i * cols + j] = v;
}


static void mat_mul_c(double* C, const double* A, const double* B,
    int rA, int cA, int cB) {
    memset(C, 0, (long)rA * cB * sizeof(double));
    for (int i = 0; i < rA; i++)
        for (int k = 0; k < cA; k++) {
            double a = mget(A, cA, i, k);
            if (a == 0.0) continue;
            for (int j = 0; j < cB; j++)
                C[(long)i * cB + j] += a * mget(B, cB, k, j);
        }
}

static void mat_transpose_c(double* T, const double* A, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mset(T, rows, j, i, mget(A, cols, i, j));
}

static void mat_identity_c(double* I, int n) {
    memset(I, 0, (long)n * n * sizeof(double));
    for (int i = 0; i < n; i++) mset(I, n, i, i, 1.0);
}


static int chol_decomp(double* L, const double* A, int n) {
    memset(L, 0, (long)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = 0;
            for (int k = 0; k < j; k++) s += mget(L, n, i, k) * mget(L, n, j, k);
            if (i == j) {
                double v = mget(A, n, i, i) - s;
                if (v <= 1e-12) return 0;
                mset(L, n, i, j, sqrt(v));
            }
            else {
                mset(L, n, i, j, (mget(A, n, i, j) - s) / mget(L, n, j, j));
            }
        }
    }
    return 1;
}

static void fwd_sub(const double* L, const double* b, double* x, int n) {
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int j = 0; j < i; j++) s -= mget(L, n, i, j) * x[j];
        x[i] = s / mget(L, n, i, i);
    }
}

static void bwd_sub(const double* L, const double* b, double* x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int j = i + 1; j < n; j++) s -= mget(L, n, j, i) * x[j];
        x[i] = s / mget(L, n, i, i);
    }
}

static void mat_inv_c(double* Inv, const double* A, int n) {
    double* L = calloc((long)n * n, sizeof(double));
    if (chol_decomp(L, A, n)) {
        double* e = calloc(n, sizeof(double));
        double* y = malloc(n * sizeof(double));
        double* col = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            memset(e, 0, n * sizeof(double)); e[j] = 1.0;
            fwd_sub(L, e, y, n);
            bwd_sub(L, y, col, n);
            for (int i = 0; i < n; i++) mset(Inv, n, i, j, col[i]);
        }
        free(e); free(y); free(col);
    }
    else {

        int n2 = 2 * n;
        double* aug = calloc((long)n * n2, sizeof(double));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) mset(aug, n2, i, j, mget(A, n, i, j));
            mset(aug, n2, i, n + i, 1.0);
        }
        for (int k = 0; k < n; k++) {
            int mr = k; double mv = fabs(mget(aug, n2, k, k));
            for (int i = k + 1; i < n; i++) if (fabs(mget(aug, n2, i, k)) > mv) { mv = fabs(mget(aug, n2, i, k)); mr = i; }
            if (mr != k) for (int j = 0; j < n2; j++) { double t = mget(aug, n2, k, j); mset(aug, n2, k, j, mget(aug, n2, mr, j)); mset(aug, n2, mr, j, t); }
            double piv = mget(aug, n2, k, k);
            if (fabs(piv) < 1e-14) piv = (piv >= 0) ? 1e-14 : -1e-14;
            for (int j = 0; j < n2; j++) mset(aug, n2, k, j, mget(aug, n2, k, j) / piv);
            for (int i = 0; i < n; i++) {
                if (i == k) continue;
                double fac = mget(aug, n2, i, k);
                for (int j = 0; j < n2; j++) mset(aug, n2, i, j, mget(aug, n2, i, j) - fac * mget(aug, n2, k, j));
            }
        }
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) mset(Inv, n, i, j, mget(aug, n2, i, n + j));
        free(aug);
    }
    free(L);
}

typedef struct { char name[MAX_JOINT_NAME]; int x_idx, y_idx, z_idx; } JointInfo;
typedef struct {
    int num_frames, num_joints, num_cols;
    char col_names[MAX_COLS][100];
    double noisy_data[MAX_COLS][MAX_FRAMES];
    double true_data[MAX_COLS][MAX_FRAMES];
    JointInfo joints[50];
} GaitDataset;

static GaitDataset ds;

static int count_cols(const char* l) { int c = 1; for (; *l; l++) if (*l == ',')c++; return c; }
static void parse_csv(const char* line, double* v, int nc) {
    char buf[MAX_LINE_LENGTH]; strncpy(buf, line, MAX_LINE_LENGTH - 1); buf[MAX_LINE_LENGTH - 1] = 0;
    char* t = strtok(buf, ","); int c = 0;
    while (t && c < nc) { v[c++] = atof(t); t = strtok(NULL, ","); }
}

static int load_noisy(const char* fn) {
    FILE* fp = fopen(fn, "r"); if (!fp) { fprintf(stderr, "Cannot open %s\n", fn); return -1; }
    char line[MAX_LINE_LENGTH];
    if (!fgets(line, sizeof line, fp)) { fclose(fp); return -1; }
    line[strcspn(line, "\n\r")] = 0;
    ds.num_cols = count_cols(line);
    char hdr[MAX_LINE_LENGTH]; strcpy(hdr, line);
    char* t = strtok(hdr, ","); int c = 0;
    while (t && c < ds.num_cols) { strncpy(ds.col_names[c], t, 99); c++; t = strtok(NULL, ","); }
    ds.num_frames = 0;
    while (fgets(line, sizeof line, fp)) ds.num_frames++;
    if (ds.num_frames > MAX_FRAMES) ds.num_frames = MAX_FRAMES;
    rewind(fp); fgets(line, sizeof line, fp);
    int fr = 0; double vals[MAX_COLS];
    while (fgets(line, sizeof line, fp) && fr < ds.num_frames) {
        line[strcspn(line, "\n\r")] = 0;
        parse_csv(line, vals, ds.num_cols);
        for (int cc = 0; cc < ds.num_cols; cc++) ds.noisy_data[cc][fr] = vals[cc];
        fr++;
    }
    ds.num_frames = fr; fclose(fp);
    ds.num_joints = 0;
    for (int i = 0; i + 2 < ds.num_cols; i += 3) {
        char* s = strstr(ds.col_names[i], "_x");
        if (s) {
            int l = (int)(s - ds.col_names[i]);
            strncpy(ds.joints[ds.num_joints].name, ds.col_names[i], l);
            ds.joints[ds.num_joints].name[l] = 0;
            ds.joints[ds.num_joints].x_idx = i;
            ds.joints[ds.num_joints].y_idx = i + 1;
            ds.joints[ds.num_joints].z_idx = i + 2;
            ds.num_joints++;
        }
    }
    printf("Noisy: %d frames, %d joints\n", ds.num_frames, ds.num_joints);
    return 0;
}

static int load_true(const char* fn) {
    FILE* fp = fopen(fn, "r"); if (!fp) return -1;
    char line[MAX_LINE_LENGTH]; fgets(line, sizeof line, fp);
    int fr = 0; double vals[MAX_COLS];
    while (fgets(line, sizeof line, fp) && fr < ds.num_frames) {
        line[strcspn(line, "\n\r")] = 0;
        parse_csv(line, vals, ds.num_cols);
        for (int cc = 0; cc < ds.num_cols; cc++) ds.true_data[cc][fr] = vals[cc];
        fr++;
    }
    fclose(fp);
    printf("True:  %d frames\n", fr);
    return 0;
}


static double* kf_x;   
static double* kf_P;   
static double* kf_F;  
static double* kf_Q;  
static double* kf_H;  
static double* kf_R;   

static void build_F(double dt) {
    memset(kf_F, 0, (long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    double dt2 = dt * dt, dt3 = dt2 * dt;
    for (int j = 0; j < JOINTS; j++) {
        int o = j * STATE_PER_JOINT;
        for (int b = 0; b < 3; b++) {
            int s = o + b * 4;
            mset(kf_F, TOTAL_STATE, s, s, 1.0);
            mset(kf_F, TOTAL_STATE, s, s + 1, dt);
            mset(kf_F, TOTAL_STATE, s, s + 2, dt2 / 2.0);
            mset(kf_F, TOTAL_STATE, s, s + 3, dt3 / 6.0);
            mset(kf_F, TOTAL_STATE, s + 1, s + 1, 1.0);
            mset(kf_F, TOTAL_STATE, s + 1, s + 2, dt);
            mset(kf_F, TOTAL_STATE, s + 1, s + 3, dt2 / 2.0);
            mset(kf_F, TOTAL_STATE, s + 2, s + 2, 1.0);
            mset(kf_F, TOTAL_STATE, s + 2, s + 3, dt);
            mset(kf_F, TOTAL_STATE, s + 3, s + 3, 1.0);
        }
    }
}

static void build_Q(double dt, double sj) {
    memset(kf_Q, 0, (long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    double s2 = sj * sj, dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt, dt6 = dt5 * dt;
    double q[4][4] = {
        {dt6 / 36, dt5 / 12, dt4 / 6,  dt3 / 6},
        {dt5 / 12, dt4 / 4,  dt3 / 2,  dt2 / 2},
        {dt4 / 6,  dt3 / 2,  dt2,    dt   },
        {dt3 / 6,  dt2 / 2,  dt,     1.0  }
    };
    for (int j = 0; j < JOINTS; j++) {
        int o = j * STATE_PER_JOINT;
        for (int b = 0; b < 3; b++) {
            int ob = o + b * 4;
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++)
                mset(kf_Q, TOTAL_STATE, ob + r, ob + c, s2 * q[r][c]);
        }
    }
}

static void build_H() {
    memset(kf_H, 0, (long)TOTAL_MEAS * TOTAL_STATE * sizeof(double));
    for (int j = 0; j < JOINTS; j++) {
        int sb = j * STATE_PER_JOINT, mb = j * MEAS_PER_JOINT;
        mset(kf_H, TOTAL_STATE, mb + 0, sb + 0, 1.0);
        mset(kf_H, TOTAL_STATE, mb + 1, sb + 4, 1.0);
        mset(kf_H, TOTAL_STATE, mb + 2, sb + 8, 1.0);
    }
}

static void build_R(double sr) {
    memset(kf_R, 0, (long)TOTAL_MEAS * TOTAL_MEAS * sizeof(double));
    double sr2 = sr * sr;
    for (int j = 0; j < JOINTS; j++) {
        int mb = j * MEAS_PER_JOINT;
        mset(kf_R, TOTAL_MEAS, mb, mb, sr2);
        mset(kf_R, TOTAL_MEAS, mb + 1, mb + 1, sr2);
        mset(kf_R, TOTAL_MEAS, mb + 2, mb + 2, sr2);
    }
}

static void init_filter(double dt, double sj, double sr) {
    kf_x = calloc(TOTAL_STATE, sizeof(double));
    kf_P = calloc((long)TOTAL_STATE * TOTAL_STATE, sizeof(double));
    kf_F = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    kf_Q = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    kf_H = malloc((long)TOTAL_MEAS * TOTAL_STATE * sizeof(double));
    kf_R = malloc((long)TOTAL_MEAS * TOTAL_MEAS * sizeof(double));

    build_F(dt); build_Q(dt, sj); build_H(); build_R(sr);


    for (int j = 0; j < JOINTS && j < ds.num_joints; j++) {
        int base = j * STATE_PER_JOINT;
        kf_x[base + 0] = ds.noisy_data[ds.joints[j].x_idx][0];
        kf_x[base + 4] = ds.noisy_data[ds.joints[j].y_idx][0];
        kf_x[base + 8] = ds.noisy_data[ds.joints[j].z_idx][0];
    }
    double vars[4] = { 10,100,1000,10000 };
    for (int j = 0; j < JOINTS; j++) {
        int b = j * STATE_PER_JOINT;
        for (int i = 0; i < STATE_PER_JOINT; i++)
            mset(kf_P, TOTAL_STATE, b + i, b + i, vars[i % 4]);
    }
}


static void kf_predict(double dt) {
    for (int j = 0; j < JOINTS; j++)
        predict_x_asm(kf_x + j * STATE_PER_JOINT, dt);

    double* Ft = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    double* FP = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    double* FPFt = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));

    mat_transpose_c(Ft, kf_F, TOTAL_STATE, TOTAL_STATE);
    mat_mul_c(FP, kf_F, kf_P, TOTAL_STATE, TOTAL_STATE, TOTAL_STATE);
    mat_mul_c(FPFt, FP, Ft, TOTAL_STATE, TOTAL_STATE, TOTAL_STATE);
    mat_add_asm(kf_P, FPFt, kf_Q, TOTAL_STATE * TOTAL_STATE);

    free(Ft); free(FP); free(FPFt);
}


static void kf_update(const double* z) {
    double* Ht = malloc((long)TOTAL_STATE * TOTAL_MEAS * sizeof(double));
    double* HP = malloc((long)TOTAL_MEAS * TOTAL_STATE * sizeof(double));
    double* HPHt = malloc((long)TOTAL_MEAS * TOTAL_MEAS * sizeof(double));
    double* S = malloc((long)TOTAL_MEAS * TOTAL_MEAS * sizeof(double));
    double* Si = malloc((long)TOTAL_MEAS * TOTAL_MEAS * sizeof(double));
    double* PHt = malloc((long)TOTAL_STATE * TOTAL_MEAS * sizeof(double));
    double* K = malloc((long)TOTAL_STATE * TOTAL_MEAS * sizeof(double));

    mat_transpose_c(Ht, kf_H, TOTAL_MEAS, TOTAL_STATE);
    mat_mul_c(HP, kf_H, kf_P, TOTAL_MEAS, TOTAL_STATE, TOTAL_STATE);
    mat_mul_c(HPHt, HP, Ht, TOTAL_MEAS, TOTAL_STATE, TOTAL_MEAS);
    mat_add_asm(S, HPHt, kf_R, TOTAL_MEAS * TOTAL_MEAS);

    mat_inv_c(Si, S, TOTAL_MEAS);
    mat_mul_c(PHt, kf_P, Ht, TOTAL_STATE, TOTAL_STATE, TOTAL_MEAS);
    mat_mul_c(K, PHt, Si, TOTAL_STATE, TOTAL_MEAS, TOTAL_MEAS);


    double* Hx = malloc(TOTAL_MEAS * sizeof(double));
    double* y = malloc(TOTAL_MEAS * sizeof(double));
    mat_mul_c(Hx, kf_H, kf_x, TOTAL_MEAS, TOTAL_STATE, 1);
    mat_sub_asm(y, z, Hx, TOTAL_MEAS);


    double* Ky = malloc(TOTAL_STATE * sizeof(double));
    mat_mul_c(Ky, K, y, TOTAL_STATE, TOTAL_MEAS, 1);
    mat_add_asm(kf_x, kf_x, Ky, TOTAL_STATE);
    free(Ky);

 
    double* KH = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    double* IKH = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    double* IKHt = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    double* IKHP = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    double* t1 = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));
    double* KR = malloc((long)TOTAL_STATE * TOTAL_MEAS * sizeof(double));
    double* Kt = malloc((long)TOTAL_MEAS * TOTAL_STATE * sizeof(double));
    double* t2 = malloc((long)TOTAL_STATE * TOTAL_STATE * sizeof(double));

    mat_mul_c(KH, K, kf_H, TOTAL_STATE, TOTAL_MEAS, TOTAL_STATE);
    mat_identity_c(IKH, TOTAL_STATE);
    mat_sub_asm(IKH, IKH, KH, TOTAL_STATE * TOTAL_STATE);
    mat_transpose_c(IKHt, IKH, TOTAL_STATE, TOTAL_STATE);
    mat_mul_c(IKHP, IKH, kf_P, TOTAL_STATE, TOTAL_STATE, TOTAL_STATE);
    mat_mul_c(t1, IKHP, IKHt, TOTAL_STATE, TOTAL_STATE, TOTAL_STATE);

    mat_mul_c(KR, K, kf_R, TOTAL_STATE, TOTAL_MEAS, TOTAL_MEAS);
    mat_transpose_c(Kt, K, TOTAL_STATE, TOTAL_MEAS);
    mat_mul_c(t2, KR, Kt, TOTAL_STATE, TOTAL_MEAS, TOTAL_STATE);
    mat_add_asm(kf_P, t1, t2, TOTAL_STATE * TOTAL_STATE);

    free(Ht); free(HP); free(HPHt); free(S); free(Si);
    free(PHt); free(K); free(Hx); free(y);
    free(KH); free(IKH); free(IKHt); free(IKHP);
    free(t1); free(KR); free(Kt); free(t2);
}


static void save_derivatives(double** states, int nf, const char* fn) {
    FILE* fp = fopen(fn, "w"); if (!fp) { perror(fn); return; }
    const char* jn = ds.joints[0].name;
    fprintf(fp, "Frame,Time,"
        "%s_px,%s_vx,%s_ax,%s_jx,"
        "%s_py,%s_vy,%s_ay,%s_jy,"
        "%s_pz,%s_vz,%s_az,%s_jz\n",
        jn, jn, jn, jn, jn, jn, jn, jn, jn, jn, jn, jn);
    for (int i = 0; i < nf; i++) {
        fprintf(fp, "%d,%.4f", i, i * 0.01);
        for (int k = 0; k < 12; k++) fprintf(fp, ",%.8f", states[i][k]);
        fprintf(fp, "\n");
    }
    fclose(fp); printf("Saved: %s\n", fn);
}

static void save_states(double** states, int nf, const char* fn) {
    FILE* fp = fopen(fn, "w"); if (!fp) { perror(fn); return; }
    fprintf(fp, "Frame");
    for (int j = 0; j < JOINTS; j++) {
        const char* n = ds.joints[j].name;
        fprintf(fp, ",%s_px,%s_vx,%s_ax,%s_jx,%s_py,%s_vy,%s_ay,%s_jy,%s_pz,%s_vz,%s_az,%s_jz",
            n, n, n, n, n, n, n, n, n, n, n, n);
    }
    fprintf(fp, "\n");
    for (int i = 0; i < nf; i++) {
        fprintf(fp, "%d", i);
        for (int s = 0; s < TOTAL_STATE; s++) fprintf(fp, ",%.8f", states[i][s]);
        fprintf(fp, "\n");
    }
    fclose(fp); printf("Saved: %s\n", fn);
}

static void save_comparison(double** states, int nf, const char* fn) {
    FILE* fp = fopen(fn, "w"); if (!fp) { perror(fn); return; }
    const char* jn = ds.joints[0].name;
    int xc = ds.joints[0].x_idx, yc = ds.joints[0].y_idx, zc = ds.joints[0].z_idx;
    fprintf(fp, "Frame,Time,%s_True_x,%s_True_y,%s_True_z,"
        "%s_Noisy_x,%s_Noisy_y,%s_Noisy_z,"
        "%s_LKF_x,%s_LKF_y,%s_LKF_z\n",
        jn, jn, jn, jn, jn, jn, jn, jn, jn);
    for (int i = 0; i < nf; i++)
        fprintf(fp, "%d,%.4f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
            i, i * 0.01,
            ds.true_data[xc][i], ds.true_data[yc][i], ds.true_data[zc][i],
            ds.noisy_data[xc][i], ds.noisy_data[yc][i], ds.noisy_data[zc][i],
            states[i][0], states[i][4], states[i][8]);
    fclose(fp); printf("Saved: %s\n", fn);
}

static void print_metrics(double** states, int nf) {
    int xc = ds.joints[0].x_idx;
    double mse_n = 0, mse_f = 0, mae_n = 0, mae_f = 0, max_n = 0, max_f = 0;
    for (int i = 0; i < nf; i++) {
        double tx = ds.true_data[xc][i], nx = ds.noisy_data[xc][i], fx = states[i][0];
        double en = fabs(nx - tx), ef = fabs(fx - tx);
        mse_n += en * en; mse_f += ef * ef; mae_n += en; mae_f += ef;
        if (en > max_n)max_n = en; if (ef > max_f)max_f = ef;
    }
    mse_n /= nf; mse_f /= nf; mae_n /= nf; mae_f /= nf;
    printf("\n==== VALIDATION: %s (X) ====\n", ds.joints[0].name);
    printf("%-12s %-12s %-12s\n", "Metric", "Noisy", "LKF");
    printf("%-12s %-12.6f %-12.6f (%.1f%% better)\n", "RMSE",
        sqrt(mse_n), sqrt(mse_f), (1 - sqrt(mse_f) / sqrt(mse_n)) * 100);
    printf("%-12s %-12.6f %-12.6f\n", "MAE", mae_n, mae_f);
    printf("%-12s %-12.6f %-12.6f\n", "MaxErr", max_n, max_f);

    double total_n = 0, total_f = 0; long total_s = 0;
    for (int jj = 0; jj < JOINTS; jj++) {
        int xci = ds.joints[jj].x_idx, yci = ds.joints[jj].y_idx, zci = ds.joints[jj].z_idx;
        int sii = jj * STATE_PER_JOINT;
        for (int i = 0; i < nf; i++) {
            double tx = ds.true_data[xci][i], ty = ds.true_data[yci][i], tz = ds.true_data[zci][i];
            double nx = ds.noisy_data[xci][i], ny = ds.noisy_data[yci][i], nz = ds.noisy_data[zci][i];
            double fx = states[i][sii + 0], fy = states[i][sii + 4], fz = states[i][sii + 8];
            total_n += (nx - tx) * (nx - tx) + (ny - ty) * (ny - ty) + (nz - tz) * (nz - tz);
            total_f += (fx - tx) * (fx - tx) + (fy - ty) * (fy - ty) + (fz - tz) * (fz - tz);
            total_s += 3;
        }
    }
    printf("\nAll Joints 3D Position:\n");
    printf("  Noisy RMSE : %.6f m\n", sqrt(total_n / total_s));
    printf("  LKF   RMSE : %.6f m\n", sqrt(total_f / total_s));
    printf("  Improvement: %.2f%%\n", (1.0 - sqrt(total_f / total_n)) * 100.0);
}


int main(void) {
    printf("============================================================\n");
    printf("  LKF - RISC-V Assembly (Milestone 3)\n");
    printf("  Assembly: predict_x, update_x, mat_add, mat_sub, axpy\n");
    printf("============================================================\n\n");

    memset(&ds, 0, sizeof(ds));
    if (load_noisy("3DDataset(Noisy Values).csv") != 0) {
        fprintf(stderr, "Failed to load noisy data\n"); return 1;
    }
    load_true("3DDataset(True Values).csv");

    double dt = 0.01, sj = 1.0, sr = 0.5;
    printf("dt=%.3f  sigma_j=%.1f  sigma_r=%.1f\n", dt, sj, sr);

    init_filter(dt, sj, sr);

    int nf = ds.num_frames;
    double** filtered = malloc(nf * sizeof(double*));
    for (int i = 0; i < nf; i++) filtered[i] = malloc(TOTAL_STATE * sizeof(double));

    printf("Processing %d frames...\n", nf);
    for (int t = 0; t < nf; t++) {
        kf_predict(dt);
        double z[TOTAL_MEAS];
        for (int j = 0; j < JOINTS && j < ds.num_joints; j++) {
            z[j * 3 + 0] = ds.noisy_data[ds.joints[j].x_idx][t];
            z[j * 3 + 1] = ds.noisy_data[ds.joints[j].y_idx][t];
            z[j * 3 + 2] = ds.noisy_data[ds.joints[j].z_idx][t];
        }
        kf_update(z);
        memcpy(filtered[t], kf_x, TOTAL_STATE * sizeof(double));
        if ((t + 1) % 500 == 0) { printf("  %d/%d\r", t + 1, nf); fflush(stdout); }
    }
    printf("\nDone.\n");

    print_metrics(filtered, nf);

    system("mkdir -p LKF_Output");
    save_states(filtered, nf, "LKF_Output/LKF_Filtered_States.csv");
    save_comparison(filtered, nf, "LKF_Output/LKF_Comparison_FirstJoint.csv");
    save_derivatives(filtered, nf, "LKF_Output/LKF_Derivatives_FirstJoint.csv");

    for (int i = 0; i < nf; i++) free(filtered[i]);
    free(filtered);
    free(kf_x); free(kf_P); free(kf_F); free(kf_Q); free(kf_H); free(kf_R);
    printf("\nOutput in LKF_Output/\n");
    return 0;
}
