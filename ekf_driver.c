

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define JOINTS       23
#define SPJ          12   
#define MPJ           3 
#define N           276   
#define M            69   
#define MAX_FRAMES 4096
#define MAX_LINE   10000
#define MAX_COLS    200
#define MAX_JN       50

extern double atan_poly_asm(double x);
extern double atan2_asm(double y, double x);
extern void   h_func_asm(double px, double py, double pz, double* z_sph);
extern double wrap_angle_asm(double v);
extern void   build_jblock_asm(double* Hj36, double px, double py, double pz);
extern void   predict_x_asm(double* x12, double dt);
extern void   inv3x3_asm(double* Ainv9, const double* A9, int* ok);
extern void   mat_add_asm(double* C, const double* A, const double* B, int n);
extern void   mat_sub_asm(double* C, const double* A, const double* B, int n);


static inline double mg(const double* m, int c, int i, int j) { return m[(long)i * c + j]; }
static inline void   ms(double* m, int c, int i, int j, double v) { m[(long)i * c + j] = v; }


typedef struct { char name[MAX_JN]; int x, y, z; } JI;
typedef struct {
    int nf, nj, nc;
    char cn[MAX_COLS][100];
    double nd[MAX_COLS][MAX_FRAMES];
    double td[MAX_COLS][MAX_FRAMES];
    JI j[50];
} DS;
static DS ds;

static int ccols(const char* l) { int c = 1; for (; *l; l++) if (*l == ',') c++; return c; }
static void pcsv(const char* line, double* v, int nc) {
    char b[MAX_LINE]; strncpy(b, line, MAX_LINE - 1); b[MAX_LINE - 1] = 0;
    char* t = strtok(b, ","); int c = 0;
    while (t && c < nc) { v[c++] = atof(t); t = strtok(NULL, ","); }
}

static int load_nd(const char* fn) {
    FILE* fp = fopen(fn, "r"); if (!fp) { fprintf(stderr, "Cannot open %s\n", fn); return -1; }
    char line[MAX_LINE];
    if (!fgets(line, sizeof line, fp)) { fclose(fp); return -1; }
    line[strcspn(line, "\n\r")] = 0;
    ds.nc = ccols(line);
    char h[MAX_LINE]; strcpy(h, line);
    char* t = strtok(h, ","); int c = 0;
    while (t && c < ds.nc) { strncpy(ds.cn[c], t, 99); c++; t = strtok(NULL, ","); }
    ds.nf = 0; while (fgets(line, sizeof line, fp)) ds.nf++;
    if (ds.nf > MAX_FRAMES) ds.nf = MAX_FRAMES;
    rewind(fp); fgets(line, sizeof line, fp);
    int fr = 0; double v[MAX_COLS];
    while (fgets(line, sizeof line, fp) && fr < ds.nf) {
        line[strcspn(line, "\n\r")] = 0; pcsv(line, v, ds.nc);
        for (int cc = 0; cc < ds.nc; cc++) ds.nd[cc][fr] = v[cc]; fr++;
    }
    ds.nf = fr; fclose(fp);
    ds.nj = 0;
    for (int i = 0; i + 2 < ds.nc; i += 3) {
        char* s = strstr(ds.cn[i], "_x");
        if (s) {
            int l = (int)(s - ds.cn[i]); strncpy(ds.j[ds.nj].name, ds.cn[i], l);
            ds.j[ds.nj].name[l] = 0; ds.j[ds.nj].x = i;
            ds.j[ds.nj].y = i + 1; ds.j[ds.nj].z = i + 2; ds.nj++;
        }
    }
    printf("Noisy: %d frames, %d joints\n", ds.nf, ds.nj);
    return 0;
}

static int load_td(const char* fn) {
    FILE* fp = fopen(fn, "r"); if (!fp) return -1;
    char line[MAX_LINE]; fgets(line, sizeof line, fp);
    int fr = 0; double v[MAX_COLS];
    while (fgets(line, sizeof line, fp) && fr < ds.nf) {
        line[strcspn(line, "\n\r")] = 0; pcsv(line, v, ds.nc);
        for (int cc = 0; cc < ds.nc; cc++) ds.td[cc][fr] = v[cc]; fr++;
    }
    fclose(fp); printf("True:  %d frames\n", fr); return 0;
}


static void mmul(double* C, const double* A, const double* B, int rA, int cA, int cB) {
    memset(C, 0, (long)rA * cB * sizeof(double));
    for (int i = 0; i < rA; i++)
        for (int k = 0; k < cA; k++) {
            double a = mg(A, cA, i, k); if (a == 0.0) continue;
            for (int j = 0; j < cB; j++) C[(long)i * cB + j] += a * mg(B, cB, k, j);
        }
}

static void mT(double* T, const double* A, int r, int c) {
    for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) ms(T, r, j, i, mg(A, c, i, j));
}

static void mI(double* I, int n) {
    memset(I, 0, (long)n * n * sizeof(double));
    for (int i = 0; i < n; i++) ms(I, n, i, i, 1.0);
}


static void inv_chol_c(double* Inv, const double* A, int n) {
    double* L = calloc((long)n * n, sizeof(double));
    int ok = 1;
    for (int i = 0; i < n && ok; i++) {
        for (int j = 0; j <= i; j++) {
            double s = 0;
            for (int k = 0; k < j; k++) s += mg(L, n, i, k) * mg(L, n, j, k);
            if (i == j) { double v = mg(A, n, i, i) - s; if (v <= 1e-12) { ok = 0; break; } ms(L, n, i, j, sqrt(v)); }
            else ms(L, n, i, j, (mg(A, n, i, j) - s) / mg(L, n, j, j));
        }
    }
    if (ok) {
        double* e = calloc(n, sizeof(double)), * y = malloc(n * sizeof(double)), * col = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            memset(e, 0, n * sizeof(double)); e[j] = 1.0;
            for (int i = 0; i < n; i++) { double s = e[i]; for (int k = 0; k < i; k++) s -= mg(L, n, i, k) * y[k]; y[i] = s / mg(L, n, i, i); }
            for (int i = n - 1; i >= 0; i--) { double s = y[i]; for (int k = i + 1; k < n; k++) s -= mg(L, n, k, i) * col[k]; col[i] = s / mg(L, n, i, i); }
            for (int i = 0; i < n; i++) ms(Inv, n, i, j, col[i]);
        }
        free(e); free(y); free(col);
    }
    else {
        int n2 = 2 * n; double* aug = calloc((long)n * n2, sizeof(double));
        for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) ms(aug, n2, i, j, mg(A, n, i, j)); ms(aug, n2, i, n + i, 1.0); }
        for (int k = 0; k < n; k++) {
            int mr = k; double mv = fabs(mg(aug, n2, k, k));
            for (int i = k + 1; i < n; i++) if (fabs(mg(aug, n2, i, k)) > mv) { mv = fabs(mg(aug, n2, i, k)); mr = i; }
            if (mr != k) for (int j = 0; j < n2; j++) { double t = mg(aug, n2, k, j); ms(aug, n2, k, j, mg(aug, n2, mr, j)); ms(aug, n2, mr, j, t); }
            double piv = mg(aug, n2, k, k); if (fabs(piv) < 1e-14) piv = (piv >= 0) ? 1e-14 : -1e-14;
            for (int j = 0; j < n2; j++) ms(aug, n2, k, j, mg(aug, n2, k, j) / piv);
            for (int i = 0; i < n; i++) {
                if (i == k) continue; double f = mg(aug, n2, i, k);
                for (int j = 0; j < n2; j++) ms(aug, n2, i, j, mg(aug, n2, i, j) - f * mg(aug, n2, k, j));
            }
        }
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) ms(Inv, n, i, j, mg(aug, n2, i, n + j));
        free(aug);
    }
    free(L);
}

static double* ekf_x; 
static double* ekf_P;  
static double* ekf_F;  
static double* ekf_Q;  
static double* ekf_R; 

static void build_matrices(double dt, double sj, double sr) {
    double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt, dt6 = dt5 * dt;
    double s2 = sj * sj;


    memset(ekf_F, 0, (long)N * N * sizeof(double));
    for (int jj = 0; jj < JOINTS; jj++) {
        int o = jj * SPJ;
        for (int b = 0; b < 3; b++) {
            int s = o + b * 4;
            ms(ekf_F, N, s, s, 1.0); ms(ekf_F, N, s, s + 1, dt);
            ms(ekf_F, N, s, s + 2, dt2 / 2); ms(ekf_F, N, s, s + 3, dt3 / 6);
            ms(ekf_F, N, s + 1, s + 1, 1.0); ms(ekf_F, N, s + 1, s + 2, dt);
            ms(ekf_F, N, s + 1, s + 3, dt2 / 2); ms(ekf_F, N, s + 2, s + 2, 1.0);
            ms(ekf_F, N, s + 2, s + 3, dt); ms(ekf_F, N, s + 3, s + 3, 1.0);
        }
    }


    memset(ekf_Q, 0, (long)N * N * sizeof(double));
    double q[4][4] = {
        {dt6 / 36, dt5 / 12, dt4 / 6,  dt3 / 6},
        {dt5 / 12, dt4 / 4,  dt3 / 2,  dt2 / 2},
        {dt4 / 6,  dt3 / 2,  dt2,    dt   },
        {dt3 / 6,  dt2 / 2,  dt,     1.0  }
    };
    for (int jj = 0; jj < JOINTS; jj++) {
        int o = jj * SPJ;
        for (int b = 0; b < 3; b++) {
            int ob = o + b * 4;
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) ms(ekf_Q, N, ob + r, ob + c, s2 * q[r][c]);
        }
    }


    memset(ekf_R, 0, (long)M * M * sizeof(double));
    double sr2 = sr * sr;
    for (int jj = 0; jj < JOINTS; jj++) {
        int mb = jj * MPJ;
        ms(ekf_R, M, mb, mb, sr2);
        ms(ekf_R, M, mb + 1, mb + 1, sr2);
        ms(ekf_R, M, mb + 2, mb + 2, sr2);
    }
}

static void init_filter(double dt, double sj, double sr) {
    ekf_x = calloc(N, sizeof(double));
    ekf_P = calloc((long)N * N, sizeof(double));
    ekf_F = malloc((long)N * N * sizeof(double));
    ekf_Q = malloc((long)N * N * sizeof(double));
    ekf_R = malloc((long)M * M * sizeof(double));
    build_matrices(dt, sj, sr);

    for (int jj = 0; jj < JOINTS && jj < ds.nj; jj++) {
        int b = jj * SPJ;
        ekf_x[b + 0] = ds.nd[ds.j[jj].x][0];
        ekf_x[b + 4] = ds.nd[ds.j[jj].y][0];
        ekf_x[b + 8] = ds.nd[ds.j[jj].z][0];
    }

    double vars[4] = { 10,100,1000,10000 };
    for (int jj = 0; jj < JOINTS; jj++) {
        int b = jj * SPJ;
        for (int i = 0; i < SPJ; i++) ms(ekf_P, N, b + i, b + i, vars[i % 4]);
    }
}


static void build_H_full(double* H, const double* x) {
    memset(H, 0, (long)M * N * sizeof(double));
    double Hj[36];
    for (int jj = 0; jj < JOINTS; jj++) {
        int sb = jj * SPJ, mb = jj * MPJ;
        double px = x[sb + 0], py = x[sb + 4], pz = x[sb + 8];
        build_jblock_asm(Hj, px, py, pz);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 12; c++)
                ms(H, N, mb + r, sb + c, Hj[r * 12 + c]);
    }
}

static void inv_S_blockdiag(double* Si, const double* S) {
    memset(Si, 0, (long)M * M * sizeof(double));
    double blk[9], inv[9];
    int ok;
    for (int jj = 0; jj < JOINTS; jj++) {
        int off = jj * MPJ;
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
            blk[r * 3 + c] = mg(S, M, off + r, off + c);
        inv3x3_asm(inv, blk, &ok);
        if (!ok) inv_chol_c(inv, blk, 3);
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
            ms(Si, M, off + r, off + c, inv[r * 3 + c]);
    }
}


static void ekf_predict(double dt) {
    for (int jj = 0; jj < JOINTS; jj++)
        predict_x_asm(ekf_x + jj * SPJ, dt);

    double* Ft = malloc((long)N * N * sizeof(double));
    double* FP = malloc((long)N * N * sizeof(double));
    double* FPFt = malloc((long)N * N * sizeof(double));
    mT(Ft, ekf_F, N, N);
    mmul(FP, ekf_F, ekf_P, N, N, N);
    mmul(FPFt, FP, Ft, N, N, N);
    mat_add_asm(ekf_P, FPFt, ekf_Q, N * N);
    free(Ft); free(FP); free(FPFt);
}


static void ekf_update(int t) {
   
    double z[M];
    for (int jj = 0; jj < JOINTS && jj < ds.nj; jj++) {
        double sph[3];
        h_func_asm(ds.nd[ds.j[jj].x][t],
            ds.nd[ds.j[jj].y][t],
            ds.nd[ds.j[jj].z][t], sph);
        z[jj * 3] = sph[0]; z[jj * 3 + 1] = sph[1]; z[jj * 3 + 2] = sph[2];
    }


    double* H = malloc((long)M * N * sizeof(double));
    build_H_full(H, ekf_x);


    double hx[M];
    for (int jj = 0; jj < JOINTS; jj++) {
        double sph[3];
        h_func_asm(ekf_x[jj * SPJ + 0], ekf_x[jj * SPJ + 4], ekf_x[jj * SPJ + 8], sph);
        hx[jj * 3] = sph[0]; hx[jj * 3 + 1] = sph[1]; hx[jj * 3 + 2] = sph[2];
    }

  
    double y[M];
    mat_sub_asm(y, z, hx, M);
    for (int jj = 0; jj < JOINTS; jj++) {
        y[jj * 3 + 1] = wrap_angle_asm(y[jj * 3 + 1]);
        y[jj * 3 + 2] = wrap_angle_asm(y[jj * 3 + 2]);
    }

    double* Ht = malloc((long)N * M * sizeof(double));
    double* HP = malloc((long)M * N * sizeof(double));
    double* HPHt = malloc((long)M * M * sizeof(double));
    double* S = malloc((long)M * M * sizeof(double));
    double* Si = malloc((long)M * M * sizeof(double));
    mT(Ht, H, M, N);
    mmul(HP, H, ekf_P, M, N, N);
    mmul(HPHt, HP, Ht, M, N, M);
    mat_add_asm(S, HPHt, ekf_R, M * M);
    inv_S_blockdiag(Si, S);

    double* PHt = malloc((long)N * M * sizeof(double));
    double* K = malloc((long)N * M * sizeof(double));
    mmul(PHt, ekf_P, Ht, N, N, M);
    mmul(K, PHt, Si, N, M, M);


    double* Ky = malloc(N * sizeof(double));
    mmul(Ky, K, y, N, M, 1);
    mat_add_asm(ekf_x, ekf_x, Ky, N);
    free(Ky);


    double* KH = malloc((long)N * N * sizeof(double));
    double* IKH = malloc((long)N * N * sizeof(double));
    double* IKHt = malloc((long)N * N * sizeof(double));
    double* IKHP = malloc((long)N * N * sizeof(double));
    double* t1 = malloc((long)N * N * sizeof(double));
    double* KR = malloc((long)N * M * sizeof(double));
    double* Kt = malloc((long)M * N * sizeof(double));
    double* t2 = malloc((long)N * N * sizeof(double));

    mmul(KH, K, H, N, M, N);
    mI(IKH, N);
    mat_sub_asm(IKH, IKH, KH, N * N);
    mT(IKHt, IKH, N, N);
    mmul(IKHP, IKH, ekf_P, N, N, N);
    mmul(t1, IKHP, IKHt, N, N, N);
    mmul(KR, K, ekf_R, N, M, M);
    mT(Kt, K, N, M);
    mmul(t2, KR, Kt, N, M, N);
    mat_add_asm(ekf_P, t1, t2, N * N);

    free(H); free(Ht); free(HP); free(HPHt); free(S); free(Si);
    free(PHt); free(K);
    free(KH); free(IKH); free(IKHt); free(IKHP);
    free(t1); free(KR); free(Kt); free(t2);
}


static void save_states(double** st, int nf, const char* fn) {
    FILE* fp = fopen(fn, "w"); if (!fp) { perror(fn); return; }
    fprintf(fp, "Frame");
    for (int jj = 0; jj < JOINTS; jj++) {
        const char* nm = ds.j[jj].name;
        fprintf(fp, ",%s_px,%s_vx,%s_ax,%s_jx,%s_py,%s_vy,%s_ay,%s_jy,%s_pz,%s_vz,%s_az,%s_jz",
            nm, nm, nm, nm, nm, nm, nm, nm, nm, nm, nm, nm);
    }
    fprintf(fp, "\n");
    for (int i = 0; i < nf; i++) {
        fprintf(fp, "%d", i);
        for (int s = 0; s < N; s++) fprintf(fp, ",%.8f", st[i][s]);
        fprintf(fp, "\n");
    }
    fclose(fp); printf("Saved: %s\n", fn);
}

static void save_cmp(double** st, int nf, const char* fn) {
    FILE* fp = fopen(fn, "w"); if (!fp) { perror(fn); return; }
    const char* nm = ds.j[0].name;
    int xc = ds.j[0].x, yc = ds.j[0].y, zc = ds.j[0].z;
    fprintf(fp, "Frame,Time,%s_True_x,%s_True_y,%s_True_z,"
        "%s_Noisy_x,%s_Noisy_y,%s_Noisy_z,"
        "%s_EKF_x,%s_EKF_y,%s_EKF_z\n",
        nm, nm, nm, nm, nm, nm, nm, nm, nm);
    for (int i = 0; i < nf; i++)
        fprintf(fp, "%d,%.4f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
            i, i * 0.01, ds.td[xc][i], ds.td[yc][i], ds.td[zc][i],
            ds.nd[xc][i], ds.nd[yc][i], ds.nd[zc][i],
            st[i][0], st[i][4], st[i][8]);
    fclose(fp); printf("Saved: %s\n", fn);
}

static void save_derivatives(double** st, int nf, const char* fn) {
    FILE* fp = fopen(fn, "w"); if (!fp) { perror(fn); return; }
    const char* jn = ds.j[0].name;
    fprintf(fp, "Frame,Time,"
        "%s_px,%s_vx,%s_ax,%s_jx,"
        "%s_py,%s_vy,%s_ay,%s_jy,"
        "%s_pz,%s_vz,%s_az,%s_jz\n",
        jn, jn, jn, jn, jn, jn, jn, jn, jn, jn, jn, jn);
    for (int i = 0; i < nf; i++) {
        fprintf(fp, "%d,%.4f", i, i * 0.01);
        for (int k = 0; k < 12; k++) fprintf(fp, ",%.8f", st[i][k]);
        fprintf(fp, "\n");
    }
    fclose(fp); printf("Saved: %s\n", fn);
}

static void print_metrics(double** st, int nf) {
    int xc = ds.j[0].x;
    double mn = 0, mf = 0, an = 0, af = 0, xn = 0, xf = 0;
    for (int i = 0; i < nf; i++) {
        double tx = ds.td[xc][i], nx = ds.nd[xc][i], fx = st[i][0];
        double en = fabs(nx - tx), ef = fabs(fx - tx);
        mn += en * en; mf += ef * ef; an += en; af += ef;
        if (en > xn) xn = en; if (ef > xf) xf = ef;
    }
    mn /= nf; mf /= nf; an /= nf; af /= nf;
    printf("\n==== VALIDATION: %s (X) ====\n", ds.j[0].name);
    printf("%-10s %-12s %-12s\n", "Metric", "Noisy", "EKF");
    printf("%-10s %-12.6f %-12.6f (%.1f%% better)\n", "RMSE",
        sqrt(mn), sqrt(mf), (1 - sqrt(mf) / sqrt(mn)) * 100);
    printf("%-10s %-12.6f %-12.6f\n", "MAE", an, af);
    printf("%-10s %-12.6f %-12.6f\n", "MaxErr", xn, xf);

    double total_n = 0, total_f = 0; long total_s = 0;
    for (int jj = 0; jj < JOINTS; jj++) {
        int xci = ds.j[jj].x, yci = ds.j[jj].y, zci = ds.j[jj].z;
        int sii = jj * SPJ;
        for (int i = 0; i < nf; i++) {
            double tx = ds.td[xci][i], ty = ds.td[yci][i], tz = ds.td[zci][i];
            double nx = ds.nd[xci][i], ny = ds.nd[yci][i], nz = ds.nd[zci][i];
            double fx = st[i][sii + 0], fy = st[i][sii + 4], fz = st[i][sii + 8];
            total_n += (nx - tx) * (nx - tx) + (ny - ty) * (ny - ty) + (nz - tz) * (nz - tz);
            total_f += (fx - tx) * (fx - tx) + (fy - ty) * (fy - ty) + (fz - tz) * (fz - tz);
            total_s += 3;
        }
    }
    printf("\nAll Joints 3D Position:\n");
    printf("  Noisy RMSE : %.6f m\n", sqrt(total_n / total_s));
    printf("  EKF   RMSE : %.6f m\n", sqrt(total_f / total_s));
    printf("  Improvement: %.2f%%\n", (1.0 - sqrt(total_f / total_n)) * 100.0);
}


int main(void) {
    printf("============================================================\n");
    printf("  EKF - RISC-V Assembly (Milestone 3)\n");
    printf("  Assembly: atan2, h_func, Jacobian, predict_x, update_x,\n");
    printf("            inv3x3 (Cramer), mat_add, mat_sub\n");
    printf("============================================================\n\n");

    memset(&ds, 0, sizeof(ds));

    const char* nnames[] = { "3DDataset(Noisy Values).csv","3DDataset_Noisy_Values_.csv",NULL };
    int loaded = 0;
    for (int fi = 0; nnames[fi]; fi++) if (load_nd(nnames[fi]) == 0) { loaded = 1; break; }
    if (!loaded) { fprintf(stderr, "Cannot find noisy CSV\n"); return 1; }

    const char* tnames[] = { "3DDataset(True Values).csv","3DDataset_True_Values_.csv",NULL };
    for (int fi = 0; tnames[fi]; fi++) if (load_td(tnames[fi]) == 0) break;

    double dt = 0.01, sj = 1.0, sr = 0.05;
    printf("dt=%.3f  sigma_j=%.1f  sigma_r=%.2f\n", dt, sj, sr);

    init_filter(dt, sj, sr);

    int nf = ds.nf;
    double** filtered = malloc(nf * sizeof(double*));
    for (int i = 0; i < nf; i++) filtered[i] = malloc(N * sizeof(double));

    printf("Processing %d frames...\n", nf);
    for (int t = 0; t < nf; t++) {
        ekf_predict(dt);
        ekf_update(t);
        memcpy(filtered[t], ekf_x, N * sizeof(double));
        if ((t + 1) % 500 == 0) { printf("  %d/%d\r", t + 1, nf); fflush(stdout); }
    }
    printf("\nDone.\n");

    print_metrics(filtered, nf);

    system("mkdir -p EKF_Output");
    save_states(filtered, nf, "EKF_Output/EKF_Filtered_States.csv");
    save_cmp(filtered, nf, "EKF_Output/EKF_Comparison_FirstJoint.csv");
    save_derivatives(filtered, nf, "EKF_Output/EKF_Derivatives_FirstJoint.csv");

    for (int i = 0; i < nf; i++) free(filtered[i]);
    free(filtered);
    free(ekf_x); free(ekf_P); free(ekf_F); free(ekf_Q); free(ekf_R);
    printf("\nOutput in EKF_Output/\n");
    return 0;
}
