/* main_driver.c – Milestone 4 Master Driver: LKF + EKF */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

extern FILE* lkf_fptr;
extern FILE* ekf_fptr;

static long lkf_frame_count = 0;
static long ekf_frame_count = 0;

/* Assembly filter functions */
extern void lkf_predict(void);
extern void lkf_update(void);
extern void ekf_predict(void);
extern void ekf_update(void);

/* Exposed state vectors for verification */
extern double lkf_x[4];
extern double ekf_x[4];
extern FILE  *lkf_fptr;
extern FILE  *ekf_fptr;


static void run_verify(const char *label, const char *out_f, const char *ref_f)
{
    printf("\n=== Numerical Verification: %s ===\n", label);

    FILE *fo = fopen(out_f, "r");
    FILE *fr = fopen(ref_f, "r");

    if (!fo) { printf("  [SKIP] Output file '%s' not found.\n", out_f); if(fr) fclose(fr); return; }
    if (!fr) {
        printf("  [INFO] Reference file '%s' not found.\n", ref_f);
        printf("  Output file exists with %ld bytes.\n", (long)(fseek(fo,0,SEEK_END),ftell(fo)));
        fclose(fo);
        return;
    }

    /* skip headers */
    char line[256];
    { char *_r = fgets(line, sizeof(line), fo); (void)_r; }
    { char *_r = fgets(line, sizeof(line), fr); (void)_r; }

    long   frames = 0;
    double max_err = 0.0, sum_err = 0.0;
    const double TOL = 1e-9;
    int pass = 1;

    while (1) {
        long   fo_frame, fr_frame;
        double fo_v[4], fr_v[4];
        int ro = fscanf(fo, "%ld,%lf,%lf,%lf,%lf\n",
                        &fo_frame, &fo_v[0], &fo_v[1], &fo_v[2], &fo_v[3]);
        int rr = fscanf(fr, "%ld,%lf,%lf,%lf,%lf\n",
                        &fr_frame, &fr_v[0], &fr_v[1], &fr_v[2], &fr_v[3]);
        if (ro != 5 || rr != 5) break;
        frames++;
        for (int k = 0; k < 4; k++) {
            double e = fabs(fo_v[k] - fr_v[k]);
            if (e > max_err) max_err = e;
            sum_err += e;
            if (e > TOL) pass = 0;
        }
    }

    double avg_err = (frames > 0) ? sum_err / (frames * 4) : 0.0;
    printf("  Frames compared : %ld\n", frames);
    printf("  Max |error|     : %.4e\n", max_err);
    printf("  Avg |error|     : %.4e\n", avg_err);
    printf("  Tolerance       : %.0e\n", TOL);
    printf("  PASS            : %s\n", pass ? "YES" : "NO (exceeds tolerance)");

    fclose(fo);
    fclose(fr);
}

int main(void)
{
    printf("=== MILESTONE 4: FINAL MASTER DRIVER (LKF + EKF) ===\n");

    // 1. Open the output files for writing
    lkf_fptr = fopen("lkf_output.csv", "w");
    ekf_fptr = fopen("ekf_output.csv", "w");

    if (lkf_fptr == NULL || ekf_fptr == NULL) {
        printf("[ERROR] Failed to open output files for writing.\n");
        if (lkf_fptr) fclose(lkf_fptr);
        if (ekf_fptr) fclose(ekf_fptr);
        return 1;
    }

    // 2. Write the mandatory headers (so run_verify's fgets has a line to skip)
    fprintf(lkf_fptr, "frame,px,py,vx,vy\n");
    fprintf(ekf_fptr, "frame,px,py,vx,vy\n");

    // 3. Define frame logging counters
    static long lkf_frame_count = 0;
    static long ekf_frame_count = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const int FRAMES = 3040;
    for (int i = 0; i < FRAMES; i++) {
        // Run LKF Cycle and log state
        lkf_predict();
        lkf_update();
        fprintf(lkf_fptr, "%ld,%.15f,%.15f,%.15f,%.15f\n",
            lkf_frame_count++, lkf_x[0], lkf_x[1], lkf_x[2], lkf_x[3]);

        // Run EKF Cycle and log state
        ekf_predict();
        ekf_update();
        fprintf(ekf_fptr, "%ld,%.15f,%.15f,%.15f,%.15f\n",
            ekf_frame_count++, ekf_x[0], ekf_x[1], ekf_x[2], ekf_x[3]);
    }

    /* flush CSV files */
    if (lkf_fptr) fclose(lkf_fptr);
    if (ekf_fptr) fclose(ekf_fptr);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1e3 +
        (t1.tv_nsec - t0.tv_nsec) * 1e-6;
    printf("[INFO] Simulation Done. (%.2f ms/frame)\n", elapsed_ms / FRAMES);

    printf("\n[INFO] Final LKF state: px=%.4f py=%.4f vx=%.4f vy=%.4f\n",
        lkf_x[0], lkf_x[1], lkf_x[2], lkf_x[3]);
    printf("[INFO] Final EKF state: px=%.4f py=%.4f vx=%.4f vy=%.4f\n",
        ekf_x[0], ekf_x[1], ekf_x[2], ekf_x[3]);

    run_verify("LINEAR KALMAN FILTER (LKF)", "lkf_output.csv", "lkf_ref.csv");
    run_verify("EXTENDED KALMAN FILTER (EKF)", "ekf_output.csv", "ekf_ref.csv");

    return 0;
}