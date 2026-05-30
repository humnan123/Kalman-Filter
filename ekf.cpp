// ============================================================
//  Extended Kalman Filter (EKF) — Full-Body 3D Human Gait
//  Milestone 2
//
//  State per joint : [px vx ax jx | py vy ay jy | pz vz az jz]  (12)
//  23 joints  =>  276-dim full state vector
//  Measurement (nonlinear):  z = [r, θ, φ]  spherical coords
//    r = sqrt(px²+py²+pz²)
//    θ = arctan2(py, px)          (azimuth)
//    φ = arctan2(pz, sqrt(px²+py²)) (elevation)
//
//  Design decisions:
//    - All large matrices HEAP-allocated via std::vector<double>
//    - Matrix inversion AVOIDED: Cholesky decomposition solves
//      (Hk Pp Hkᵀ + R) for the Kalman gain
//    - Covariance updated with Joseph form for numerical stability
//    - arctan2 implemented MANUALLY via a polynomial approximation
//      (Bhaskara I method + sign correction, no <cmath> atan2 used)
//    - Jacobian Hk recomputed each step from predicted state
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <chrono>

// ─── dimensions ─────────────────────────────────────────────
static constexpr int N_JOINTS = 23;
static constexpr int S1       = 12;
static constexpr int M1       = 3;   // r, theta, phi

// ─── tuning ─────────────────────────────────────────────────
static constexpr double DT = 0.01;
static constexpr double SJ = 1.0;
static constexpr double SR_r   = 0.05;   // range noise std-dev (m)
static constexpr double SR_ang = 0.01;   // angle noise std-dev (rad)

static constexpr double PI = 3.14159265358979323846;

// ============================================================
//  Manual arctan2 approximation
//  Method: Polynomial rational approximation of atan on [0,1],
//          then use identities to cover all quadrants.
//  Max error: ~0.0015 rad (~0.086°) — sufficient for EKF use.
//  Reference: "Efficient approximations for the arctangent function"
//             Abramowitz & Stegun, Eq 4.4.47
// ============================================================
static double manual_atan(double x) {
    // atan(x) for x in [-1, 1]  using rational polynomial
    double x2 = x * x;
    return x * (1.0 + x2 * (-0.3333314528 + x2 * (0.1999355085 + x2 *
               (-0.1420889944 + x2 * (0.1065626393 + x2 *
               (-0.0752896400 + x2 * (0.0429096138 + x2 *
               (-0.0161657367 + x2 * 0.0028662257))))))));
}

static double manual_atan2(double y, double x) {
    // Handle degenerate cases
    if (x == 0.0 && y == 0.0) return 0.0;
    if (x == 0.0) return (y > 0.0) ?  PI/2.0 : -PI/2.0;

    double ratio = y / x;
    double at;

    // Reduce to [-1,1] for stable polynomial evaluation
    if (std::fabs(ratio) <= 1.0) {
        at = manual_atan(ratio);
        if (x < 0.0) at += (y >= 0.0) ? PI : -PI;
    } else {
        // atan(y/x) = pi/2 - atan(x/y)
        at = (y > 0.0 ? PI/2.0 : -PI/2.0) - manual_atan(x / y);
        if (x < 0.0) at += (y >= 0.0) ? PI : -PI;
        // re-centre
        if (at >  PI) at -= 2.0*PI;
        if (at < -PI) at += 2.0*PI;
    }
    return at;
}

// ============================================================
//  Dense matrix — heap storage
// ============================================================
struct Mat {
    int R = 0, C = 0;
    std::vector<double> d;

    Mat() = default;
    Mat(int r, int c, double v = 0.0) : R(r), C(c), d(r * c, v) {}

    double& operator()(int r, int c)       { return d[r * C + c]; }
    double  operator()(int r, int c) const { return d[r * C + c]; }

    void zero() { std::fill(d.begin(), d.end(), 0.0); }
    void eye()  { zero(); for (int i = 0; i < R; ++i) (*this)(i,i) = 1.0; }
};

static Mat matmul(const Mat& A, const Mat& B) {
    Mat C(A.R, B.C, 0.0);
    for (int i = 0; i < A.R; ++i)
        for (int k = 0; k < A.C; ++k) {
            double a = A(i,k);
            if (a == 0.0) continue;
            for (int j = 0; j < B.C; ++j) C(i,j) += a * B(k,j);
        }
    return C;
}

static Mat transpose(const Mat& A) {
    Mat T(A.C, A.R);
    for (int i = 0; i < A.R; ++i)
        for (int j = 0; j < A.C; ++j) T(j,i) = A(i,j);
    return T;
}

static Mat matadd(const Mat& A, const Mat& B) {
    Mat C(A.R, A.C);
    for (size_t i = 0; i < A.d.size(); ++i) C.d[i] = A.d[i] + B.d[i];
    return C;
}

static Mat matsub(const Mat& A, const Mat& B) {
    Mat C(A.R, A.C);
    for (size_t i = 0; i < A.d.size(); ++i) C.d[i] = A.d[i] - B.d[i];
    return C;
}

// ============================================================
//  Cholesky solver — avoids explicit inversion
// ============================================================
static Mat solveSPD(const Mat& A, const Mat& B) {
    int n = A.R;
    Mat L(n, n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A(i,j);
            for (int k = 0; k < j; ++k) s -= L(i,k)*L(j,k);
            if (i == j) {
                if (s < 1e-15) s = 1e-15;
                L(i,j) = std::sqrt(s);
            } else {
                L(i,j) = s / L(j,j);
            }
        }
    }
    Mat X(n, B.C);
    std::vector<double> col(n), y(n), x(n);
    for (int c = 0; c < B.C; ++c) {
        for (int i = 0; i < n; ++i) col[i] = B(i,c);
        for (int i = 0; i < n; ++i) {
            double s = col[i];
            for (int k = 0; k < i; ++k) s -= L(i,k)*y[k];
            y[i] = s / L(i,i);
        }
        for (int i = n-1; i >= 0; --i) {
            double s = y[i];
            for (int k = i+1; k < n; ++k) s -= L(k,i)*x[k];
            x[i] = s / L(i,i);
        }
        for (int i = 0; i < n; ++i) X(i,c) = x[i];
    }
    return X;
}

// ============================================================
//  System matrices (same F and Q as LKF — dynamics are linear)
// ============================================================
static Mat buildF_joint() {
    double dt=DT, dt2=dt*dt/2.0, dt3=dt*dt*dt/6.0;
    Mat F(S1, S1, 0.0);
    for (int b = 0; b < 3; ++b) {
        int o = b*4;
        F(o+0,o+0)=1; F(o+0,o+1)=dt;  F(o+0,o+2)=dt2; F(o+0,o+3)=dt3;
        F(o+1,o+1)=1; F(o+1,o+2)=dt;  F(o+1,o+3)=dt2;
        F(o+2,o+2)=1; F(o+2,o+3)=dt;
        F(o+3,o+3)=1;
    }
    return F;
}

static Mat buildQ_joint() {
    double dt=DT, s2=SJ*SJ;
    double q00=s2*std::pow(dt,7)/252.0, q01=s2*std::pow(dt,6)/72.0;
    double q02=s2*std::pow(dt,5)/30.0,  q03=s2*std::pow(dt,4)/24.0;
    double q11=s2*std::pow(dt,5)/20.0,  q12=s2*std::pow(dt,4)/8.0;
    double q13=s2*std::pow(dt,3)/6.0,   q22=s2*std::pow(dt,3)/3.0;
    double q23=s2*dt*dt/2.0,            q33=s2*dt;
    Mat Q(S1, S1, 0.0);
    for (int b = 0; b < 3; ++b) {
        int o = b*4;
        Q(o+0,o+0)=q00; Q(o+0,o+1)=q01; Q(o+0,o+2)=q02; Q(o+0,o+3)=q03;
        Q(o+1,o+0)=q01; Q(o+1,o+1)=q11; Q(o+1,o+2)=q12; Q(o+1,o+3)=q13;
        Q(o+2,o+0)=q02; Q(o+2,o+1)=q12; Q(o+2,o+2)=q22; Q(o+2,o+3)=q23;
        Q(o+3,o+0)=q03; Q(o+3,o+1)=q13; Q(o+3,o+2)=q23; Q(o+3,o+3)=q33;
    }
    return Q;
}

// ============================================================
//  EKF nonlinear measurement function h(x)  →  [r, θ, φ]
// ============================================================
static Mat h_func(double px, double py, double pz) {
    double r   = std::sqrt(px*px + py*py + pz*pz);
    if (r < 1e-9) r = 1e-9;
    double rho = std::sqrt(px*px + py*py);
    if (rho < 1e-9) rho = 1e-9;
    Mat z(M1, 1);
    z(0,0) = r;
    z(1,0) = manual_atan2(py, px);            // azimuth θ
    z(2,0) = manual_atan2(pz, rho);           // elevation φ
    return z;
}

// ============================================================
//  EKF Jacobian  Hk = ∂h/∂x  (3×12)
//  Only columns 0 (px), 4 (py), 8 (pz) are non-zero.
// ============================================================
static Mat buildHk(double px, double py, double pz) {
    double r2  = px*px + py*py + pz*pz;
    double r   = std::sqrt(r2);
    if (r  < 1e-9) r  = 1e-9;
    double rho2 = px*px + py*py;
    double rho  = std::sqrt(rho2);
    if (rho < 1e-9) rho = 1e-9;

    Mat Hk(M1, S1, 0.0);

    // ∂r/∂p
    Hk(0, 0) =  px / r;
    Hk(0, 4) =  py / r;
    Hk(0, 8) =  pz / r;

    // ∂θ/∂p  (azimuth)
    Hk(1, 0) = -py / rho2;
    Hk(1, 4) =  px / rho2;
    Hk(1, 8) =  0.0;

    // ∂φ/∂p  (elevation)
    Hk(2, 0) = -px * pz / (r2 * rho);
    Hk(2, 4) = -py * pz / (r2 * rho);
    Hk(2, 8) =  rho / r2;

    return Hk;
}

// ============================================================
//  CSV I/O
// ============================================================
static std::vector<std::vector<double>> readCSV(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::string line;
    std::getline(f, line);
    std::vector<std::vector<double>> rows;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line.back()=='\r') line.pop_back();
        std::vector<double> row;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) row.push_back(std::stod(tok));
        rows.push_back(row);
    }
    return rows;
}

static void writeCSV(const std::string& path,
                     const std::vector<std::string>& headers,
                     const std::vector<std::vector<double>>& data) {
    std::ofstream f(path);
    for (int i = 0; i < (int)headers.size(); ++i) {
        f << headers[i];
        if (i+1 < (int)headers.size()) f << ',';
    }
    f << '\n';
    for (auto& row : data) {
        for (int i = 0; i < (int)row.size(); ++i) {
            f << std::fixed << std::setprecision(8) << row[i];
            if (i+1 < (int)row.size()) f << ',';
        }
        f << '\n';
    }
}

// ============================================================
//  Angle-difference wrapping helper (keep innovation in [-π, π])
// ============================================================
static double wrapAngle(double a) {
    while (a >  PI) a -= 2.0*PI;
    while (a < -PI) a += 2.0*PI;
    return a;
}

// ============================================================
//  MAIN
// ============================================================
int main(int argc, char* argv[]) {
    std::string noisyPath = "3D Full Body Humain Gait Walking Dataset (Noisy Values).csv";
    std::string outPath   = "ekf_output.csv";
    if (argc > 1) noisyPath = argv[1];
    if (argc > 2) outPath   = argv[2];

    std::cout << "[EKF] Reading: " << noisyPath << "\n";
    auto noisy = readCSV(noisyPath);
    int T = (int)noisy.size();
    std::cout << "[EKF] " << T << " timesteps, " << noisy[0].size() << " cols\n";

    // Pre-build linear matrices
    Mat Fj  = buildF_joint();
    Mat Qj  = buildQ_joint();
    Mat FjT = transpose(Fj);

    // Measurement noise R (3×3 diagonal)
    Mat Rj(M1, M1, 0.0);
    Rj(0,0) = SR_r   * SR_r;
    Rj(1,1) = SR_ang * SR_ang;
    Rj(2,2) = SR_ang * SR_ang;

    // Per-joint state and covariance
    std::vector<std::vector<double>> xj(N_JOINTS, std::vector<double>(S1, 0.0));
    std::vector<Mat> Pj(N_JOINTS, Mat(S1, S1, 0.0));

    // Initialise from first noisy measurement
    for (int j = 0; j < N_JOINTS; ++j) {
        int b = j * M1;
        xj[j][0]=noisy[0][b+0];
        xj[j][4]=noisy[0][b+1];
        xj[j][8]=noisy[0][b+2];
        for (int i = 0; i < S1; ++i) Pj[j](i,i) = 500.0;
    }

    std::vector<std::vector<double>> output;
    output.reserve(T);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < T; ++t) {
        std::vector<double> row;
        row.reserve(N_JOINTS * S1);

        for (int j = 0; j < N_JOINTS; ++j) {
            Mat x(S1, 1);
            for (int i = 0; i < S1; ++i) x(i,0) = xj[j][i];

            // ── PREDICTION (linear dynamics) ─────────────────
            Mat xp = matmul(Fj, x);
            Mat Pp = matadd(matmul(matmul(Fj, Pj[j]), FjT), Qj);

            // ── UPDATE ───────────────────────────────────────
            // Read noisy Cartesian position, convert to spherical measurement
            int base = j * M1;
            double npx = noisy[t][base+0];
            double npy = noisy[t][base+1];
            double npz = noisy[t][base+2];
            Mat z_meas = h_func(npx, npy, npz);   // [r, θ, φ]

            // Predicted measurement from predicted state
            double ppx = xp(0,0), ppy = xp(4,0), ppz = xp(8,0);
            Mat z_pred = h_func(ppx, ppy, ppz);

            // Innovation with angle wrapping
            Mat innov(M1, 1);
            innov(0,0) = z_meas(0,0) - z_pred(0,0);          // range diff
            innov(1,0) = wrapAngle(z_meas(1,0) - z_pred(1,0)); // azimuth diff
            innov(2,0) = wrapAngle(z_meas(2,0) - z_pred(2,0)); // elevation diff

            // Jacobian at predicted state
            Mat Hk  = buildHk(ppx, ppy, ppz);
            Mat HkT = transpose(Hk);

            // Innovation covariance S = Hk Pp Hkᵀ + R
            Mat PHkT = matmul(Pp, HkT);                       // 12×3
            Mat S    = matadd(matmul(Hk, PHkT), Rj);          // 3×3

            // Kalman gain via Cholesky solve
            Mat KT = solveSPD(S, transpose(PHkT));             // 3×12
            Mat K  = transpose(KT);                            // 12×3

            // State update
            Mat xu = matadd(xp, matmul(K, innov));

            // Covariance — Joseph form
            Mat IKH(S1,S1); IKH.eye();
            IKH = matsub(IKH, matmul(K, Hk));
            Mat Pu = matadd(
                matmul(matmul(IKH, Pp), transpose(IKH)),
                matmul(matmul(K, Rj), transpose(K))
            );

            for (int i = 0; i < S1; ++i) xj[j][i] = xu(i,0);
            Pj[j] = Pu;
            for (int i = 0; i < S1; ++i) row.push_back(xu(i,0));
        }
        output.push_back(row);
        if ((t+1) % 500 == 0)
            std::cout << "[EKF] step " << t+1 << "/" << T << "\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[EKF] Completed in "
              << std::chrono::duration<double>(t1-t0).count() << " s\n";

    // Headers
    std::vector<std::string> jnames = {
        "pelvis","L5","L3","T12","T8","neck","head",
        "shoulderRight","upperArmRight","forearmRight","handRight",
        "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
        "upperLegRight","lowerLegRight","footRight","toeRight",
        "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
    };
    std::vector<std::string> snames = {
        "px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"
    };
    std::vector<std::string> headers;
    for (auto& jn : jnames)
        for (auto& sn : snames)
            headers.push_back(jn + "_" + sn);

    writeCSV(outPath, headers, output);
    std::cout << "[EKF] Output: " << outPath << "\n";
    return 0;
}
