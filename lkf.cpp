// ============================================================
//  Linear Kalman Filter (LKF) — Full-Body 3D Human Gait
//  Milestone 2
//
//  State per joint : [px vx ax jx | py vy ay jy | pz vz az jz]  (12)
//  23 joints  =>  276-dim full state vector
//  Measurement    : direct Cartesian positions [px py pz] per joint
//
//  Design decisions:
//    - All large matrices HEAP-allocated via std::vector<double>
//    - Matrix inversion AVOIDED: Cholesky decomposition solves
//      (H P Hᵀ + R) X = PHᵀ  for the Kalman gain without inversion
//    - Covariance updated with numerically-stable Joseph form:
//      P = (I-KH) P (I-KH)ᵀ + K R Kᵀ
//    - Block-diagonal sparsity of F and Q exploited: per-joint
//      12×12 prediction keeps cost O(n) in number of joints
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

// ─── compile-time dimensions ────────────────────────────────
static constexpr int N_JOINTS = 23;
static constexpr int S1       = 12;              // state per joint
static constexpr int M1       = 3;               // meas  per joint

// ─── tuning parameters ──────────────────────────────────────
static constexpr double DT = 0.01;   // 100 Hz
static constexpr double SJ = 1.0;    // jerk process-noise std-dev
static constexpr double SR = 0.01;   // position measurement noise std-dev (m)

// ============================================================
//  Dense matrix — row-major, HEAP storage
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
//  Cholesky solver: solve A*X = B  (A symmetric positive-definite)
//  Replaces explicit matrix inversion — more numerically stable.
// ============================================================
static Mat solveSPD(const Mat& A, const Mat& B) {
    int n = A.R;
    // Cholesky  A = L Lᵀ
    Mat L(n, n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A(i,j);
            for (int k = 0; k < j; ++k) s -= L(i,k) * L(j,k);
            if (i == j) {
                if (s < 1e-15) s = 1e-15;  // guard against near-zero
                L(i,j) = std::sqrt(s);
            } else {
                L(i,j) = s / L(j,j);
            }
        }
    }
    // Solve L Lᵀ X = B column-by-column
    Mat X(n, B.C);
    std::vector<double> col(n), y(n), x(n);
    for (int c = 0; c < B.C; ++c) {
        for (int i = 0; i < n; ++i) col[i] = B(i,c);
        // forward: L y = col
        for (int i = 0; i < n; ++i) {
            double s = col[i];
            for (int k = 0; k < i; ++k) s -= L(i,k) * y[k];
            y[i] = s / L(i,i);
        }
        // backward: Lᵀ x = y
        for (int i = n-1; i >= 0; --i) {
            double s = y[i];
            for (int k = i+1; k < n; ++k) s -= L(k,i) * x[k];
            x[i] = s / L(i,i);
        }
        for (int i = 0; i < n; ++i) X(i,c) = x[i];
    }
    return X;
}

// ============================================================
//  Build matrices (per-joint, 12×12 block-diagonal of 3 × 4×4)
// ============================================================
static Mat buildF_joint() {
    double dt=DT, dt2=dt*dt/2.0, dt3=dt*dt*dt/6.0;
    Mat F(S1, S1, 0.0);
    for (int b = 0; b < 3; ++b) {
        int o = b * 4;
        F(o+0,o+0)=1; F(o+0,o+1)=dt; F(o+0,o+2)=dt2; F(o+0,o+3)=dt3;
        F(o+1,o+1)=1; F(o+1,o+2)=dt; F(o+1,o+3)=dt2;
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
        int o = b * 4;
        Q(o+0,o+0)=q00; Q(o+0,o+1)=q01; Q(o+0,o+2)=q02; Q(o+0,o+3)=q03;
        Q(o+1,o+0)=q01; Q(o+1,o+1)=q11; Q(o+1,o+2)=q12; Q(o+1,o+3)=q13;
        Q(o+2,o+0)=q02; Q(o+2,o+1)=q12; Q(o+2,o+2)=q22; Q(o+2,o+3)=q23;
        Q(o+3,o+0)=q03; Q(o+3,o+1)=q13; Q(o+3,o+2)=q23; Q(o+3,o+3)=q33;
    }
    return Q;
}

static Mat buildH_joint() {
    // 3×12: picks px(0), py(4), pz(8)
    Mat H(M1, S1, 0.0);
    H(0,0)=1.0; H(1,4)=1.0; H(2,8)=1.0;
    return H;
}

// ============================================================
//  CSV I/O
// ============================================================
static std::vector<std::vector<double>> readCSV(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::string line;
    std::getline(f, line); // skip header
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
//  MAIN
// ============================================================
int main(int argc, char* argv[]) {
    std::string noisyPath = "3D Full Body Humain Gait Walking Dataset (Noisy Values).csv";
    std::string outPath   = "lkf_output.csv";
    if (argc > 1) noisyPath = argv[1];
    if (argc > 2) outPath   = argv[2];

    std::cout << "[LKF] Reading: " << noisyPath << "\n";
    auto noisy = readCSV(noisyPath);
    int T = (int)noisy.size();
    std::cout << "[LKF] " << T << " timesteps, " << noisy[0].size() << " cols\n";

    // Pre-build matrices
    Mat Fj  = buildF_joint();
    Mat Qj  = buildQ_joint();
    Mat Hj  = buildH_joint();
    Mat FjT = transpose(Fj);
    Mat HjT = transpose(Hj);
    Mat Rj(M1, M1, 0.0);
    Rj(0,0)=Rj(1,1)=Rj(2,2)=SR*SR;

    // Per-joint state and covariance (heap)
    std::vector<std::vector<double>> xj(N_JOINTS, std::vector<double>(S1, 0.0));
    std::vector<Mat> Pj(N_JOINTS, Mat(S1, S1, 0.0));

    // Initialise from first noisy measurement
    for (int j = 0; j < N_JOINTS; ++j) {
        int b = j * M1;
        xj[j][0]=noisy[0][b+0];  // px
        xj[j][4]=noisy[0][b+1];  // py
        xj[j][8]=noisy[0][b+2];  // pz
        for (int i = 0; i < S1; ++i) Pj[j](i,i) = 500.0;
    }

    std::vector<std::vector<double>> output;
    output.reserve(T);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < T; ++t) {
        std::vector<double> row;
        row.reserve(N_JOINTS * S1);

        for (int j = 0; j < N_JOINTS; ++j) {
            // Wrap state
            Mat x(S1, 1);
            for (int i = 0; i < S1; ++i) x(i,0) = xj[j][i];

            // ── PREDICTION ───────────────────────────────────
            Mat xp = matmul(Fj, x);
            Mat Pp = matadd(matmul(matmul(Fj, Pj[j]), FjT), Qj);

            // ── UPDATE ───────────────────────────────────────
            int base = j * M1;
            Mat z(M1, 1);
            z(0,0)=noisy[t][base+0];
            z(1,0)=noisy[t][base+1];
            z(2,0)=noisy[t][base+2];

            // Innovation covariance S = H Pp Hᵀ + R  (3×3)
            Mat PHT = matmul(Pp, HjT);                    // 12×3
            Mat Sinv_lhs = matadd(matmul(Hj, PHT), Rj);  // 3×3

            // Kalman gain K = PHT * S^{-1}  =>  K = solveSPD(S, PHT^T)^T
            Mat KT = solveSPD(Sinv_lhs, transpose(PHT));  // 3×12
            Mat K  = transpose(KT);                        // 12×3

            // Innovation
            Mat innov = matsub(z, matmul(Hj, xp));        // 3×1

            // State update
            Mat xu = matadd(xp, matmul(K, innov));

            // Covariance — Joseph form
            Mat IKH(S1,S1); IKH.eye();
            IKH = matsub(IKH, matmul(K, Hj));
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
            std::cout << "[LKF] step " << t+1 << "/" << T << "\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[LKF] Completed in "
              << std::chrono::duration<double>(t1-t0).count() << " s\n";

    // Build headers
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
    std::cout << "[LKF] Output: " << outPath << "\n";
    return 0;
}
