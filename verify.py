#!/usr/bin/env python3
"""
verify.py — Milestone-3 Numerical Verification
================================================
Compares Milestone-2 (pure-C) filter output CSVs against
Milestone-3 (RISC-V assembly) filter output CSVs and
produces a detailed verification table (printed + saved to
verification_table.csv and verification_report.txt).

Expected directory layout:
    Milestone2/LKF_Output/LKF_Filtered_States.csv
    Milestone2/EKF_Output/EKF_Filtered_States.csv
    LKF_Output/LKF_Filtered_States.csv     (Milestone-3)
    EKF_Output/EKF_Filtered_States.csv     (Milestone-3)

The script also accepts command-line overrides:
    python3 verify.py --m2lkf PATH --m2ekf PATH --m3lkf PATH --m3ekf PATH
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #
JOINTS = 23
SPJ    = 12          # states per joint
N      = JOINTS * SPJ  # 276 total states

# Column order inside each per-joint block
STATE_LABELS = ["px", "vx", "ax", "jx",
                "py", "vy", "ay", "jy",
                "pz", "vz", "az", "jz"]

# Equation (3): |x_asm - x_C++| <= epsilon_tol = 1e-9
TOLERANCE_PASS = 1e-9   # absolute error threshold for PASS (eq. 3)
TOLERANCE_WARN = 1e-6   # warn level — within FP rounding but above spec


# ------------------------------------------------------------------ #
#  CSV loading                                                         #
# ------------------------------------------------------------------ #
def load_states_csv(path: str) -> list[list[float]]:
    """Return list-of-rows, each row being N floats (state vector)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = [float(v) for k, v in row.items() if k != "Frame"]
            rows.append(vals)
    return rows


def load_or_die(path: str, label: str):
    if not os.path.isfile(path):
        print(f"[ERROR] {label} file not found: {path}")
        print("        Run the filter executables first, or supply --m2lkf / --m2ekf / --m3lkf / --m3ekf")
        sys.exit(1)
    return load_states_csv(path)


# ------------------------------------------------------------------ #
#  Statistics                                                          #
# ------------------------------------------------------------------ #
def rmse(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def mae(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def max_abs_err(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def min_abs_err(a: list[float], b: list[float]) -> float:
    return min(abs(x - y) for x, y in zip(a, b))


def avg_abs_err(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def extract_col(rows: list[list[float]], col: int) -> list[float]:
    return [r[col] for r in rows]


# ------------------------------------------------------------------ #
#  Per-function comparison                                             #
# ------------------------------------------------------------------ #
def compare_filter(m2_rows, m3_rows, filter_name: str) -> list[dict]:
    """Compare two filter outputs joint-by-joint and state-by-state.
    Tolerance follows Equation (3): |x_asm - x_C++| <= 1e-9.
    """
    n_frames = min(len(m2_rows), len(m3_rows))
    results = []

    for jj in range(JOINTS):
        joint_base = jj * SPJ
        for si, slab in enumerate(STATE_LABELS):
            col = joint_base + si
            a = extract_col(m2_rows[:n_frames], col)
            b = extract_col(m3_rows[:n_frames], col)

            r   = rmse(a, b)
            m   = mae(a, b)
            avg = avg_abs_err(a, b)
            mx  = max_abs_err(a, b)
            mn  = min_abs_err(a, b)

            # Equation (3): pass only if ALL frames satisfy |err| <= 1e-9
            if mx <= TOLERANCE_PASS:
                status = "PASS"
            elif mx <= TOLERANCE_WARN:
                status = "WARN"
            else:
                status = "FAIL"

            results.append({
                "filter"  : filter_name,
                "joint"   : jj,
                "state"   : slab,
                "avg_err" : avg,
                "rmse"    : r,
                "mae"     : m,
                "max_err" : mx,
                "min_err" : mn,
                "status"  : status,
            })

    return results


# ------------------------------------------------------------------ #
#  Summary table                                                       #
# ------------------------------------------------------------------ #
def summarise(results: list[dict], filter_name: str) -> dict:
    """Overall PASS/FAIL summary for one filter."""
    total  = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    max_e  = max(r["max_err"] for r in results)
    min_e  = min(r["min_err"] for r in results)
    avg_e  = sum(r["avg_err"] for r in results) / total

    return {
        "filter"  : filter_name,
        "total"   : total,
        "pass"    : passed,
        "warn"    : warned,
        "fail"    : failed,
        "avg_err" : avg_e,
        "max_err" : max_e,
        "min_err" : min_e,
        "overall" : "PASS" if failed == 0 and warned == 0 else
                    ("WARN" if failed == 0 else "FAIL"),
    }


# ------------------------------------------------------------------ #
#  Required Table A: Average error per joint                           #
#  (averaged across all 12 state components for that joint)            #
# ------------------------------------------------------------------ #
def print_table_per_joint(results: list[dict], filter_name: str):
    filt = [r for r in results if r["filter"] == filter_name]
    print(f"\n{'='*80}")
    print(f"  TABLE A — {filter_name}: Average Error per Joint  "
          f"(Eq.3 tol = {TOLERANCE_PASS:.0e})")
    print(f"  (averaged across all {SPJ} state components per joint)")
    print(f"{'='*80}")
    hdr = (f"{'Joint':>6} {'AvgAbsErr':>13} {'MaxAbsErr':>13} "
           f"{'MinAbsErr':>13} {'RMSE':>13} {'Status':>7}")
    print(hdr)
    print("-" * len(hdr))
    for jj in range(JOINTS):
        rows = [r for r in filt if r["joint"] == jj]
        if not rows: continue
        avg = sum(r["avg_err"] for r in rows) / len(rows)
        mx  = max(r["max_err"] for r in rows)
        mn  = min(r["min_err"] for r in rows)
        rm  = math.sqrt(sum(r["rmse"]**2 for r in rows) / len(rows))
        st  = "PASS" if mx <= TOLERANCE_PASS else \
              ("WARN" if mx <= TOLERANCE_WARN else "FAIL")
        mark = "**" if st == "FAIL" else ("~" if st == "WARN" else " ")
        print(f"{jj:>6} {avg:>13.3e} {mx:>13.3e} {mn:>13.3e} "
              f"{rm:>13.3e} {mark}{st}")


# ------------------------------------------------------------------ #
#  Required Table B: Average error per state component                 #
#  (averaged across all 23 joints for that state)                      #
# ------------------------------------------------------------------ #
def print_table_per_state(results: list[dict], filter_name: str):
    filt = [r for r in results if r["filter"] == filter_name]
    print(f"\n{'='*80}")
    print(f"  TABLE B — {filter_name}: Average Error per State Component  "
          f"(Eq.3 tol = {TOLERANCE_PASS:.0e})")
    print(f"  (averaged across all {JOINTS} joints per state component)")
    print(f"{'='*80}")
    hdr = (f"{'State':>6} {'AvgAbsErr':>13} {'MaxAbsErr':>13} "
           f"{'MinAbsErr':>13} {'RMSE':>13} {'Status':>7}")
    print(hdr)
    print("-" * len(hdr))
    for slab in STATE_LABELS:
        rows = [r for r in filt if r["state"] == slab]
        if not rows: continue
        avg = sum(r["avg_err"] for r in rows) / len(rows)
        mx  = max(r["max_err"] for r in rows)
        mn  = min(r["min_err"] for r in rows)
        rm  = math.sqrt(sum(r["rmse"]**2 for r in rows) / len(rows))
        st  = "PASS" if mx <= TOLERANCE_PASS else \
              ("WARN" if mx <= TOLERANCE_WARN else "FAIL")
        mark = "**" if st == "FAIL" else ("~" if st == "WARN" else " ")
        print(f"{slab:>6} {avg:>13.3e} {mx:>13.3e} {mn:>13.3e} "
              f"{rm:>13.3e} {mark}{st}")



# ------------------------------------------------------------------ #
#  Printing                                                            #
# ------------------------------------------------------------------ #
SEP = "=" * 90

def print_header():
    print(SEP)
    print("  Milestone-3 Numerical Verification Report")
    print("  Comparing Milestone-2 (C) vs Milestone-3 (RISC-V Assembly)")
    print(SEP)


def print_summary_table(summaries: list[dict]):
    print(f"\n{'='*80}")
    print("  OVERALL SUMMARY  (Eq.3 tolerance = 1e-9)")
    print(f"{'='*80}")
    hdr = (f"{'Filter':<8} {'Total':>6} {'PASS':>6} {'WARN':>6} {'FAIL':>6} "
           f"{'AvgAbsErr':>13} {'MaxAbsErr':>13} {'MinAbsErr':>13} {'Result':>7}")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        print(f"{s['filter']:<8} {s['total']:>6} {s['pass']:>6} {s['warn']:>6} {s['fail']:>6} "
              f"{s['avg_err']:>13.3e} {s['max_err']:>13.3e} {s['min_err']:>13.3e} "
              f"{s['overall']:>7}")


def print_detailed_table(results: list[dict], filter_name: str, max_show: int = 30):
    """Print the worst rows sorted by max_err."""
    print(f"\n--- {filter_name}: Worst State Comparisons (top {max_show} by MaxAbsErr) ---")
    hdr = (f"{'Joint':>6} {'State':>5} {'AvgAbsErr':>13} {'MaxAbsErr':>13} "
           f"{'MinAbsErr':>13} {'RMSE':>13} {'Status':>7}")
    print(hdr)
    print("-" * len(hdr))
    filt_res = [r for r in results if r["filter"] == filter_name]
    sorted_r = sorted(filt_res, key=lambda x: -x["max_err"])
    for r in sorted_r[:max_show]:
        mark = "**" if r["status"] == "FAIL" else ("~" if r["status"] == "WARN" else " ")
        print(f"{r['joint']:>6} {r['state']:>5} {r['avg_err']:>13.3e} {r['max_err']:>13.3e} "
              f"{r['min_err']:>13.3e} {r['rmse']:>13.3e} {mark}{r['status']}")


def print_per_joint_position_rmse(results: list[dict], filter_name: str):
    """Summarise position (px,py,pz) avg/max/min error per joint."""
    print(f"\n--- {filter_name}: Per-Joint Position Error (M2 vs M3 asm) ---")
    hdr = (f"{'Joint':>6} {'Avg_px':>11} {'Max_px':>11} {'Avg_py':>11} "
           f"{'Max_py':>11} {'Avg_pz':>11} {'Max_pz':>11}")
    print(hdr)
    print("-" * len(hdr))
    by_joint = {}
    for r in results:
        if r["filter"] != filter_name: continue
        if r["state"] not in ("px","py","pz"): continue
        by_joint.setdefault(r["joint"], {})[r["state"]] = r
    for jj in range(JOINTS):
        d = by_joint.get(jj, {})
        def g(s, k): return d[s][k] if s in d else float("nan")
        print(f"{jj:>6} {g('px','avg_err'):>11.3e} {g('px','max_err'):>11.3e} "
              f"{g('py','avg_err'):>11.3e} {g('py','max_err'):>11.3e} "
              f"{g('pz','avg_err'):>11.3e} {g('pz','max_err'):>11.3e}")


# ------------------------------------------------------------------ #
#  CSV / text save                                                     #
# ------------------------------------------------------------------ #
def save_results_csv(results: list[dict], path: str):
    fieldnames = ["filter","joint","state","avg_err","rmse","mae",
                  "max_err","min_err","status"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nDetailed CSV saved: {path}")


def save_report_txt(content: str, path: str):
    with open(path, "w") as f:
        f.write(content)
    print(f"Text report saved: {path}")


# ------------------------------------------------------------------ #
#  Spot-check: atan2 approximation quality                             #
# ------------------------------------------------------------------ #
import math as _math

def check_atan2_poly(n=2000):
    """
    Evaluate the A&S 4.4.49 minimax polynomial used in atan_poly_asm
    against math.atan over [-1,1] and report max absolute error.
    """
    # Coefficients from ekf_asm.s atan_c table
    C = [1.0, -0.3333314528, 0.1999355085, -0.1420889944,
         0.1065626393, -0.0752896400, 0.0429096138,
        -0.0161657367,  0.0028662257]

    def atan_poly(x):
        x2 = x * x
        acc = C[8]
        for c in C[7::-1]:
            acc = acc * x2 + c
        return acc * x

    max_err = 0.0
    for i in range(n):
        t = -1.0 + 2.0 * i / (n - 1)
        err = abs(atan_poly(t) - _math.atan(t))
        max_err = max(max_err, err)

    return max_err


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Milestone-3 Numerical Verification")
    parser.add_argument("--m2lkf", default="Milestone2/LKF_Output/LKF_Filtered_States.csv")
    parser.add_argument("--m2ekf", default="Milestone2/EKF_Output/EKF_Filtered_States.csv")
    parser.add_argument("--m3lkf", default="LKF_Output/LKF_Filtered_States.csv")
    parser.add_argument("--m3ekf", default="EKF_Output/EKF_Filtered_States.csv")
    parser.add_argument("--out",   default="verification_table.csv")
    parser.add_argument("--report",default="verification_report.txt")
    args = parser.parse_args()

    import io
    buf = io.StringIO()

    def tee(msg=""):
        print(msg)
        buf.write(msg + "\n")

    print_header()

    # ---- atan2 polynomial spot-check --------------------------------
    tee(SEP)
    tee("  atan2 Polynomial Accuracy Check")
    tee(SEP)
    poly_err = check_atan2_poly()
    tee(f"  Max |atan_poly(x) - atan(x)| over [-1,1]: {poly_err:.3e}")
    status_poly = "PASS" if poly_err < 1e-8 else "FAIL"
    tee(f"  Status (threshold 1e-8): {status_poly}")
    tee()

    # ---- Load CSVs --------------------------------------------------
    all_results = []
    summaries   = []

    for label, m2path, m3path in [("LKF", args.m2lkf, args.m3lkf),
                                   ("EKF", args.m2ekf, args.m3ekf)]:
        tee(SEP)
        tee(f"  {label} Comparison")
        tee(f"  Milestone-2 : {m2path}")
        tee(f"  Milestone-3 : {m3path}")
        tee(SEP)

        if not os.path.isfile(m2path):
            tee(f"  [SKIP] Milestone-2 file not found: {m2path}")
            continue
        if not os.path.isfile(m3path):
            tee(f"  [SKIP] Milestone-3 file not found: {m3path}")
            continue

        m2 = load_states_csv(m2path)
        m3 = load_states_csv(m3path)
        tee(f"  Frames — M2: {len(m2)}, M3: {len(m3)}")

        res = compare_filter(m2, m3, label)
        all_results.extend(res)

        s = summarise(res, label)
        summaries.append(s)

        # Table A: average error per joint (averaged across 12 states)
        print_table_per_joint(res, label)
        # Table B: average error per state component (averaged across 23 joints)
        print_table_per_state(res, label)
        # Detailed worst-case table
        print_detailed_table(res, label)
        # Per-joint position error breakdown
        print_per_joint_position_rmse(res, label)

    # ---- Overall summary --------------------------------------------
    if summaries:
        print_summary_table(summaries)

    # ---- Save outputs -----------------------------------------------
    if all_results:
        save_results_csv(all_results, args.out)

    # Compile text report
    tee("\n" + SEP)
    tee("  Verification complete.")
    if summaries:
        overall = "PASS" if all(s["overall"] == "PASS" for s in summaries) else "FAIL"
        tee(f"  Overall result: {overall}")
        tee(f"  atan2 poly:     {status_poly}")
    tee(SEP)

    save_report_txt(buf.getvalue(), args.report)


if __name__ == "__main__":
    main()
