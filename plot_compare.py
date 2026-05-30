#!/usr/bin/env python3
"""
plot_compare.py — Milestone-2 vs Milestone-3 Comparison Plots
===============================================================
Overlays Milestone-2 (pure-C) and Milestone-3 (RISC-V assembly)
filter outputs side-by-side to visually verify numerical equivalence.

Plots produced (saved to Plots_Compare/):
  1.  lkf_pos_m2_vs_m3.png      — LKF position joint-0: M2(C) vs M3(asm)
  2.  ekf_pos_m2_vs_m3.png      — EKF position joint-0: M2(C) vs M3(asm)
  3.  lkf_error_m2_vs_m3.png    — LKF abs error vs true: M2 vs M3
  4.  ekf_error_m2_vs_m3.png    — EKF abs error vs true: M2 vs M3
  5.  lkf_diff_m2_m3.png        — LKF |M2 - M3| per state (joint 0)
  6.  ekf_diff_m2_m3.png        — EKF |M2 - M3| per state (joint 0)
  7.  rmse_summary.png          — Side-by-side RMSE bar chart (M2 vs M3, all joints)
  8.  lkf_velocity_m2_vs_m3.png — LKF velocity comparison
  9.  ekf_velocity_m2_vs_m3.png — EKF velocity comparison

Directory convention:
  Milestone-2 outputs : Milestone2/LKF_Output/  Milestone2/EKF_Output/
  Milestone-3 outputs : LKF_Output/             EKF_Output/
"""

import csv
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("[ERROR] matplotlib / numpy required.  pip install matplotlib numpy")
    sys.exit(1)

# ------------------------------------------------------------------ #
OUT_DIR = "Plots_Compare"
DT      = 0.01
JOINTS  = 23
SPJ     = 12

os.makedirs(OUT_DIR, exist_ok=True)

COL = {
    "m2" : dict(color="#e74c3c", lw=1.8, alpha=0.85),
    "m3" : dict(color="#3498db", lw=1.5, ls="--", alpha=0.9),
    "true": dict(color="#2ecc71", lw=1.2, alpha=0.7),
}


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
def load_csv(path):
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


def load_states(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            vals = [float(row[k]) for k in fieldnames if k != "Frame"]
            rows.append(vals)
    return rows


def check(path, label):
    if not os.path.isfile(path):
        print(f"[SKIP] {label}: {path} not found")
        return False
    return True


def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def time_axis(n):
    return [i * DT for i in range(n)]


def pick_col(data, *suffixes):
    for suf in suffixes:
        for k in data:
            if k.lower().endswith(suf.lower()):
                return data[k]
    return None


# ------------------------------------------------------------------ #
#  Plot 1 & 2: Position overlay                                        #
# ------------------------------------------------------------------ #
def plot_pos_overlay(m2_cmp, m3_cmp, label, fname):
    if not (check(m2_cmp, f"M2 {label}") and check(m3_cmp, f"M3 {label}")): return
    d2 = load_csv(m2_cmp)
    d3 = load_csv(m3_cmp)
    time = d2.get("Time", time_axis(len(list(d2.values())[0])))

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(f"{label} — Joint 0 Position: Milestone-2 (C) vs Milestone-3 (Asm)",
                 fontsize=13, fontweight="bold")

    for ax, axis_name in zip(axes, ["x", "y", "z"]):
        tv = pick_col(d2, f"_true_{axis_name}")
        n2 = pick_col(d2, f"_{label.lower()}_{axis_name}", f"_filtered_{axis_name}")
        n3 = pick_col(d3, f"_{label.lower()}_{axis_name}", f"_filtered_{axis_name}")

        if tv is not None:
            ax.plot(time[:len(tv)], tv, label="True",
                    **{k: v for k, v in COL["true"].items()})
        if n2 is not None:
            ax.plot(time[:len(n2)], n2, label="M2 (C)",
                    **{k: v for k, v in COL["m2"].items()})
        if n3 is not None:
            ax.plot(time[:len(n3)], n3, label="M3 (Asm)",
                    **{k: v for k, v in COL["m3"].items()})

        ax.set_ylabel(f"{axis_name.upper()} (m)")
        ax.set_title(f"{axis_name.upper()} Position", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, fname)


# ------------------------------------------------------------------ #
#  Plot 3 & 4: Error vs true                                           #
# ------------------------------------------------------------------ #
def plot_error_compare(m2_cmp, m3_cmp, label, fname):
    if not (check(m2_cmp, f"M2 {label} err") and check(m3_cmp, f"M3 {label} err")): return
    d2 = load_csv(m2_cmp)
    d3 = load_csv(m3_cmp)
    time = d2.get("Time", time_axis(len(list(d2.values())[0])))

    tv  = pick_col(d2, "_true_x")
    f2x = pick_col(d2, f"_{label.lower()}_x", "_filtered_x")
    f3x = pick_col(d3, f"_{label.lower()}_x", "_filtered_x")
    if tv is None or f2x is None or f3x is None:
        print(f"[SKIP] {label} error compare: columns missing")
        return

    n = min(len(tv), len(f2x), len(f3x))
    e2 = [abs(f2x[i] - tv[i]) for i in range(n)]
    e3 = [abs(f3x[i] - tv[i]) for i in range(n)]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(time[:n], e2, label="M2 (C) |err|",   **{k: v for k, v in COL["m2"].items()})
    ax.plot(time[:n], e3, label="M3 (Asm) |err|", **{k: v for k, v in COL["m3"].items()})
    ax.set_title(f"{label} — Joint 0 X-Position Absolute Error: M2 vs M3", fontsize=12)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("|Error| (m)")
    ax.legend(); ax.grid(True, alpha=0.3)
    rmse2 = math.sqrt(sum(e**2 for e in e2) / n)
    rmse3 = math.sqrt(sum(e**2 for e in e3) / n)
    ax.text(0.02, 0.95, f"M2 RMSE={rmse2:.4f}m   M3 RMSE={rmse3:.4f}m",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    fig.tight_layout()
    save(fig, fname)


# ------------------------------------------------------------------ #
#  Plot 5 & 6: |M2 - M3| numerical difference                         #
# ------------------------------------------------------------------ #
def plot_diff(m2_states, m3_states, label, fname):
    if not (check(m2_states, f"M2 {label} states") and check(m3_states, f"M3 {label} states")):
        return
    r2 = load_states(m2_states)
    r3 = load_states(m3_states)
    n = min(len(r2), len(r3))
    time = time_axis(n)

    # Joint 0, all 12 states
    state_labels = ["px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"]
    diffs = [[abs(r2[i][si] - r3[i][si]) for i in range(n)] for si in range(12)]

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    fig.suptitle(f"{label} — |M2(C) − M3(Asm)| per State, Joint 0", fontsize=13)
    axes_flat = axes.flatten()
    for ax, diff, sl in zip(axes_flat, diffs, state_labels):
        ax.semilogy(time, [max(d, 1e-18) for d in diff], color="#8e44ad", lw=0.9)
        ax.set_title(sl, fontsize=9)
        ax.set_ylabel("|diff|", fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
        max_d = max(diff)
        ax.axhline(max_d, color="red", ls=":", lw=0.8, alpha=0.6)
        ax.text(0.5, 0.92, f"max={max_d:.1e}", transform=ax.transAxes,
                fontsize=7, ha="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    for ax in axes_flat:
        ax.set_xlabel("t (s)", fontsize=7)
    fig.tight_layout()
    save(fig, fname)


# ------------------------------------------------------------------ #
#  Plot 7: RMSE summary bar chart (all joints, M2 vs M3)               #
# ------------------------------------------------------------------ #
def compute_pos_rmse_per_joint(rows):
    n = len(rows)
    rmses = []
    for jj in range(JOINTS):
        base = jj * SPJ
        err = 0.0
        for i in range(n):
            px = rows[i][base + 0]
            py = rows[i][base + 4]
            pz = rows[i][base + 8]
            err += px**2 + py**2 + pz**2
        rmses.append(math.sqrt(err / (3 * n)))
    return rmses


def plot_rmse_summary(m2_lkf, m3_lkf, m2_ekf, m3_ekf, fname):
    available = {}
    for key, path in [("M2-LKF", m2_lkf), ("M3-LKF", m3_lkf),
                      ("M2-EKF", m2_ekf), ("M3-EKF", m3_ekf)]:
        if os.path.isfile(path):
            available[key] = compute_pos_rmse_per_joint(load_states(path))

    if not available:
        print("[SKIP] RMSE summary: no state files found")
        return

    x = list(range(JOINTS))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    fig.suptitle("Per-Joint Position RMS: Milestone-2 (C) vs Milestone-3 (Asm)", fontsize=13)

    for ax, ftype in zip(axes, ["LKF", "EKF"]):
        m2key = f"M2-{ftype}"
        m3key = f"M3-{ftype}"
        w = 0.35
        if m2key in available:
            ax.bar([xi - w/2 for xi in x], available[m2key], w,
                   label="M2 (C)", color="#e74c3c", alpha=0.8)
        if m3key in available:
            ax.bar([xi + w/2 for xi in x], available[m3key], w,
                   label="M3 (Asm)", color="#3498db", alpha=0.8)
        ax.set_title(ftype, fontsize=12)
        ax.set_xlabel("Joint Index")
        ax.set_ylabel("Position RMS (m)")
        ax.set_xticks(x); ax.set_xticklabels([f"J{i}" for i in x], fontsize=7)
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, fname)


# ------------------------------------------------------------------ #
#  Plot 8 & 9: Velocity comparison                                     #
# ------------------------------------------------------------------ #
def plot_vel_compare(m2_deriv, m3_deriv, label, fname):
    if not (check(m2_deriv, f"M2 {label} vel") and check(m3_deriv, f"M3 {label} vel")): return
    d2 = load_csv(m2_deriv)
    d3 = load_csv(m3_deriv)
    time = d2.get("Time", time_axis(len(list(d2.values())[0])))

    vx2 = pick_col(d2, "_vx")
    vx3 = pick_col(d3, "_vx")
    if vx2 is None or vx3 is None:
        print(f"[SKIP] {label} velocity compare: _vx not found")
        return

    n = min(len(vx2), len(vx3))
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(time[:n], vx2[:n], label="M2 (C) vx",   **{k: v for k, v in COL["m2"].items()})
    ax.plot(time[:n], vx3[:n], label="M3 (Asm) vx", **{k: v for k, v in COL["m3"].items()})
    ax.set_title(f"{label} — Joint 0 X-Velocity: M2 vs M3", fontsize=12)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Velocity (m/s)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, fname)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    print("=== Milestone-2 vs Milestone-3 Comparison Plots ===")
    print(f"Output directory: {OUT_DIR}/\n")

    m2_lkf_cmp    = "Milestone2/LKF_Output/LKF_Comparison_FirstJoint.csv"
    m2_lkf_deriv  = "Milestone2/LKF_Output/LKF_Derivatives_FirstJoint.csv"
    m2_lkf_states = "Milestone2/LKF_Output/LKF_Filtered_States.csv"

    m3_lkf_cmp    = "LKF_Output/LKF_Comparison_FirstJoint.csv"
    m3_lkf_deriv  = "LKF_Output/LKF_Derivatives_FirstJoint.csv"
    m3_lkf_states = "LKF_Output/LKF_Filtered_States.csv"

    m2_ekf_cmp    = "Milestone2/EKF_Output/EKF_Comparison_FirstJoint.csv"
    m2_ekf_deriv  = "Milestone2/EKF_Output/EKF_Derivatives_FirstJoint.csv"
    m2_ekf_states = "Milestone2/EKF_Output/EKF_Filtered_States.csv"

    m3_ekf_cmp    = "EKF_Output/EKF_Comparison_FirstJoint.csv"
    m3_ekf_deriv  = "EKF_Output/EKF_Derivatives_FirstJoint.csv"
    m3_ekf_states = "EKF_Output/EKF_Filtered_States.csv"

    print("-- Position overlays --")
    plot_pos_overlay(m2_lkf_cmp, m3_lkf_cmp, "LKF", "lkf_pos_m2_vs_m3.png")
    plot_pos_overlay(m2_ekf_cmp, m3_ekf_cmp, "EKF", "ekf_pos_m2_vs_m3.png")

    print("-- Error vs true --")
    plot_error_compare(m2_lkf_cmp, m3_lkf_cmp, "LKF", "lkf_error_m2_vs_m3.png")
    plot_error_compare(m2_ekf_cmp, m3_ekf_cmp, "EKF", "ekf_error_m2_vs_m3.png")

    print("-- Numerical difference (|M2 - M3|) --")
    plot_diff(m2_lkf_states, m3_lkf_states, "LKF", "lkf_diff_m2_m3.png")
    plot_diff(m2_ekf_states, m3_ekf_states, "EKF", "ekf_diff_m2_m3.png")

    print("-- RMSE summary --")
    plot_rmse_summary(m2_lkf_states, m3_lkf_states,
                      m2_ekf_states, m3_ekf_states, "rmse_summary.png")

    print("-- Velocity comparison --")
    plot_vel_compare(m2_lkf_deriv, m3_lkf_deriv, "LKF", "lkf_velocity_m2_vs_m3.png")
    plot_vel_compare(m2_ekf_deriv, m3_ekf_deriv, "EKF", "ekf_velocity_m2_vs_m3.png")

    print(f"\nAll comparison plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
