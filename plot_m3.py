
import csv
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
except ImportError:
    print("[ERROR] matplotlib / numpy not installed. Run: pip install matplotlib numpy")
    sys.exit(1)


OUT_DIR   = "Plots_M3"
DT        = 0.01
JOINTS    = 23
SPJ       = 12

os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    "true"     : dict(color="#2ecc71", lw=1.8, label="True"),
    "noisy"    : dict(color="#e74c3c", lw=0.8, alpha=0.6, label="Noisy"),
    "filtered" : dict(color="#3498db", lw=1.8, label="Filtered (Asm)"),
}


def load_comparison(path):
    """Load *_Comparison_FirstJoint.csv → dict of column lists."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


def load_derivatives(path):
    """Load *_Derivatives_FirstJoint.csv → dict of column lists."""
    return load_comparison(path)


def load_states(path):
    """Load full *_Filtered_States.csv → list of rows (list of float)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            vals = [float(row[k]) for k in fieldnames if k != "Frame"]
            rows.append(vals)
    return rows, [k for k in fieldnames if k != "Frame"]


def check_file(path, label):
    if not os.path.isfile(path):
        print(f"[SKIP] {label}: {path} not found.")
        return False
    return True


def time_axis(n):
    return [i * DT for i in range(n)]



def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def tri_panel(time, true_vals, noisy_vals, filt_vals,
              titles, ylabels, suptitle, filename):
    """3-row panel for X, Y, Z position or derivatives."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    for ax, tv, nv, fv, title, ylabel in zip(axes, true_vals, noisy_vals, filt_vals,
                                              titles, ylabels):
        if tv  is not None: ax.plot(time, tv, **STYLE["true"])
        if nv  is not None: ax.plot(time, nv, **STYLE["noisy"])
        if fv  is not None: ax.plot(time, fv, **STYLE["filtered"])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, filename)



def plot_position(cmp_path, label, fname):
    if not check_file(cmp_path, f"{label} comparison"): return
    d = load_comparison(cmp_path)

    keys = list(d.keys())
    time = d.get("Time", time_axis(len(d[keys[0]])))

    true_x = noisy_x = filt_x = None
    true_y = noisy_y = filt_y = None
    true_z = noisy_z = filt_z = None
    for k in keys:
        kl = k.lower()
        if "true_x" in kl: true_x = d[k]
        elif "true_y" in kl: true_y = d[k]
        elif "true_z" in kl: true_z = d[k]
        elif "noisy_x" in kl: noisy_x = d[k]
        elif "noisy_y" in kl: noisy_y = d[k]
        elif "noisy_z" in kl: noisy_z = d[k]
        elif ("ekf_x" in kl or "lkf_x" in kl or "filtered_x" in kl or
              (k.endswith("_x") and "true" not in kl and "noisy" not in kl)):
            filt_x = d[k]
        elif ("ekf_y" in kl or "lkf_y" in kl or "filtered_y" in kl or
              (k.endswith("_y") and "true" not in kl and "noisy" not in kl)):
            filt_y = d[k]
        elif ("ekf_z" in kl or "lkf_z" in kl or "filtered_z" in kl or
              (k.endswith("_z") and "true" not in kl and "noisy" not in kl)):
            filt_z = d[k]

    tri_panel(
        time,
        [true_x, true_y, true_z],
        [noisy_x, noisy_y, noisy_z],
        [filt_x, filt_y, filt_z],
        ["X Position", "Y Position", "Z Position"],
        ["X (m)", "Y (m)", "Z (m)"],
        f"{label} — Joint 0 Position: True vs Noisy vs Filtered",
        fname,
    )



def plot_all_kinematics(deriv_path, cmp_path, label, joint_name="Joint 0"):
    """
    Produces 4 figures, one per kinematic quantity (pos/vel/acc/jerk),
    each with 3 subplots for x, y, z axes.
    """
    if not check_file(deriv_path, f"{label} derivatives"): return

    dd = load_derivatives(deriv_path)
    keys = list(dd.keys())
    time = dd.get("Time", time_axis(len(dd[keys[0]])))


    def pick(suffix):
        for k in keys:
            if k.lower().endswith(suffix.lower()): return dd[k]
        return [0.0] * len(time)

    true_x = noisy_x = est_x = None
    true_y = noisy_y = est_y = None
    true_z = noisy_z = est_z = None
    if check_file(cmp_path, f"{label} cmp"):
        dc = load_comparison(cmp_path)
        dkeys = list(dc.keys())
        for k in dkeys:
            kl = k.lower()
            if "true_x"  in kl: true_x  = dc[k]
            elif "true_y"  in kl: true_y  = dc[k]
            elif "true_z"  in kl: true_z  = dc[k]
            elif "noisy_x" in kl: noisy_x = dc[k]
            elif "noisy_y" in kl: noisy_y = dc[k]
            elif "noisy_z" in kl: noisy_z = dc[k]
            elif k.lower().endswith("_x") and "true" not in kl and "noisy" not in kl:
                est_x = dc[k]
            elif k.lower().endswith("_y") and "true" not in kl and "noisy" not in kl:
                est_y = dc[k]
            elif k.lower().endswith("_z") and "true" not in kl and "noisy" not in kl:
                est_z = dc[k]


    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"{label} — {joint_name}: Position (x, y, z)", fontsize=14, fontweight="bold")
    for ax, tv, nv, fv, axis in zip(axes,
                                     [true_x,  true_y,  true_z],
                                     [noisy_x, noisy_y, noisy_z],
                                     [est_x,   est_y,   est_z],
                                     ["X", "Y", "Z"]):
        if tv  is not None: ax.plot(time[:len(tv)],  tv,  color="#2ecc71", lw=1.8, label="True")
        if nv  is not None: ax.plot(time[:len(nv)],  nv,  color="#e74c3c", lw=0.8, alpha=0.5, label="Noisy")
        if fv  is not None: ax.plot(time[:len(fv)],  fv,  color="#3498db", lw=1.8, label="Filtered (Asm)")
        ax.set_ylabel(f"{axis} (m)"); ax.set_title(f"{axis} Position"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, f"{label.lower()}_position_xyz_joint0.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"{label} — {joint_name}: Velocity (x, y, z)", fontsize=14, fontweight="bold")
    for ax, sfx, axis, col in zip(axes,
                                   ["_vx","_vy","_vz"],
                                   ["X","Y","Z"],
                                   ["#e67e22","#9b59b6","#1abc9c"]):
        vals = pick(sfx)
        ax.plot(time[:len(vals)], vals[:len(time)], color=col, lw=1.5)
        ax.set_ylabel(f"v{axis.lower()} (m/s)"); ax.set_title(f"{axis} Velocity"); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, f"{label.lower()}_velocity_xyz_joint0.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"{label} — {joint_name}: Acceleration (x, y, z)", fontsize=14, fontweight="bold")
    for ax, sfx, axis, col in zip(axes,
                                   ["_ax","_ay","_az"],
                                   ["X","Y","Z"],
                                   ["#e67e22","#9b59b6","#1abc9c"]):
        vals = pick(sfx)
        ax.plot(time[:len(vals)], vals[:len(time)], color=col, lw=1.5)
        ax.set_ylabel(f"a{axis.lower()} (m/s²)"); ax.set_title(f"{axis} Acceleration"); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, f"{label.lower()}_acceleration_xyz_joint0.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"{label} — {joint_name}: Jerk (x, y, z)", fontsize=14, fontweight="bold")
    for ax, sfx, axis, col in zip(axes,
                                   ["_jx","_jy","_jz"],
                                   ["X","Y","Z"],
                                   ["#e67e22","#9b59b6","#1abc9c"]):
        vals = pick(sfx)
        ax.plot(time[:len(vals)], vals[:len(time)], color=col, lw=1.5)
        ax.set_ylabel(f"j{axis.lower()} (m/s³)"); ax.set_title(f"{axis} Jerk"); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, f"{label.lower()}_jerk_xyz_joint0.png")

    print(f"  [{label}] 4 kinematic plots saved (position/velocity/acceleration/jerk)")



def plot_derivatives(deriv_path, label, fname):
    """Legacy: X-axis only vel/acc/jerk. Use plot_all_kinematics for full requirement."""
    if not check_file(deriv_path, f"{label} derivatives"): return
    d = load_derivatives(deriv_path)
    keys = list(d.keys())
    time = d.get("Time", time_axis(len(d[keys[0]])))
    def pick(suffix):
        for k in keys:
            if k.lower().endswith(suffix): return d[k]
        return None
    vx = pick("_vx"); ax_ = pick("_ax"); jx = pick("_jx")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"{label} — Joint 0 Derivatives (Assembly Output)", fontsize=14)
    for ax, vals, title, ylabel in zip(axes,
            [vx, ax_, jx], ["Velocity X", "Acceleration X", "Jerk X"],
            ["vel (m/s)", "acc (m/s²)", "jerk (m/s³)"]):
        if vals is not None:
            ax.plot(time, vals, color="#9b59b6", lw=1.5, label=title)
        ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, fname)



def plot_error(cmp_path, label, fname):
    if not check_file(cmp_path, f"{label} error"): return
    d = load_comparison(cmp_path)
    keys = list(d.keys())
    time = d.get("Time", time_axis(len(d[keys[0]])))

    def pick(suffix):
        for k in keys:
            if k.lower().endswith(suffix): return d[k]
        return None

    true_x = pick("_true_x"); noisy_x = pick("_noisy_x"); filt_x = pick("_ekf_x") or pick("_lkf_x")
    if true_x is None or noisy_x is None or filt_x is None:
        print(f"[SKIP] {label} error plot: columns not found")
        return

    n = min(len(true_x), len(noisy_x), len(filt_x))
    err_noisy = [abs(noisy_x[i] - true_x[i]) for i in range(n)]
    err_filt  = [abs(filt_x[i]  - true_x[i]) for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time[:n], err_noisy, color="#e74c3c", lw=0.9, alpha=0.7, label="|Noisy – True|")
    ax.plot(time[:n], err_filt,  color="#3498db", lw=1.5, label="|Filtered – True|")
    ax.set_title(f"{label} — Joint 0  X-Position Absolute Error", fontsize=13)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute Error (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, fname)



def plot_joint_rmse(states_path, label, fname):
    if not check_file(states_path, f"{label} states"): return
    rows, cols = load_states(states_path)
    n = len(rows)
    rmses = []
    for jj in range(JOINTS):
        base = jj * SPJ      
        px = [rows[i][base + 0] for i in range(n)]
        py = [rows[i][base + 4] for i in range(n)]
        pz = [rows[i][base + 8] for i in range(n)]
        def rms(v): return math.sqrt(sum(x**2 for x in v) / len(v))
        rmses.append(math.sqrt((rms(px)**2 + rms(py)**2 + rms(pz)**2) / 3))

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#3498db" if r < 0.05 else "#e67e22" if r < 0.15 else "#e74c3c" for r in rmses]
    bars = ax.bar(range(JOINTS), rmses, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(JOINTS))
    ax.set_xticklabels([f"J{i}" for i in range(JOINTS)], fontsize=8)
    ax.set_xlabel("Joint Index")
    ax.set_ylabel("Position RMS (m)")
    ax.set_title(f"{label} — Per-Joint Position RMS (Assembly Output)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, fname)



def plot_atan2_accuracy():
    C = [1.0, -0.3333314528, 0.1999355085, -0.1420889944,
         0.1065626393, -0.0752896400, 0.0429096138,
        -0.0161657367,  0.0028662257]

    def atan_poly(x):
        x2 = x * x
        acc = C[8]
        for c in C[7::-1]:
            acc = acc * x2 + c
        return acc * x

    xs = [i / 500.0 - 1.0 for i in range(1001)]
    poly_vals = [atan_poly(x) for x in xs]
    true_vals = [math.atan(x)  for x in xs]
    err       = [abs(poly_vals[i] - true_vals[i]) for i in range(len(xs))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(xs, true_vals, color="#2ecc71", lw=2, label="math.atan(x)")
    ax1.plot(xs, poly_vals, "--", color="#e74c3c", lw=1.5, label="atan_poly_asm (9-term)")
    ax1.set_title("atan Polynomial Approximation vs Reference", fontsize=12)
    ax1.set_ylabel("atan(x)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.semilogy(xs, [max(e, 1e-18) for e in err], color="#9b59b6", lw=1.5)
    ax2.set_title("Absolute Error |atan_poly(x) − atan(x)|", fontsize=12)
    ax2.set_xlabel("x"); ax2.set_ylabel("Absolute Error (log)")
    ax2.axhline(1e-8, color="gray", ls="--", lw=1, label="1e-8 threshold")
    ax2.legend(); ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("EKF atan_poly_asm Accuracy (A&S 4.4.49, 9 Terms)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "ekf_atan2_accuracy.png")



def plot_lkf_vs_ekf(lkf_cmp, ekf_cmp, lkf_states, ekf_states):
    """
    Side-by-side and overlay comparison of LKF and EKF filtered outputs.
    Includes:
      (a) Position overlay (x, y, z) — joint 0
      (b) Absolute error vs true for both filters
      (c) Per-joint position RMS bar chart (LKF vs EKF)
    """

    dl = None; de = None
    if check_file(lkf_cmp, "LKF cmp for LvE"): dl = load_comparison(lkf_cmp)
    if check_file(ekf_cmp, "EKF cmp for LvE"): de = load_comparison(ekf_cmp)

    if dl is not None or de is not None:
        ref = dl if dl is not None else de
        time = ref.get("Time", time_axis(len(list(ref.values())[0])))

        def pick_col(data, *suffixes):
            if data is None: return None
            for suf in suffixes:
                for k in data:
                    if k.lower().endswith(suf.lower()): return data[k]
            return None

        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        fig.suptitle("LKF vs EKF — Joint 0 Filtered Position (x, y, z)",
                     fontsize=14, fontweight="bold")

        for ax, axis in zip(axes, ["x","y","z"]):
            tv  = pick_col(dl, f"_true_{axis}")
            lv  = pick_col(dl, f"_lkf_{axis}", f"_filtered_{axis}")
            ev  = pick_col(de, f"_ekf_{axis}", f"_filtered_{axis}")
            nv  = pick_col(dl, f"_noisy_{axis}")

            if tv is not None:
                ax.plot(time[:len(tv)], tv, color="#2ecc71", lw=1.5, alpha=0.7, label="True")
            if nv is not None:
                ax.plot(time[:len(nv)], nv, color="#bdc3c7", lw=0.6, alpha=0.5, label="Noisy")
            if lv is not None:
                ax.plot(time[:len(lv)], lv, color="#3498db", lw=1.8, label="LKF (Asm)")
            if ev is not None:
                ax.plot(time[:len(ev)], ev, color="#e74c3c", lw=1.8, ls="--", label="EKF (Asm)")

            ax.set_ylabel(f"{axis.upper()} (m)")
            ax.set_title(f"{axis.upper()} Position", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        save(fig, "lkf_vs_ekf_position_joint0.png")

    # ---- (b) Error comparison: |LKF-True| vs |EKF-True| ------------
    if dl is not None and de is not None:
        def pck(data, *sfx):
            for s in sfx:
                for k in data:
                    if k.lower().endswith(s): return data[k]
            return None
        tv  = pck(dl, "_true_x")
        lv  = pck(dl, "_lkf_x", "_filtered_x")
        ev  = pck(de, "_ekf_x", "_filtered_x")
        time2 = dl.get("Time", time_axis(len(list(dl.values())[0])))

        if tv and lv and ev:
            n = min(len(tv), len(lv), len(ev))
            el = [abs(lv[i] - tv[i]) for i in range(n)]
            ee = [abs(ev[i] - tv[i]) for i in range(n)]

            fig, ax = plt.subplots(figsize=(13, 4))
            ax.plot(time2[:n], el, color="#3498db", lw=1.5, label="LKF |error|")
            ax.plot(time2[:n], ee, color="#e74c3c", lw=1.5, ls="--", label="EKF |error|")
            ax.set_title("LKF vs EKF — Joint 0 X-Position Absolute Error vs True",
                         fontsize=12)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("|Error| (m)")
            rmse_l = math.sqrt(sum(e**2 for e in el)/n)
            rmse_e = math.sqrt(sum(e**2 for e in ee)/n)
            ax.text(0.02, 0.95,
                    f"LKF RMSE={rmse_l:.4f}m   EKF RMSE={rmse_e:.4f}m\n"
                    f"EKF improvement: {(1-rmse_e/rmse_l)*100:.1f}%",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            save(fig, "lkf_vs_ekf_error_joint0.png")

    # ---- (c) Per-joint RMSE bar: LKF vs EKF ------------------------
    def compute_joint_rms(rows):
        n = len(rows)
        rmses = []
        for jj in range(JOINTS):
            base = jj * SPJ
            err = 0.0
            for i in range(n):
                err += rows[i][base+0]**2 + rows[i][base+4]**2 + rows[i][base+8]**2
            rmses.append(math.sqrt(err / (3*n)))
        return rmses

    lkf_r = ekf_r = None
    if check_file(lkf_states, "LKF states LvE"):
        lkf_rows, _ = load_states(lkf_states)
        lkf_r = compute_joint_rms(lkf_rows)
    if check_file(ekf_states, "EKF states LvE"):
        ekf_rows, _ = load_states(ekf_states)
        ekf_r = compute_joint_rms(ekf_rows)

    if lkf_r or ekf_r:
        x = list(range(JOINTS))
        w = 0.35
        fig, ax = plt.subplots(figsize=(15, 5))
        if lkf_r:
            ax.bar([xi - w/2 for xi in x], lkf_r, w,
                   label="LKF", color="#3498db", alpha=0.85, edgecolor="white")
        if ekf_r:
            ax.bar([xi + w/2 for xi in x], ekf_r, w,
                   label="EKF", color="#e74c3c", alpha=0.85, edgecolor="white")
        ax.set_title("LKF vs EKF — Per-Joint Position RMS (All 23 Joints)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Joint Index"); ax.set_ylabel("Position RMS (m)")
        ax.set_xticks(x); ax.set_xticklabels([f"J{i}" for i in x], fontsize=8)
        ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        save(fig, "lkf_vs_ekf_joint_rmse.png")

    print("  LKF vs EKF comparison plots saved.")


def main():
    print("=== Milestone-3 Plot Generation ===")
    print(f"Output directory: {OUT_DIR}/\n")

    # LKF paths
    lkf_cmp   = "LKF_Output/LKF_Comparison_FirstJoint.csv"
    lkf_deriv = "LKF_Output/LKF_Derivatives_FirstJoint.csv"
    lkf_states= "LKF_Output/LKF_Filtered_States.csv"

    # EKF paths
    ekf_cmp   = "EKF_Output/EKF_Comparison_FirstJoint.csv"
    ekf_deriv = "EKF_Output/EKF_Derivatives_FirstJoint.csv"
    ekf_states= "EKF_Output/EKF_Filtered_States.csv"

    print("-- Full kinematic plots (pos/vel/acc/jerk × x,y,z) --")
    plot_all_kinematics(lkf_deriv, lkf_cmp, "LKF")
    plot_all_kinematics(ekf_deriv, ekf_cmp, "EKF")


    print("-- True / Noisy / Estimated position comparison --")
    plot_position(lkf_cmp, "LKF", "lkf_position_joint0.png")
    plot_position(ekf_cmp, "EKF", "ekf_position_joint0.png")


    print("-- LKF vs EKF comparison --")
    plot_lkf_vs_ekf(lkf_cmp, ekf_cmp, lkf_states, ekf_states)

    print("-- Absolute error plots --")
    plot_error(lkf_cmp, "LKF", "lkf_error_joint0.png")
    plot_error(ekf_cmp, "EKF", "ekf_error_joint0.png")

    print("-- Per-joint RMSE bar charts --")
    plot_joint_rmse(lkf_states, "LKF", "lkf_all_joints_rmse.png")
    plot_joint_rmse(ekf_states, "EKF", "ekf_all_joints_rmse.png")

    print("-- atan2 accuracy plot --")
    plot_atan2_accuracy()

    print(f"\nAll plots saved to {OUT_DIR}/")
    print("\nPlot summary:")
    print("  lkf_position_xyz_joint0.png     — LKF position x/y/z (True/Noisy/Filtered)")
    print("  lkf_velocity_xyz_joint0.png     — LKF velocity x/y/z")
    print("  lkf_acceleration_xyz_joint0.png — LKF acceleration x/y/z")
    print("  lkf_jerk_xyz_joint0.png         — LKF jerk x/y/z")
    print("  ekf_position_xyz_joint0.png     — EKF position x/y/z (True/Noisy/Filtered)")
    print("  ekf_velocity_xyz_joint0.png     — EKF velocity x/y/z")
    print("  ekf_acceleration_xyz_joint0.png — EKF acceleration x/y/z")
    print("  ekf_jerk_xyz_joint0.png         — EKF jerk x/y/z")
    print("  lkf_vs_ekf_position_joint0.png  — LKF vs EKF position overlay")
    print("  lkf_vs_ekf_error_joint0.png     — LKF vs EKF error vs true")
    print("  lkf_vs_ekf_joint_rmse.png       — LKF vs EKF per-joint RMS bar")
    print("  lkf_position_joint0.png         — LKF True/Noisy/Filtered (standalone)")
    print("  ekf_position_joint0.png         — EKF True/Noisy/Filtered (standalone)")
    print("  lkf_error_joint0.png / ekf_error_joint0.png — absolute error plots")
    print("  lkf_all_joints_rmse.png / ekf_all_joints_rmse.png — per-joint bars")
    print("  ekf_atan2_accuracy.png          — atan_poly_asm accuracy")


if __name__ == "__main__":
    main()
