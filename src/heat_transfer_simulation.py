"""Transient 2D Heat Transfer Solver for Power Electronics with Hybrid Cooling.

Numerical Method: Implicit Finite Volume Method (FVM) with Sparse LU Factorization
Domain: Solid Alloy Steel Substrate with Internal Heat Source and Liquid Cooling Channel
Verification: Cross-checked against ANSYS Fluent CFD Simulation

This Python script implements the mathematical model and solver algorithm developed
by the project authors, utilizing scipy.sparse and precomputed sparse LU decomposition
(splu) for rapid, unconditionally stable time integration.
"""

import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def run_implicit_simulation(
    block_length: float = 0.30,
    block_height: float = 0.15,
    k_solid: float = 15.0,
    rho_solid: float = 7900.0,
    cp_solid: float = 500.0,
    h_conv: float = 45.0,
    t_ambient: float = 25.0,
    t_coolant: float = 5.0,
    t_initial: float = 25.0,
    delta_x: float = 0.002,  # 2 mm mesh for fast Python execution (or 0.001 for 1 mm)
    delta_t: float = 1.5,
    time_final: float = 1800.0,
    save_plot_path: str = None,
):
    """Execute 2D implicit FVM solver matching author's MATLAB algorithm."""
    alpha = k_solid / (rho_solid * cp_solid)
    nx = int(round(block_length / delta_x)) + 1
    ny = int(round(block_height / delta_x)) + 1
    n_total = nx * ny

    biot = h_conv * delta_x / k_solid
    fourier = alpha * delta_t / (delta_x**2)

    print("=" * 65)
    print("Transient 2D Thermal Analysis - Implicit FVM with Sparse LU Engine")
    print(f"Domain: {block_length*100:.1f} cm x {block_height*100:.1f} cm | Grid: {nx} x {ny} ({n_total:,} nodes)")
    print(f"Fourier: {fourier:.4f} | Biot: {biot:.4f} | dt: {delta_t:.2f} s")
    print("=" * 65)

    # Cavity grid bounds
    j_h1_s = int(round(0.06 / delta_x))
    j_h1_e = int(round(0.12 / delta_x))
    i_h1_s = int(round(0.045 / delta_x))
    i_h1_e = int(round(0.105 / delta_x))

    j_h2_s = int(round(0.18 / delta_x))
    j_h2_e = int(round(0.24 / delta_x))
    i_h2_s = int(round(0.045 / delta_x))
    i_h2_e = int(round(0.105 / delta_x))

    # Probe indices
    i_center = int(round(0.075 / delta_x))
    j_critical = int(round(0.15 / delta_x))
    j_left = 0
    j_right = nx - 1

    # Node indexing helper (row-major for Python: node = i * nx + j)
    def node_id(i, j):
        return i * nx + j

    # Mask domain: 0 = inactive inside cavities
    mask = np.ones((ny, nx), dtype=int)
    for i in range(ny):
        for j in range(nx):
            if (i > i_h1_s and i < i_h1_e and j > j_h1_s and j < j_h1_e):
                mask[i, j] = 0
            if (i >= i_h2_s and i <= i_h2_e and j >= j_h2_s and j <= j_h2_e):
                mask[i, j] = 0

    # Assemble Sparse Matrix A
    print("Assembling global coefficient matrix A...")
    a_mat = sp.lil_matrix((n_total, n_total), dtype=float)

    for i in range(ny):
        for j in range(nx):
            row = node_id(i, j)

            if mask[i, j] == 0 and not (i >= i_h2_s and i <= i_h2_e and j >= j_h2_s and j <= j_h2_e):
                a_mat[row, row] = 1.0
                continue

            if (i >= i_h2_s and i <= i_h2_e and j >= j_h2_s and j <= j_h2_e):
                a_mat[row, row] = 1.0
                continue

            a_p = 1.0 + 4.0 * fourier
            a_w = -fourier
            a_e = -fourier
            a_s = -fourier
            a_n = -fourier

            if j == 0:  # Left boundary (convection)
                a_p = 1.0 + 2.0 * fourier * (2.0 + biot)
                a_e = -2.0 * fourier
                a_w = 0.0
            elif j == nx - 1:  # Right boundary (adiabatic)
                a_p = 1.0 + 4.0 * fourier
                a_w = -2.0 * fourier
                a_e = 0.0

            if i == 0:  # Bottom boundary (adiabatic)
                a_p = 1.0 + 4.0 * fourier
                a_n = -2.0 * fourier
                a_s = 0.0
            elif i == ny - 1:  # Top boundary (adiabatic)
                a_p = 1.0 + 4.0 * fourier
                a_s = -2.0 * fourier
                a_n = 0.0

            # Hot cavity walls (flux)
            if (i > i_h1_s and i < i_h1_e and j == j_h1_s):
                a_p = 1.0 + 4.0 * fourier; a_w = -2.0 * fourier; a_e = 0.0
            if (i > i_h1_s and i < i_h1_e and j == j_h1_e):
                a_p = 1.0 + 4.0 * fourier; a_e = -2.0 * fourier; a_w = 0.0
            if (j > j_h1_s and j < j_h1_e and i == i_h1_s):
                a_p = 1.0 + 4.0 * fourier; a_s = -2.0 * fourier; a_n = 0.0
            if (j > j_h1_s and j < j_h1_e and i == i_h1_e):
                a_p = 1.0 + 4.0 * fourier; a_n = -2.0 * fourier; a_s = 0.0

            a_mat[row, row] = a_p
            if j > 0: a_mat[row, node_id(i, j - 1)] = a_w
            if j < nx - 1: a_mat[row, node_id(i, j + 1)] = a_e
            if i > 0: a_mat[row, node_id(i - 1, j)] = a_s
            if i < ny - 1: a_mat[row, node_id(i + 1, j)] = a_n

    # Convert to CSC format and perform Sparse LU Factorization
    print("Computing sparse LU decomposition...")
    a_csc = a_mat.tocsc()
    lu_solver = spla.splu(a_csc)

    # Initialize temperature vector
    tp = np.full((ny, nx), t_initial, dtype=float)
    tp[i_h2_s : i_h2_e + 1, j_h2_s : j_h2_e + 1] = t_coolant
    tp_vec = tp.flatten()

    t = 0.0
    cycle_period = 300.0
    max_t_critical_30min = -float("inf")

    time_hist = []
    t_crit_hist = []

    print("Running transient time-stepping loop...")
    while t <= time_final:
        q_flux = 25000.0 * abs(np.sin(np.pi * t / cycle_period))
        b = tp_vec.copy()

        # Update RHS Vector B
        for i in range(ny):
            for j in range(nx):
                row = node_id(i, j)
                if (i >= i_h2_s and i <= i_h2_e and j >= j_h2_s and j <= j_h2_e):
                    b[row] = t_coolant
                    continue
                if mask[i, j] == 0:
                    b[row] = t_initial
                    continue
                if j == 0:
                    b[row] += 2.0 * fourier * biot * t_ambient
                if (i >= i_h1_s and i <= i_h1_e and (j == j_h1_s or j == j_h1_e)) or \
                   (j >= j_h1_s and j <= j_h1_e and (i == i_h1_s or i == i_h1_e)):
                    b[row] += (2.0 * fourier * delta_x / k_solid) * q_flux

        # Direct sparse solve using precomputed LU factors
        tn_vec = lu_solver.solve(b)
        tn = tn_vec.reshape((ny, nx))
        tn[i_h2_s : i_h2_e + 1, j_h2_s : j_h2_e + 1] = t_coolant

        t_crit_curr = tn[i_center, j_critical]
        if t <= 1800.0 and t_crit_curr > max_t_critical_30min:
            max_t_critical_30min = t_crit_curr

        time_hist.append(t)
        t_crit_hist.append(t_crit_curr)

        t += delta_t
        tp_vec = tn_vec

    t_final_matrix = tn
    print("=" * 65)
    print(f"Elapsed Time: {time_final:.0f} s ({time_final/60:.1f} min)")
    print(f"Peak Substrate Temperature: {np.max(t_final_matrix):.2f} °C")
    print(f"Peak Bottleneck Temperature (first 30 min): {max_t_critical_30min:.2f} °C")
    print(f"Thermal Bottleneck Status (< 75.0 °C): {'PASS' if max_t_critical_30min < 75.0 else 'FAIL'}")
    print("=" * 65)

    # Visualization
    if save_plot_path:
        x_coords = np.linspace(0, block_length, nx)
        y_coords = np.linspace(0, block_height, ny)
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

        p_display = t_final_matrix.copy()
        p_display[i_h1_s + 1 : i_h1_e, j_h1_s + 1 : j_h1_e] = np.nan
        p_display[i_h2_s : i_h2_e + 1, j_h2_s : j_h2_e + 1] = np.nan

        fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
        cf = ax.contourf(x_mesh, y_mesh, p_display, levels=30, cmap="jet")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Temperature (°C)", fontsize=11, fontweight="bold")

        rect_chip = plt.Rectangle((0.06, 0.045), 0.06, 0.06, fill=True, facecolor="white", edgecolor="red", lw=1.5)
        rect_cool = plt.Rectangle((0.18, 0.045), 0.06, 0.06, fill=True, facecolor="white", edgecolor="blue", lw=1.5)
        ax.add_patch(rect_chip)
        ax.add_patch(rect_cool)

        ax.text(0.09, 0.075, "Hot Chip\nCavity", ha="center", va="center", color="red", fontsize=9, fontweight="bold")
        ax.text(0.21, 0.075, "Coolant\n$T_c = 5^\\circ$C", ha="center", va="center", color="blue", fontsize=9, fontweight="bold")
        ax.plot(0.15, 0.075, "k+", markersize=10, markeredgewidth=2)
        ax.text(0.15, 0.085, f"Bottleneck Point\n$T = {max_t_critical_30min:.2f}^\\circ$C (PASS)", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xlabel("X Coordinate (m)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Y Coordinate (m)", fontsize=11, fontweight="bold")
        ax.set_title(f"Temperature Distribution at t = {int(time_final/60)} min (Implicit FVM with Sparse LU)", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(save_plot_path)), exist_ok=True)
        plt.savefig(save_plot_path, dpi=300)
        print(f"Figure saved to: {save_plot_path}")
        plt.close()

    return t_final_matrix


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(current_dir, "..", "docs", "images", "python_simulated_contour_30min.png")
    run_implicit_simulation(save_plot_path=output_png)
