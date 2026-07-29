import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import torch

from glasso_pytorch import graphical_lasso
from glasso_pytorch.benchmark_glasso import (
    dual_gap,
    make_sparse_precision_problem,
    precision_stats,
    quic,
    solve_quic,
    solve_sklearn,
    true_precision_errors,
)


def parse_csv_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def collect_admm_path(
    emp_cov_np: np.ndarray,
    emp_cov_torch: torch.Tensor,
    true_precision: np.ndarray,
    alpha: float,
    rho: float,
    checkpoints: List[int],
    fixed_tol: float,
    eps: float,
    eigh_shift: float,
    eigh_shift_retries: int,
    eigh_cpu_fallback: bool,
) -> Dict[str, List[float]]:
    nnz_values = []
    gap_values = []
    fro_values = []
    spec_values = []
    for n_iter in checkpoints:
        result = graphical_lasso(
            emp_cov_torch,
            alpha=alpha,
            rho=rho,
            max_iter=n_iter,
            tol=fixed_tol,
            rtol=fixed_tol,
            eps=eps,
            eigh_shift=eigh_shift,
            eigh_shift_retries=eigh_shift_retries,
            eigh_cpu_fallback=eigh_cpu_fallback,
            return_info=True,
        )
        precision = result.precision.detach().cpu().numpy()
        errors = true_precision_errors(precision, true_precision)
        nnz_values.append(float(precision_stats(precision)["nnz_offdiag"]))
        gap_values.append(abs(dual_gap(emp_cov_np, precision, alpha)))
        fro_values.append(errors["rel_fro_to_true"])
        spec_values.append(errors["rel_spec_to_true"])
    return {"nnz": nnz_values, "gap": gap_values, "fro": fro_values, "spec": spec_values}


def collect_solver_path(
    solver_name: str,
    emp_cov_np: np.ndarray,
    true_precision: np.ndarray,
    alpha: float,
    checkpoints: List[int],
    fixed_tol: float,
) -> Dict[str, List[float]]:
    nnz_values = []
    gap_values = []
    fro_values = []
    spec_values = []
    for n_iter in checkpoints:
        if solver_name == "sklearn_cd":
            precision, _, gap = solve_sklearn(emp_cov_np, alpha, fixed_tol, n_iter)
        elif solver_name == "quic":
            precision, _, gap = solve_quic(emp_cov_np, alpha, fixed_tol, n_iter)
        else:
            raise ValueError(f"unknown solver: {solver_name}")
        errors = true_precision_errors(precision, true_precision)
        nnz_values.append(float(precision_stats(precision)["nnz_offdiag"]))
        gap_values.append(abs(gap))
        fro_values.append(errors["rel_fro_to_true"])
        spec_values.append(errors["rel_spec_to_true"])
    return {"nnz": nnz_values, "gap": gap_values, "fro": fro_values, "spec": spec_values}


def time_to_tol(
    emp_cov_np: np.ndarray,
    emp_cov_torch: torch.Tensor,
    alpha: float,
    rhos: List[float],
    tol: float,
    max_iter: int,
    device: str,
    eps: float,
    eigh_shift: float,
    eigh_shift_retries: int,
    eigh_cpu_fallback: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    def add_row(name: str, rho: object, precision: np.ndarray, n_iter: int, gap: float, seconds: float, converged: object) -> None:
        rows.append(
            {
                "method": name,
                "rho": rho,
                "n_iter": n_iter,
                "time_ms": 1000.0 * seconds,
                "abs_dual_gap": abs(gap),
                "offdiag_nnz": precision_stats(precision)["nnz_offdiag"],
                "converged": converged,
            }
        )

    start = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    wall_start = time.perf_counter()
    precision, n_iter, gap = solve_sklearn(emp_cov_np, alpha, tol, max_iter)
    add_row("sklearn_cd", "-", precision, n_iter, gap, time.perf_counter() - wall_start, "reported")

    if quic is not None:
        wall_start = time.perf_counter()
        precision, n_iter, gap = solve_quic(emp_cov_np, alpha, tol, max_iter)
        add_row("quic", "-", precision, n_iter, gap, time.perf_counter() - wall_start, "reported")

    for rho in rhos:
        if device == "cuda":
            torch.cuda.synchronize()
            start.record()
        wall_start = time.perf_counter()
        result = graphical_lasso(
            emp_cov_torch,
            alpha=alpha,
            rho=rho,
            max_iter=max_iter,
            tol=tol,
            rtol=tol,
            eps=eps,
            eigh_shift=eigh_shift,
            eigh_shift_retries=eigh_shift_retries,
            eigh_cpu_fallback=eigh_cpu_fallback,
            return_info=True,
        )
        if device == "cuda":
            end.record()
            torch.cuda.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
        else:
            seconds = time.perf_counter() - wall_start
        precision = result.precision.detach().cpu().numpy()
        add_row(
            "torch_admm",
            f"{rho:g}",
            precision,
            int(result.n_iter),
            dual_gap(emp_cov_np, precision, alpha),
            seconds,
            result.converged,
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot fixed-checkpoint solver paths and ADMM rho sensitivity."
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--rhos", type=str, default="0.25,0.5,1.0,2.0,5.0,10.0")
    parser.add_argument("--checkpoints", type=str, default="1,2,5,10,20,40,60,80,100,150")
    parser.add_argument("--fixed-tol", type=float, default=1e-20)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--convergence-max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-prob", type=float, default=0.02)
    parser.add_argument("--edge-weight-min", type=float, default=0.15)
    parser.add_argument("--edge-weight-max", type=float, default=0.35)
    parser.add_argument("--diagonal-shift", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--eps", type=float, default=0.0, help="covariance jitter; changes the GLASSO problem")
    parser.add_argument("--eigh-shift", type=float, default=1e-6, help="eigensolver-only shift; recovered after eigh")
    parser.add_argument("--eigh-shift-retries", type=int, default=4)
    parser.add_argument("--no-eigh-cpu-fallback", dest="eigh_cpu_fallback", action="store_false")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("glasso_pytorch/figures/admm_rho_path.png"),
    )
    parser.set_defaults(eigh_cpu_fallback=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    rhos = parse_csv_floats(args.rhos)
    checkpoints = parse_csv_ints(args.checkpoints)
    np_dtype = np.float64 if args.dtype == "float64" else np.float32
    torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32

    problem = make_sparse_precision_problem(
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed,
        dtype=np_dtype,
        edge_prob=args.edge_prob,
        edge_weight_min=args.edge_weight_min,
        edge_weight_max=args.edge_weight_max,
        diagonal_shift=args.diagonal_shift,
    )
    emp_cov_torch = torch.as_tensor(problem.emp_cov, dtype=torch_dtype, device=args.device)
    true_nnz = precision_stats(problem.true_precision)["nnz_offdiag"]

    admm_paths = {
        rho: collect_admm_path(
            problem.emp_cov,
            emp_cov_torch,
            problem.true_precision,
            alpha=args.alpha,
            rho=rho,
            checkpoints=checkpoints,
            fixed_tol=args.fixed_tol,
            eps=args.eps,
            eigh_shift=args.eigh_shift,
            eigh_shift_retries=args.eigh_shift_retries,
            eigh_cpu_fallback=args.eigh_cpu_fallback,
        )
        for rho in rhos
    }
    baseline_paths = {
        "sklearn_cd": collect_solver_path(
            "sklearn_cd",
            problem.emp_cov,
            problem.true_precision,
            args.alpha,
            checkpoints,
            args.fixed_tol,
        )
    }
    if quic is not None:
        baseline_paths["quic"] = collect_solver_path(
            "quic",
            problem.emp_cov,
            problem.true_precision,
            args.alpha,
            checkpoints,
            args.fixed_tol,
        )

    time_rows = time_to_tol(
        problem.emp_cov,
        emp_cov_torch,
        alpha=args.alpha,
        rhos=rhos,
        tol=args.tol,
        max_iter=args.convergence_max_iter,
        device=args.device,
        eps=args.eps,
        eigh_shift=args.eigh_shift,
        eigh_shift_retries=args.eigh_shift_retries,
        eigh_cpu_fallback=args.eigh_cpu_fallback,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    ax_nnz, ax_gap = axes[0]
    ax_fro, ax_spec = axes[1]

    for rho, path in admm_paths.items():
        label = f"rho={rho:g}"
        ax_nnz.plot(checkpoints, path["nnz"], marker="o", linewidth=1.8, label=label)
        ax_gap.plot(checkpoints, path["gap"], marker="o", linewidth=1.8, label=label)
        ax_fro.plot(checkpoints, path["fro"], marker="o", linewidth=1.8, label=label)
        ax_spec.plot(checkpoints, path["spec"], marker="o", linewidth=1.8, label=label)

    baseline_styles = {
        "sklearn_cd": {"color": "black", "linestyle": ":", "marker": "s", "label": "sklearn_cd"},
        "quic": {"color": "gray", "linestyle": "-.", "marker": "D", "label": "quic"},
    }
    for name, path in baseline_paths.items():
        style = baseline_styles[name]
        ax_nnz.plot(
            checkpoints,
            path["nnz"],
            linewidth=2.2,
            markersize=4,
            **style,
        )
        ax_gap.plot(
            checkpoints,
            path["gap"],
            linewidth=2.2,
            markersize=4,
            **style,
        )
        ax_fro.plot(
            checkpoints,
            path["fro"],
            linewidth=2.2,
            markersize=4,
            **style,
        )
        ax_spec.plot(
            checkpoints,
            path["spec"],
            linewidth=2.2,
            markersize=4,
            **style,
        )

    ax_nnz.axhline(true_nnz, color="black", linestyle="--", linewidth=1.2, label="true nnz")
    ax_nnz.set_title("Off-diagonal nnz")
    ax_nnz.set_ylabel("offdiag nnz")
    ax_nnz.set_yscale("symlog", linthresh=100)
    ax_nnz.grid(True, alpha=0.3)

    ax_gap.set_title("Absolute dual gap")
    ax_gap.set_ylabel("abs dual gap")
    ax_gap.set_yscale("log")
    ax_gap.grid(True, which="both", alpha=0.3)

    ax_fro.set_title("Relative Frobenius error to true precision")
    ax_fro.set_xlabel("max_iter checkpoint")
    ax_fro.set_ylabel("relative Frobenius error")
    ax_fro.grid(True, alpha=0.3)

    ax_spec.set_title("Relative spectral error to true precision")
    ax_spec.set_xlabel("max_iter checkpoint")
    ax_spec.set_ylabel("relative spectral error")
    ax_spec.grid(True, alpha=0.3)

    for ax in [ax_nnz, ax_gap]:
        ax.set_xlabel("max_iter checkpoint")

    handles, labels = ax_nnz.get_legend_handles_labels()
    ax_gap.legend(handles, labels, loc="upper right", fontsize=8)

    fig.suptitle(
        (
            f"Solver path diagnostic: n_samples={args.n_samples}, "
            f"p={args.n_features}, alpha={args.alpha}, seed={args.seed}, "
            f"eigh_shift={args.eigh_shift:g}"
        ),
        fontsize=12,
    )
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"saved: {args.output}")
    print(
        "glasso settings: "
        f"eps={args.eps:g}, eigh_shift={args.eigh_shift:g}, "
        f"eigh_shift_retries={args.eigh_shift_retries}, "
        f"eigh_cpu_fallback={args.eigh_cpu_fallback}"
    )
    print(f"true_offdiag_nnz: {true_nnz}")
    for name, path in baseline_paths.items():
        print(
            f"{name}: final_checkpoint={checkpoints[-1]}, "
            f"final_nnz={path['nnz'][-1]:.0f}, final_abs_gap={path['gap'][-1]:.6g}"
        )
    for rho, path in admm_paths.items():
        print(
            f"rho={rho:g}: final_checkpoint={checkpoints[-1]}, "
            f"final_nnz={path['nnz'][-1]:.0f}, final_abs_gap={path['gap'][-1]:.6g}"
        )
    print()
    print(f"CPU/GPU time to tol={args.tol} with max_iter={args.convergence_max_iter}")
    print("method | rho | n_iter | time_ms | abs_dual_gap | offdiag_nnz | converged")
    print("--- | --- | ---: | ---: | ---: | ---: | ---")
    for row in time_rows:
        print(
            f"{row['method']} | {row['rho']} | {row['n_iter']} | "
            f"{row['time_ms']:.3f} | {row['abs_dual_gap']:.6g} | "
            f"{row['offdiag_nnz']} | {row['converged']}"
        )


if __name__ == "__main__":
    main()
