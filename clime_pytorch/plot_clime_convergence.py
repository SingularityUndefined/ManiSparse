import argparse
import os
from pathlib import Path

import numpy as np
import torch

from clime_pytorch import clime_cpu, clime_torch
from clime_pytorch.benchmark_clime import clime_residual, make_sparse_precision_problem


def as_dtype(dtype_name: str):
    """Return matching numpy and torch dtypes."""
    if dtype_name == "float64":
        return np.float64, torch.float64
    if dtype_name == "float32":
        return np.float32, torch.float32
    raise ValueError("dtype must be one of: 'float32', 'float64'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot split-ADMM CLIME convergence diagnostics.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=40)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-prob", type=float, default=0.05)
    parser.add_argument("--edge-weight-min", type=float, default=0.15)
    parser.add_argument("--edge-weight-max", type=float, default=0.35)
    parser.add_argument("--diagonal-shift", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--output", type=Path, default=Path("clime_pytorch/figures/clime_admm_convergence.png"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    np_dtype, torch_dtype = as_dtype(args.dtype)
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

    result = clime_torch(
        emp_cov_torch,
        lambda_=args.lambda_,
        rho=args.rho,
        eta=args.eta,
        max_iter=args.max_iter,
        tol=args.tol,
        symmetrize=False,
        return_info=True,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()

    cpu_result = clime_cpu(problem.emp_cov, lambda_=args.lambda_, symmetrize=False, return_info=True)
    cpu_objective = float(np.abs(cpu_result.raw_precision).sum())
    cpu_constraint = clime_residual(problem.emp_cov, cpu_result.raw_precision)
    cpu_nnz_ratio_percent = 100.0 * float(np.count_nonzero(np.abs(cpu_result.raw_precision) > 1e-8)) / cpu_result.raw_precision.size

    matplotlib_cache = Path(f"/tmp/matplotlib-cache-{os.getuid()}")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib.pyplot as plt

    iterations = np.arange(1, result.n_iter + 1)
    primal = np.asarray(result.residual_history, dtype=np.float64)
    dual = np.asarray(result.dual_residual_history, dtype=np.float64)
    constraint = np.asarray(result.constraint_history, dtype=np.float64)
    objective = np.asarray(result.objective_history, dtype=np.float64)
    nnz_ratio_percent = 100.0 * np.asarray(result.nnz_ratio_history, dtype=np.float64)
    overshoot = np.maximum(constraint - args.lambda_, 0.0)
    objective_gap = np.abs(objective - cpu_objective) / max(cpu_objective, np.finfo(np.float64).eps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    eps = np.finfo(np.float64).tiny
    axes[0].semilogy(iterations, np.maximum(primal, eps), label="primal residual")
    axes[0].semilogy(iterations, np.maximum(dual, eps), label="dual residual")
    axes[0].axhline(args.tol, color="black", linestyle="--", linewidth=1.0, label=f"tol={args.tol:g}")
    axes[0].set_xlabel("ADMM iteration")
    axes[0].set_ylabel("max-norm residual")
    axes[0].set_title("Split-ADMM residuals")
    axes[0].grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[0].legend()

    axes[1].semilogy(iterations, np.maximum(overshoot, eps), label="constraint overshoot")
    axes[1].semilogy(iterations, np.maximum(objective_gap, eps), label="relative L1 objective gap")
    axes[1].axhline(max(cpu_constraint - args.lambda_, eps), color="black", linestyle="--", linewidth=1.0, label="CPU LP overshoot")
    axes[1].set_xlabel("ADMM iteration")
    axes[1].set_ylabel("diagnostic value")
    axes[1].set_title("Returned precision diagnostics")
    axes[1].grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[1].legend()

    axes[2].plot(iterations, nnz_ratio_percent, color="tab:green", linewidth=1.6, label="torch split-ADMM")
    axes[2].axhline(cpu_nnz_ratio_percent, color="black", linestyle="--", linewidth=1.0, label="CPU LP")
    axes[2].set_xlabel("ADMM iteration")
    axes[2].set_ylabel("nnz ratio (%)")
    axes[2].set_title("Returned matrix sparsity")
    axes[2].set_ylim(0.0, 100.0)
    axes[2].grid(True, linewidth=0.4, alpha=0.35)
    axes[2].legend()

    fig.suptitle(
        "CLIME split-ADMM convergence "
        f"(p={args.n_features}, lambda={args.lambda_}, rho={args.rho}, eta={args.eta})"
    )
    fig.savefig(args.output, dpi=180)

    print(
        "CLIME split-ADMM convergence: "
        f"n_iter={result.n_iter}, converged={result.converged}, "
        f"final_primal={result.primal_residual:.6g}, "
        f"final_dual={result.dual_residual:.6g}, "
        f"final_constraint={constraint[-1]:.6g}, "
        f"final_constraint_overshoot={overshoot[-1]:.6g}, "
        f"final_rel_objective_gap={objective_gap[-1]:.6g}, "
        f"final_nnz_ratio={nnz_ratio_percent[-1]:.2f}%, "
        f"cpu_nnz_ratio={cpu_nnz_ratio_percent:.2f}%, "
        f"cpu_constraint={cpu_constraint:.6g}, "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
