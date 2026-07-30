import argparse
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from clime_pytorch import clime_cpu, clime_torch


@dataclass
class SparsePrecisionProblem:
    emp_cov: np.ndarray
    true_precision: np.ndarray
    true_covariance: np.ndarray


def make_sparse_precision(
    n_features: int,
    edge_prob: float,
    seed: int,
    dtype: np.dtype,
    edge_weight_min: float,
    edge_weight_max: float,
    diagonal_shift: float,
) -> np.ndarray:
    """Create a sparse SPD precision matrix by diagonal dominance."""
    rng = np.random.default_rng(seed)
    upper_mask = rng.random((n_features, n_features)) < edge_prob
    upper_mask = np.triu(upper_mask, k=1)
    weights = rng.uniform(edge_weight_min, edge_weight_max, size=(n_features, n_features))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=dtype), size=(n_features, n_features))
    upper = upper_mask * weights * signs
    precision = upper + upper.T
    np.fill_diagonal(precision, np.abs(precision).sum(axis=1) + diagonal_shift)
    return precision.astype(dtype, copy=False)


def make_sparse_precision_problem(
    n_samples: int,
    n_features: int,
    seed: int,
    dtype: np.dtype,
    edge_prob: float,
    edge_weight_min: float,
    edge_weight_max: float,
    diagonal_shift: float,
) -> SparsePrecisionProblem:
    """Sample data from the sparse precision model and compute empirical covariance."""
    rng = np.random.default_rng(seed)
    true_precision = make_sparse_precision(
        n_features=n_features,
        edge_prob=edge_prob,
        seed=seed,
        dtype=dtype,
        edge_weight_min=edge_weight_min,
        edge_weight_max=edge_weight_max,
        diagonal_shift=diagonal_shift,
    )
    true_covariance = np.linalg.inv(true_precision)
    chol_covariance = np.linalg.cholesky(true_covariance)
    x = rng.standard_normal((n_samples, n_features)).astype(dtype) @ chol_covariance.T
    x -= x.mean(axis=0, keepdims=True)
    emp_cov = (x.T @ x) / (n_samples - 1)
    return SparsePrecisionProblem(
        emp_cov=emp_cov.astype(dtype, copy=False),
        true_precision=true_precision,
        true_covariance=true_covariance.astype(dtype, copy=False),
    )


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def format_value(value: float) -> str:
    return f"{value:.6g}"


def clime_residual(emp_cov: np.ndarray, precision: np.ndarray) -> float:
    """Return ``||S Omega - I||_max``."""
    eye = np.eye(emp_cov.shape[0], dtype=emp_cov.dtype)
    return float(np.max(np.abs(emp_cov @ precision - eye)))


def true_precision_errors(precision: np.ndarray, true_precision: np.ndarray) -> Dict[str, float]:
    diff = precision - true_precision
    true_fro = np.linalg.norm(true_precision, ord="fro")
    true_spec = np.linalg.norm(true_precision, ord=2)
    return {
        "rel_fro_to_true": float(np.linalg.norm(diff, ord="fro") / true_fro),
        "rel_spec_to_true": float(np.linalg.norm(diff, ord=2) / true_spec),
    }


def precision_stats(emp_cov: np.ndarray, precision: np.ndarray, true_precision: np.ndarray) -> Dict[str, float]:
    """Collect output diagnostics shared by CPU LP and torch ADMM."""
    offdiag = precision.copy()
    np.fill_diagonal(offdiag, 0.0)
    errors = true_precision_errors(precision, true_precision)
    return {
        "objective_l1": float(np.abs(precision).sum()),
        "constraint_max": clime_residual(emp_cov, precision),
        "nnz_offdiag": float(np.count_nonzero(np.abs(offdiag) > 1e-8)),
        "rel_fro_to_true": errors["rel_fro_to_true"],
        "rel_spec_to_true": errors["rel_spec_to_true"],
    }


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def time_solver(
    name: str,
    solver: Callable[[], Tuple[np.ndarray, int, bool, float, float]],
    repeat: int,
    warmup: int,
    problem: SparsePrecisionProblem,
    device: str,
) -> Dict[str, object]:
    for _ in range(warmup):
        solver()
    sync_if_needed(device)

    times = []
    precision = None
    n_iter = 0
    converged = True
    primal = 0.0
    dual = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        precision, n_iter, converged, primal, dual = solver()
        sync_if_needed(device)
        times.append(time.perf_counter() - start)

    stats = precision_stats(problem.emp_cov, precision, problem.true_precision)
    return {
        "name": name,
        "precision": precision,
        "mean_ms": 1000.0 * float(np.mean(times)),
        "std_ms": 1000.0 * float(np.std(times)),
        "min_ms": 1000.0 * float(np.min(times)),
        "n_iter": n_iter,
        "converged": converged,
        "primal": primal,
        "dual": dual,
        **stats,
    }


def add_cpu_reference_errors(rows: List[Dict[str, object]]) -> None:
    cpu_precision = rows[0]["precision"]
    for row in rows:
        diff = row["precision"] - cpu_precision
        row["diff_fro_vs_cpu_lp"] = float(np.linalg.norm(diff, ord="fro"))
        row["diff_spec_vs_cpu_lp"] = float(np.linalg.norm(diff, ord=2))


def print_timing_table(rows: List[Dict[str, object]]) -> None:
    print()
    print("Experiment 1: timing and output diagnostics on the first sparse recovery problem")
    headers = [
        "method",
        "mean_ms",
        "std_ms",
        "min_ms",
        "n_iter",
        "converged",
        "constraint_max",
        "objective_l1",
        "offdiag_nnz",
        "rel_fro_to_true",
        "rel_spec_to_true",
        "diff_fro_vs_cpu_lp",
        "diff_spec_vs_cpu_lp",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(
            f"{row['name']} | "
            f"{row['mean_ms']:.3f} | "
            f"{row['std_ms']:.3f} | "
            f"{row['min_ms']:.3f} | "
            f"{row['n_iter']} | "
            f"{row['converged']} | "
            f"{format_value(row['constraint_max'])} | "
            f"{format_value(row['objective_l1'])} | "
            f"{int(row['nnz_offdiag'])} | "
            f"{format_value(row['rel_fro_to_true'])} | "
            f"{format_value(row['rel_spec_to_true'])} | "
            f"{format_value(row['diff_fro_vs_cpu_lp'])} | "
            f"{format_value(row['diff_spec_vs_cpu_lp'])}"
        )


def print_stats_table(title: str, rows: List[Dict[str, object]], metric_name: str) -> None:
    print()
    print(title)
    headers = [
        "method",
        f"{metric_name}_mean",
        f"{metric_name}_std",
        f"{metric_name}_median",
        f"{metric_name}_p95",
        f"{metric_name}_max",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        stats = row["stats"]
        print(
            f"{row['name']} | "
            f"{format_value(stats['mean'])} | "
            f"{format_value(stats['std'])} | "
            f"{format_value(stats['median'])} | "
            f"{format_value(stats['p95'])} | "
            f"{format_value(stats['max'])}"
        )


def solve_cpu_lp(emp_cov: np.ndarray, lambda_: float) -> Tuple[np.ndarray, int, bool, float, float]:
    result = clime_cpu(emp_cov, lambda_=lambda_, symmetrize=False, return_info=True)
    return result.precision, 0, True, 0.0, 0.0


def solve_torch_split_admm(
    emp_cov_torch: torch.Tensor,
    lambda_: float,
    rho: float,
    eta: float,
    max_iter: int,
    tol: float,
    linear_solver: str,
) -> Tuple[np.ndarray, int, bool, float, float]:
    result = clime_torch(
        emp_cov_torch,
        lambda_=lambda_,
        rho=rho,
        eta=eta,
        max_iter=max_iter,
        tol=tol,
        symmetrize=False,
        linear_solver=linear_solver,
        return_info=True,
    )
    precision = result.precision.detach().cpu().numpy()
    return precision, result.n_iter, result.converged, result.primal_residual, result.dual_residual


def run_problem_stats(
    problems: List[SparsePrecisionProblem],
    covariances_torch: List[torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    time_rows = []
    residual_rows = []
    fro_rows = []
    spec_rows = []
    for name in ["cpu_lp", "torch_split_admm"]:
        time_values = []
        residual_values = []
        fro_values = []
        spec_values = []
        for problem, emp_cov_torch in zip(problems, covariances_torch):
            start = time.perf_counter()
            if name == "cpu_lp":
                precision, _, _, _, _ = solve_cpu_lp(problem.emp_cov, args.lambda_)
            else:
                precision, _, _, _, _ = solve_torch_split_admm(
                    emp_cov_torch,
                    args.lambda_,
                    args.rho,
                    args.eta,
                    args.max_iter,
                    args.tol,
                    args.linear_solver,
                )
                sync_if_needed(args.device)
            time_values.append(1000.0 * (time.perf_counter() - start))
            stats = precision_stats(problem.emp_cov, precision, problem.true_precision)
            residual_values.append(stats["constraint_max"])
            fro_values.append(stats["rel_fro_to_true"])
            spec_values.append(stats["rel_spec_to_true"])
        time_rows.append({"name": name, "stats": summarize(time_values)})
        residual_rows.append({"name": name, "stats": summarize(residual_values)})
        fro_rows.append({"name": name, "stats": summarize(fro_values)})
        spec_rows.append({"name": name, "stats": summarize(spec_values)})
    return time_rows, residual_rows, fro_rows, spec_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact CPU CLIME LP and PyTorch split-ADMM CLIME."
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=40)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--num-problems", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-prob", type=float, default=0.05)
    parser.add_argument("--edge-weight-min", type=float, default=0.15)
    parser.add_argument("--edge-weight-max", type=float, default=0.35)
    parser.add_argument("--diagonal-shift", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--linear-solver", type=str, default="auto", choices=["auto", "cholesky", "solve"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    np_dtype = np.float64 if args.dtype == "float64" else np.float32
    torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32

    problems = [
        make_sparse_precision_problem(
            n_samples=args.n_samples,
            n_features=args.n_features,
            seed=args.seed + idx,
            dtype=np_dtype,
            edge_prob=args.edge_prob,
            edge_weight_min=args.edge_weight_min,
            edge_weight_max=args.edge_weight_max,
            diagonal_shift=args.diagonal_shift,
        )
        for idx in range(args.num_problems)
    ]
    covariances_torch = [
        torch.as_tensor(problem.emp_cov, dtype=torch_dtype, device=args.device)
        for problem in problems
    ]

    print(
        "Simulated sparse precision recovery problems: "
        f"num_problems={args.num_problems}, n_samples={args.n_samples}, "
        f"n_features={args.n_features}, lambda={args.lambda_}, rho={args.rho}, "
        f"eta={args.eta}, tol={args.tol}, max_iter={args.max_iter}, "
        f"repeat={args.repeat}, warmup={args.warmup}, dtype={args.dtype}, "
        f"torch_device={args.device}, linear_solver={args.linear_solver}"
    )

    first_problem = problems[0]
    first_cov_torch = covariances_torch[0]
    timing_rows = [
        time_solver(
            "cpu_lp",
            lambda: solve_cpu_lp(first_problem.emp_cov, args.lambda_),
            args.repeat,
            args.warmup,
            first_problem,
            "cpu",
        ),
        time_solver(
            "torch_split_admm",
            lambda: solve_torch_split_admm(
                first_cov_torch,
                args.lambda_,
                args.rho,
                args.eta,
                args.max_iter,
                args.tol,
                args.linear_solver,
            ),
            args.repeat,
            args.warmup,
            first_problem,
            args.device,
        ),
    ]
    add_cpu_reference_errors(timing_rows)
    print_timing_table(timing_rows)

    time_rows, residual_rows, fro_rows, spec_rows = run_problem_stats(problems, covariances_torch, args)
    print_stats_table(
        f"Experiment 2a: runtime over {args.num_problems} sparse recovery problems",
        time_rows,
        "time_ms",
    )
    print_stats_table(
        f"Experiment 2b: CLIME constraint ||S Omega - I||_max over {args.num_problems} problems",
        residual_rows,
        "constraint_max",
    )
    print_stats_table(
        f"Experiment 2c: relative Frobenius error to true precision over {args.num_problems} problems",
        fro_rows,
        "rel_fro",
    )
    print_stats_table(
        f"Experiment 2d: relative spectral error to true precision over {args.num_problems} problems",
        spec_rows,
        "rel_spec",
    )


if __name__ == "__main__":
    main()
