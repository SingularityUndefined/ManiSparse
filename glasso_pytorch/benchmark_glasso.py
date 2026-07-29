import argparse
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.covariance import graphical_lasso as sklearn_graphical_lasso
from sklearn.exceptions import ConvergenceWarning

from glasso_pytorch import graphical_lasso as torch_graphical_lasso

try:
    from inverse_covariance import quic
except ImportError:
    quic = None


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


def dual_gap(emp_cov: np.ndarray, precision: np.ndarray, alpha: float) -> float:
    gap = float(np.sum(emp_cov * precision) - precision.shape[0])
    offdiag_l1 = np.abs(precision).sum() - np.abs(np.diag(precision)).sum()
    return gap + alpha * float(offdiag_l1)


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


def print_stats_table(title: str, rows: List[Dict[str, object]], metric_name: str) -> None:
    print()
    print(title)
    headers = ["method", f"{metric_name}_mean", f"{metric_name}_std", f"{metric_name}_median", f"{metric_name}_p95", f"{metric_name}_max"]
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


def precision_stats(precision: np.ndarray) -> Dict[str, object]:
    precision = 0.5 * (precision + precision.T)
    eig_min = float(np.linalg.eigvalsh(precision).min())
    offdiag = precision.copy()
    np.fill_diagonal(offdiag, 0.0)
    return {
        "precision": precision,
        "eig_min": eig_min,
        "nnz_offdiag": int(np.count_nonzero(np.abs(offdiag) > 1e-8)),
    }


def true_precision_errors(precision: np.ndarray, true_precision: np.ndarray) -> Dict[str, float]:
    diff = precision - true_precision
    true_fro = np.linalg.norm(true_precision, ord="fro")
    true_spec = np.linalg.norm(true_precision, ord=2)
    return {
        "rel_fro_to_true": float(np.linalg.norm(diff, ord="fro") / true_fro),
        "rel_spec_to_true": float(np.linalg.norm(diff, ord=2) / true_spec),
    }


def time_solver(
    name: str,
    solver: Callable[[], Tuple[np.ndarray, Optional[int], float]],
    repeat: int,
    warmup: int,
    true_precision: np.ndarray,
) -> Dict[str, object]:
    for _ in range(warmup):
        solver()

    times = []
    precision = None
    n_iter = None
    gap = None
    for _ in range(repeat):
        start = time.perf_counter()
        precision, n_iter, gap = solver()
        times.append(time.perf_counter() - start)

    stats = precision_stats(precision)
    recovery = true_precision_errors(stats["precision"], true_precision)
    return {
        "name": name,
        "precision": stats["precision"],
        "mean_ms": 1000.0 * float(np.mean(times)),
        "std_ms": 1000.0 * float(np.std(times)),
        "min_ms": 1000.0 * float(np.min(times)),
        "n_iter": n_iter,
        "abs_dual_gap": abs(float(gap)),
        "eig_min": stats["eig_min"],
        "nnz_offdiag": stats["nnz_offdiag"],
        "rel_fro_to_true": recovery["rel_fro_to_true"],
        "rel_spec_to_true": recovery["rel_spec_to_true"],
    }


def add_sklearn_reference_errors(rows: List[Dict[str, object]]) -> None:
    sklearn_precision = rows[0]["precision"]
    for row in rows:
        diff = row["precision"] - sklearn_precision
        row["diff_fro_vs_sklearn"] = float(np.linalg.norm(diff, ord="fro"))
        row["diff_spec_vs_sklearn"] = float(np.linalg.norm(diff, ord=2))


def print_timing_table(rows: List[Dict[str, object]]) -> None:
    print()
    print("Experiment 1: timing and output diagnostics on the first sparse recovery problem")
    headers = [
        "method",
        "mean_ms",
        "std_ms",
        "min_ms",
        "n_iter",
        "abs_dual_gap",
        "min_eig",
        "offdiag_nnz",
        "rel_fro_to_true",
        "rel_spec_to_true",
        "diff_fro_vs_sklearn",
        "diff_spec_vs_sklearn",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(
            f"{row['name']} | "
            f"{row['mean_ms']:.3f} | "
            f"{row['std_ms']:.3f} | "
            f"{row['min_ms']:.3f} | "
            f"{row['n_iter'] if row['n_iter'] is not None else '-'} | "
            f"{format_value(row['abs_dual_gap'])} | "
            f"{format_value(row['eig_min'])} | "
            f"{row['nnz_offdiag']} | "
            f"{format_value(row['rel_fro_to_true'])} | "
            f"{format_value(row['rel_spec_to_true'])} | "
            f"{format_value(row['diff_fro_vs_sklearn'])} | "
            f"{format_value(row['diff_spec_vs_sklearn'])}"
        )


def lambda_matrix(emp_cov: np.ndarray, alpha: float) -> np.ndarray:
    lam = np.full_like(emp_cov, alpha)
    np.fill_diagonal(lam, 0.0)
    return lam


def solve_sklearn(emp_cov: np.ndarray, alpha: float, tol: float, max_iter: int) -> Tuple[np.ndarray, int, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        _, precision, n_iter = sklearn_graphical_lasso(
            emp_cov,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            return_n_iter=True,
        )
    return precision, int(n_iter), dual_gap(emp_cov, precision, alpha)


def solve_quic(emp_cov: np.ndarray, alpha: float, tol: float, max_iter: int) -> Tuple[np.ndarray, int, float]:
    if quic is None:
        raise RuntimeError("inverse_covariance is not installed, so QUIC is unavailable")
    precision, _, _, _, n_iter, _ = quic(
        emp_cov,
        lambda_matrix(emp_cov, alpha),
        tol=tol,
        max_iter=max_iter,
        msg=0,
    )
    return precision, int(n_iter), dual_gap(emp_cov, precision, alpha)


def solve_admm(
    emp_cov: np.ndarray,
    emp_cov_torch: torch.Tensor,
    alpha: float,
    tol: float,
    max_iter: int,
    device: str,
    eps: float,
    eigh_shift: float,
    eigh_shift_retries: int,
    eigh_cpu_fallback: bool,
) -> Tuple[np.ndarray, int, float]:
    result = torch_graphical_lasso(
        emp_cov_torch,
        alpha=alpha,
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
        torch.cuda.synchronize()
    precision = result.precision.detach().cpu().numpy()
    return precision, int(result.n_iter), dual_gap(emp_cov, precision, alpha)


def solver_names() -> List[str]:
    names = ["sklearn_cd"]
    if quic is not None:
        names.append("quic")
    names.append("torch_admm")
    return names


def run_solver(
    name: str,
    emp_cov: np.ndarray,
    emp_cov_torch: torch.Tensor,
    alpha: float,
    tol: float,
    max_iter: int,
    device: str,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, int, float]:
    if name == "sklearn_cd":
        return solve_sklearn(emp_cov, alpha, tol, max_iter)
    if name == "quic":
        return solve_quic(emp_cov, alpha, tol, max_iter)
    if name == "torch_admm":
        return solve_admm(
            emp_cov,
            emp_cov_torch,
            alpha,
            tol,
            max_iter,
            device,
            args.eps,
            args.eigh_shift,
            args.eigh_shift_retries,
            args.eigh_cpu_fallback,
        )
    raise ValueError(f"unknown solver: {name}")


def run_convergence_stats(
    problems: List[SparsePrecisionProblem],
    covariances_torch: List[torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    iter_rows = []
    time_rows = []
    for name in solver_names():
        iter_values = []
        time_values = []
        for problem, emp_cov_torch in zip(problems, covariances_torch):
            start = time.perf_counter()
            _, n_iter, _ = run_solver(
                name,
                problem.emp_cov,
                emp_cov_torch,
                alpha=args.alpha,
                tol=args.tol,
                max_iter=args.max_iter,
                device=args.device,
                args=args,
            )
            time_values.append(1000.0 * (time.perf_counter() - start))
            iter_values.append(float(n_iter))
        iter_rows.append({"name": name, "stats": summarize(iter_values)})
        time_rows.append({"name": name, "stats": summarize(time_values)})
    return iter_rows, time_rows


def run_fixed_iteration_stats(
    problems: List[SparsePrecisionProblem],
    covariances_torch: List[torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    gap_rows = []
    fro_rows = []
    spec_rows = []
    nnz_rows = []
    eig_rows = []

    true_nnz_values = []
    true_eig_values = []
    for problem in problems:
        stats = precision_stats(problem.true_precision)
        true_nnz_values.append(float(stats["nnz_offdiag"]))
        true_eig_values.append(stats["eig_min"])
    nnz_rows.append({"name": "true_precision", "stats": summarize(true_nnz_values)})
    eig_rows.append({"name": "true_precision", "stats": summarize(true_eig_values)})

    for name in solver_names():
        gap_values = []
        fro_values = []
        spec_values = []
        nnz_values = []
        eig_values = []
        for problem, emp_cov_torch in zip(problems, covariances_torch):
            precision, _, gap = run_solver(
                name,
                problem.emp_cov,
                emp_cov_torch,
                alpha=args.alpha,
                tol=args.fixed_tol,
                max_iter=args.fixed_iter,
                device=args.device,
                args=args,
            )
            stats = precision_stats(precision)
            errors = true_precision_errors(precision, problem.true_precision)
            gap_values.append(abs(float(gap)))
            fro_values.append(errors["rel_fro_to_true"])
            spec_values.append(errors["rel_spec_to_true"])
            nnz_values.append(float(stats["nnz_offdiag"]))
            eig_values.append(stats["eig_min"])
        gap_rows.append({"name": name, "stats": summarize(gap_values)})
        fro_rows.append({"name": name, "stats": summarize(fro_values)})
        spec_rows.append({"name": name, "stats": summarize(spec_values)})
        nnz_rows.append({"name": name, "stats": summarize(nnz_values)})
        eig_rows.append({"name": name, "stats": summarize(eig_values)})
    return gap_rows, fro_rows, spec_rows, nnz_rows, eig_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sklearn GraphicalLasso, QUIC, and PyTorch ADMM on "
            "simulated empirical covariances."
        )
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--fixed-iter", type=int, default=20)
    parser.add_argument("--fixed-tol", type=float, default=1e-20)
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-prob", type=float, default=0.02)
    parser.add_argument("--edge-weight-min", type=float, default=0.15)
    parser.add_argument("--edge-weight-max", type=float, default=0.35)
    parser.add_argument("--diagonal-shift", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--eps", type=float, default=0.0, help="covariance jitter; changes the GLASSO problem")
    parser.add_argument("--eigh-shift", type=float, default=1e-6, help="eigensolver-only shift; recovered after eigh")
    parser.add_argument("--eigh-shift-retries", type=int, default=4)
    parser.add_argument("--no-eigh-cpu-fallback", dest="eigh_cpu_fallback", action="store_false")
    parser.set_defaults(eigh_cpu_fallback=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if args.fixed_tol <= 0:
        raise ValueError("fixed_tol must be positive because sklearn requires tol > 0")

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
        f"n_features={args.n_features}, alpha={args.alpha}, tol={args.tol}, "
        f"max_iter={args.max_iter}, fixed_iter={args.fixed_iter}, "
        f"fixed_tol={args.fixed_tol}, edge_prob={args.edge_prob}, "
        f"repeat={args.repeat}, warmup={args.warmup}, "
        f"dtype={args.dtype}, torch_device={args.device}, "
        f"eps={args.eps}, eigh_shift={args.eigh_shift}, "
        f"eigh_shift_retries={args.eigh_shift_retries}, "
        f"eigh_cpu_fallback={args.eigh_cpu_fallback}"
    )
    if quic is None:
        print("QUIC skipped: inverse_covariance is not installed.")

    first_problem = problems[0]
    first_cov = first_problem.emp_cov
    first_cov_torch = covariances_torch[0]
    timing_rows = []
    for name in solver_names():
        timing_rows.append(
            time_solver(
                name,
                lambda name=name: run_solver(
                    name,
                    first_cov,
                    first_cov_torch,
                    alpha=args.alpha,
                    tol=args.tol,
                    max_iter=args.max_iter,
                    device=args.device,
                    args=args,
                ),
                args.repeat,
                args.warmup,
                first_problem.true_precision,
            )
        )
    add_sklearn_reference_errors(timing_rows)
    print_timing_table(timing_rows)

    convergence_iter_rows, convergence_time_rows = run_convergence_stats(problems, covariances_torch, args)
    print_stats_table(
        f"Experiment 2a: iterations to tol={args.tol} over {args.num_problems} sparse recovery problems",
        convergence_iter_rows,
        "n_iter",
    )
    print_stats_table(
        f"Experiment 2b: runtime to tol={args.tol} over {args.num_problems} sparse recovery problems",
        convergence_time_rows,
        "time_ms",
    )

    fixed_gap_rows, fixed_fro_rows, fixed_spec_rows, fixed_nnz_rows, fixed_eig_rows = run_fixed_iteration_stats(
        problems,
        covariances_torch,
        args,
    )
    print_stats_table(
        f"Experiment 3a: absolute dual gap after fixed_iter={args.fixed_iter}",
        fixed_gap_rows,
        "abs_gap",
    )

    print_stats_table(
        f"Experiment 3b: relative Frobenius error to true precision after fixed_iter={args.fixed_iter}",
        fixed_fro_rows,
        "rel_fro",
    )
    print_stats_table(
        f"Experiment 3c: relative spectral error to true precision after fixed_iter={args.fixed_iter}",
        fixed_spec_rows,
        "rel_spec",
    )
    print_stats_table(
        f"Experiment 3d: off-diagonal nonzero entries after fixed_iter={args.fixed_iter}",
        fixed_nnz_rows,
        "nnz_offdiag",
    )
    print_stats_table(
        f"Experiment 3e: minimum eigenvalue after fixed_iter={args.fixed_iter}",
        fixed_eig_rows,
        "min_eig",
    )


if __name__ == "__main__":
    main()
