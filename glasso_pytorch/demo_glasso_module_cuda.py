import argparse
import gc
import time
from dataclasses import dataclass
from typing import List

import torch

from glasso_pytorch import GraphicalLassoModule


@dataclass
class RunStats:
    elapsed_ms: float
    dual_gap: float
    baseline_allocated_mb: float
    peak_allocated_mb: float
    peak_delta_allocated_mb: float
    peak_reserved_mb: float


def bytes_to_mb(value: int) -> float:
    return value / 1024.0 / 1024.0


def make_spd_covariance(
    batch_size: int,
    n_features: int,
    device: torch.device,
    dtype: torch.dtype,
    diagonal_jitter: float,
) -> torch.Tensor:
    with torch.no_grad():
        x = torch.randn(batch_size, n_features, n_features, device=device, dtype=dtype)
        cov = x @ x.transpose(-1, -2)
        cov = cov / n_features
        eye = torch.eye(n_features, device=device, dtype=dtype).expand_as(cov)
        return cov + diagonal_jitter * eye


def measure_once(
    module: GraphicalLassoModule,
    cov: torch.Tensor,
    device: torch.device,
    allow_backward: bool,
) -> RunStats:
    if allow_backward:
        cov = cov.detach().requires_grad_(True)
    else:
        cov = cov.detach()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)

    start = time.perf_counter()
    result = module.solve(cov, return_info=True)
    theta = result.precision
    if allow_backward:
        loss = theta.square().mean()
        loss.backward()
    torch.cuda.synchronize(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    stats = RunStats(
        elapsed_ms=elapsed_ms,
        dual_gap=result.dual_gap,
        baseline_allocated_mb=bytes_to_mb(baseline_allocated),
        peak_allocated_mb=bytes_to_mb(peak_allocated),
        peak_delta_allocated_mb=bytes_to_mb(peak_allocated - baseline_allocated),
        peak_reserved_mb=bytes_to_mb(peak_reserved),
    )

    del theta
    del result
    if allow_backward:
        del loss
    del cov
    return stats


def average_stats(stats: List[RunStats]) -> RunStats:
    n = len(stats)
    return RunStats(
        elapsed_ms=sum(item.elapsed_ms for item in stats) / n,
        dual_gap=sum(item.dual_gap for item in stats) / n,
        baseline_allocated_mb=sum(item.baseline_allocated_mb for item in stats) / n,
        peak_allocated_mb=sum(item.peak_allocated_mb for item in stats) / n,
        peak_delta_allocated_mb=sum(item.peak_delta_allocated_mb for item in stats) / n,
        peak_reserved_mb=sum(item.peak_reserved_mb for item in stats) / n,
    )


def print_summary(name: str, stats: RunStats) -> None:
    print(
        f"{name} | "
        f"{stats.elapsed_ms:.3f} | "
        f"{stats.dual_gap:.6g} | "
        f"{stats.baseline_allocated_mb:.3f} | "
        f"{stats.peak_allocated_mb:.3f} | "
        f"{stats.peak_delta_allocated_mb:.3f} | "
        f"{stats.peak_reserved_mb:.3f}"
    )


def run_mode(
    name: str,
    allow_backward: bool,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> RunStats:
    module = GraphicalLassoModule(
        alpha=args.alpha,
        rho=args.rho,
        max_iter=args.max_iter,
        tol=args.tol,
        rtol=args.rtol,
        eps=args.eps,
        allow_backward=allow_backward,
        input_mode="covariance",
    ).to(device)

    stats = []
    for _ in range(args.num_batches):
        cov = make_spd_covariance(
            batch_size=args.batch_size,
            n_features=args.n_features,
            device=device,
            dtype=dtype,
            diagonal_jitter=args.diagonal_jitter,
        )
        stats.append(measure_once(module, cov, device, allow_backward=allow_backward))
        del cov
        gc.collect()
        torch.cuda.empty_cache()

    return average_stats(stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure GraphicalLassoModule CUDA time and memory for pure forward "
            "versus forward+backward on batched covariance matrices."
        )
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--diagonal-jitter", type=float, default=1e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    print(
        "Config: "
        f"device={device}, batch_size={args.batch_size}, n_features={args.n_features}, "
        f"num_batches={args.num_batches}, dtype={args.dtype}, alpha={args.alpha}, "
        f"max_iter={args.max_iter}, tol={args.tol}, rtol={args.rtol}"
    )
    print("Timing excludes synthetic covariance generation.")
    print(
        "mode | avg_time_ms | avg_dual_gap | avg_baseline_allocated_mb | "
        "avg_peak_allocated_mb | avg_peak_delta_allocated_mb | avg_peak_reserved_mb"
    )
    print("--- | --- | --- | --- | --- | --- | ---")

    forward_stats = run_mode("forward_only", False, args, device, dtype)
    print_summary("forward_only", forward_stats)

    backward_stats = run_mode("forward_backward", True, args, device, dtype)
    print_summary("forward_backward", backward_stats)


if __name__ == "__main__":
    main()
