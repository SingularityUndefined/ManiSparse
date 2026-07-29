import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import torch


def make_repeated_spectrum_matrix(n_features, repeats, seed, dtype, device):
    """Create a symmetric matrix with many repeated / clustered eigenvalues."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q_raw = torch.randn(n_features, n_features, generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(q_raw)
    base = torch.linspace(-2.0, 2.0, steps=max(1, n_features // repeats), dtype=dtype)
    eigvals = base.repeat_interleave(repeats)[:n_features]
    if eigvals.numel() < n_features:
        eigvals = torch.cat([eigvals, eigvals.new_zeros(n_features - eigvals.numel())])
    matrix = (q * eigvals.unsqueeze(0)) @ q.T
    matrix = 0.5 * (matrix + matrix.T)
    return matrix.to(device=device), eigvals.to(device=device)


def theta_from_eigh(eigvals, eigvecs, rho):
    theta_eigvals = (eigvals + torch.sqrt(eigvals.square() + 4.0 * rho)) / (2.0 * rho)
    theta = (eigvecs * theta_eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
    return 0.5 * (theta + theta.transpose(-1, -2))


def relative_fro(a, b):
    return (torch.linalg.matrix_norm(a - b) / torch.linalg.matrix_norm(b)).item()


def run_case(matrix, rho, shift):
    """Compare exact ADMM Theta update with shift trick and S-jitter behavior."""
    start = time.perf_counter()
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    direct_time = time.perf_counter() - start
    theta_direct = theta_from_eigh(eigvals, eigvecs, rho)

    eye = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)

    start = time.perf_counter()
    shifted_eigvals, shifted_eigvecs = torch.linalg.eigh(matrix + shift * eye)
    shifted_time = time.perf_counter() - start
    theta_shift = theta_from_eigh(shifted_eigvals - shift, shifted_eigvecs, rho)

    start = time.perf_counter()
    jitter_eigvals, jitter_eigvecs = torch.linalg.eigh(matrix - shift * eye)
    jitter_time = time.perf_counter() - start
    theta_cov_jitter = theta_from_eigh(jitter_eigvals, jitter_eigvecs, rho)

    return {
        "shift": shift,
        "shift_rel_fro": relative_fro(theta_shift, theta_direct),
        "cov_jitter_rel_fro": relative_fro(theta_cov_jitter, theta_direct),
        "direct_time_ms": 1000.0 * direct_time,
        "shift_time_ms": 1000.0 * shifted_time,
        "cov_jitter_time_ms": 1000.0 * jitter_time,
    }


def print_table(rows):
    print("shift | shift_rel_fro | cov_jitter_rel_fro | direct_time_ms | shift_time_ms | cov_jitter_time_ms")
    print("---: | ---: | ---: | ---: | ---: | ---:")
    for row in rows:
        print(
            f"{row['shift']:.1e} | "
            f"{row['shift_rel_fro']:.6e} | "
            f"{row['cov_jitter_rel_fro']:.6e} | "
            f"{row['direct_time_ms']:.3f} | "
            f"{row['shift_time_ms']:.3f} | "
            f"{row['cov_jitter_time_ms']:.3f}"
        )


def plot_rows(rows, output):
    shifts = [row["shift"] for row in rows]
    shift_errors = [row["shift_rel_fro"] for row in rows]
    jitter_errors = [row["cov_jitter_rel_fro"] for row in rows]

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.loglog(shifts, shift_errors, marker="o", linewidth=2.0, label="shift trick, recover eigenvalues")
    ax.loglog(shifts, jitter_errors, marker="s", linewidth=2.0, label="covariance jitter, changes S")
    ax.set_xlabel("delta")
    ax.set_ylabel("relative Frobenius error vs direct Theta update")
    ax.set_title("ADMM GLASSO eigensolver shift preserves the original update")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ADMM GLASSO eigensolver shift trick.")
    parser.add_argument("--n-features", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--shifts", type=str, default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3")
    parser.add_argument("--output", type=Path, default=Path("glasso_pytorch/figures/eigh_shift_trick.png"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    shifts = [float(item.strip()) for item in args.shifts.split(",") if item.strip()]
    matrix, eigvals = make_repeated_spectrum_matrix(args.n_features, args.repeats, args.seed, dtype, args.device)

    print(
        "Repeated-spectrum eigensolver benchmark: "
        f"n_features={args.n_features}, repeats={args.repeats}, rho={args.rho}, "
        f"dtype={args.dtype}, device={args.device}, "
        f"unique_eigenvalues={torch.unique(eigvals.detach().cpu()).numel()}"
    )
    rows = [run_case(matrix, args.rho, shift) for shift in shifts]
    print_table(rows)
    plot_rows(rows, args.output)


if __name__ == "__main__":
    main()
