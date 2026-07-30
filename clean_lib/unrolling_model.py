import torch
import torch.nn as nn
import warnings
from torch.nn.parameter import Parameter

from clean_lib.admm_block import ADMMBlock
from clean_lib.backup_modules import (
    LR_guess,
    SpatialTemporalEmbedding,
    connect_list,
    find_k_nearest_neighbors,
    layer_norm_on_data,
    layer_recovery_on_data,
)
from clean_lib.feature_extractor import FeatureExtractor, GNNExtrapolation, GraphSAGEExtrapolation
from clean_lib.graph_learning_module import GraphLearningModule
from kalofolias_graph_learning import KalofoliasGraphLearningModule
from local_kalofolias import LocalKalofoliasGraphLearning


def _single_glasso_estimation(
    cov_matrix,
    alpha,
    method,
    rho,
    eps,
    eigh_shift,
    eigh_shift_retries,
    fallback,
    max_iter,
    tol,
    allow_backward,
    device,
    dtype,
):
    """Estimate one dense precision matrix from one covariance matrix."""
    events = []
    solver_cov = cov_matrix if allow_backward else cov_matrix.detach()
    solver_cov = 0.5 * (solver_cov + solver_cov.transpose(-1, -2))
    if not torch.isfinite(solver_cov).all():
        raise ValueError(f"glasso covariance contains NaN or Inf: shape={tuple(solver_cov.shape)}")

    if method == "admm":
        try:
            from glasso_pytorch import graphical_lasso

            # Keep the normal ADMM path in the covariance dtype used by the
            # model. The lower-level eigensolver fallback will try CPU float64
            # only if the original dtype/device eigensolve fails.
            solver_cov_torch = solver_cov
            last_error = None
            try:
                if allow_backward:
                    result = graphical_lasso(
                        solver_cov_torch,
                        alpha=alpha,
                        rho=rho,
                        max_iter=max_iter,
                        tol=tol,
                        rtol=tol,
                        eps=eps,
                        eigh_shift=eigh_shift,
                        eigh_shift_retries=eigh_shift_retries,
                        return_info=True,
                    )
                else:
                    with torch.no_grad():
                        result = graphical_lasso(
                            solver_cov_torch,
                            alpha=alpha,
                            rho=rho,
                            max_iter=max_iter,
                            tol=tol,
                            rtol=tol,
                            eps=eps,
                            eigh_shift=eigh_shift,
                            eigh_shift_retries=eigh_shift_retries,
                            return_info=True,
                        )
                for event in getattr(result, "eigh_events", []):
                    events.append(
                        {
                            "stage": "admm_eigh_fallback",
                            "solver": "torch_admm",
                            **event,
                        }
                    )
                return result.precision.to(device=device, dtype=dtype), events
            except RuntimeError as error:
                last_error = error

            if not fallback:
                raise RuntimeError(
                    "glasso_pytorch ADMM failed after spectrum-preserving "
                    f"eigh shift retries up to {eigh_shift * (10.0**eigh_shift_retries):.3e}"
                ) from last_error
            warnings.warn(
                "glasso_pytorch ADMM failed; falling back to sklearn/quic/ridge precision. "
                f"Last error: {last_error}",
                RuntimeWarning,
            )
            events.append(
                {
                    "stage": "solver_fallback",
                    "from": "torch_admm",
                    "to": "sklearn",
                    "reason": str(last_error),
                }
            )
            method = "sklearn"
        except ImportError:
            events.append(
                {
                    "stage": "solver_fallback",
                    "from": "torch_admm",
                    "to": "sklearn",
                    "reason": "glasso_pytorch import failed",
                }
            )
            method = "sklearn"

    if method == "quic":
        try:
            import numpy as np
            from inverse_covariance import quic

            cov_np = solver_cov.detach().cpu().numpy().astype(np.float64, copy=False)
            lam = np.full_like(cov_np, alpha, dtype=np.float64)
            np.fill_diagonal(lam, 0.0)
            precision, _, _, _, _, _ = quic(cov_np, lam, tol=tol, max_iter=max_iter)
            return torch.tensor(precision, dtype=dtype, device=device), events
        except ImportError:
            events.append(
                {
                    "stage": "solver_fallback",
                    "from": "quic",
                    "to": "sklearn",
                    "reason": "inverse_covariance import failed",
                }
            )
            method = "sklearn"
        except Exception as error:
            if not fallback:
                raise
            warnings.warn(f"QUIC graphical lasso failed; falling back to sklearn/ridge precision. Error: {error}", RuntimeWarning)
            events.append(
                {
                    "stage": "solver_fallback",
                    "from": "quic",
                    "to": "sklearn",
                    "reason": str(error),
                }
            )
            method = "sklearn"

    if method == "sklearn":
        try:
            from sklearn.covariance import GraphicalLasso

            model = GraphicalLasso(alpha=alpha, max_iter=max_iter, tol=tol)
            model.fit(solver_cov.detach().cpu().numpy())
            return torch.tensor(model.precision_, dtype=dtype, device=device), events
        except Exception as error:
            if not fallback:
                raise
            warnings.warn(f"sklearn GraphicalLasso failed; using ridge/pinv precision fallback. Error: {error}", RuntimeWarning)
            events.append(
                {
                    "stage": "solver_fallback",
                    "from": "sklearn",
                    "to": "ridge_pinv",
                    "reason": str(error),
                }
            )
            return _ridge_precision_fallback(solver_cov, alpha, device, dtype), events

    raise ValueError(f"Unknown graphical lasso method: {method}")


def _add_diagonal_shift(cov_matrix, shift):
    """Add a diagonal shift for the last-resort ridge inverse fallback."""
    eye = torch.eye(cov_matrix.size(-1), device=cov_matrix.device, dtype=cov_matrix.dtype)
    return cov_matrix + shift * eye


def _ridge_precision_fallback(cov_matrix, alpha, device, dtype):
    """Last-resort finite precision estimate when graphical lasso solvers fail."""
    solver_cov = cov_matrix.detach().to(torch.float64)
    ridge = max(float(alpha), torch.finfo(solver_cov.dtype).eps)
    regularized_cov = _add_diagonal_shift(solver_cov, ridge)
    try:
        precision = torch.linalg.inv(regularized_cov)
    except RuntimeError:
        precision = torch.linalg.pinv(regularized_cov)
    precision = 0.5 * (precision + precision.transpose(-1, -2))
    return precision.to(device=device, dtype=dtype)


def glasso_estimation(
    cov_matrix,
    alpha=0.2,
    method="admm",
    rho=1.0,
    eps=0.0,
    eigh_shift=1e-6,
    eigh_shift_retries=4,
    fallback=True,
    max_iter=20,
    tol=1e-4,
    allow_backward=False,
    return_events=False,
):
    """
    Estimate the precision matrix (inverse covariance) using Graphical Lasso.
    Args:
        cov_matrix: Dense node covariance matrix, shape (N, N), or batched
            covariance matrices, shape (B, N, N).
        alpha: Regularization parameter for Graphical Lasso.
        method: One of "admm", "quic", or "sklearn".
        rho: ADMM penalty parameter used only when method is "admm".
        eps: Optional covariance diagonal jitter. This perturbs the GLASSO
            problem and defaults to 0.
        eigh_shift: Initial eigensolver-only spectral shift for the ADMM Theta
            update. The shift is subtracted from recovered eigenvalues, so this
            does not perturb the GLASSO objective in exact arithmetic.
        eigh_shift_retries: Number of extra 10x shift retries.
        fallback: If True, fall back to sklearn/quic/ridge precision when the
            selected solver fails.
        max_iter: Maximum solver iterations. Capped at 20 for this model.
        tol: Solver tolerance.
        allow_backward: If False, Theta estimation is detached from autograd.
        return_events: If True, also return fallback/eigensolver diagnostic
            events produced while estimating Theta.
    Returns:
        Dense node precision matrix Theta, shape (N, N), or batched precision
        matrices, shape (B, N, N).
    """
    device = cov_matrix.device
    dtype = cov_matrix.dtype
    max_iter = min(max_iter, 20)
    method = method.lower()
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.transpose(-1, -2))

    if cov_matrix.ndim == 2:
        theta, events = _single_glasso_estimation(
            cov_matrix,
            alpha,
            method,
            rho,
            eps,
            eigh_shift,
            eigh_shift_retries,
            fallback,
            max_iter,
            tol,
            allow_backward,
            device,
            dtype,
        )
        theta = normalize_theta(theta)
        return (theta, events) if return_events else theta
    if cov_matrix.ndim == 3:
        theta_list = []
        events = []
        for i in range(cov_matrix.size(0)):
            theta_i, events_i = _single_glasso_estimation(
                cov_matrix[i],
                alpha,
                method,
                rho,
                eps,
                eigh_shift,
                eigh_shift_retries,
                fallback,
                max_iter,
                tol,
                allow_backward,
                device,
                dtype,
            )
            theta_list.append(theta_i)
            for event in events_i:
                event = dict(event)
                event["cov_batch_index"] = i
                events.append(event)
        theta = normalize_theta(torch.stack(theta_list, dim=0))
        return (theta, events) if return_events else theta
    raise ValueError("cov_matrix must have shape (N, N) or (B, N, N)")


def normalize_theta(theta):
    """Normalize Theta by its diagonal scale.

    The normalization is:
        Theta_ij = Theta_ij / sqrt(Theta_ii * Theta_jj)

    Any negative diagonal entry is first clamped to zero. Entries whose
    denominator is zero are set to zero to avoid NaN/Inf values.
    """
    if theta.ndim not in (2, 3):
        raise ValueError("theta must have shape (N, N) or (B, N, N)")

    theta = theta.clone()
    diag = torch.diagonal(theta, dim1=-2, dim2=-1).clamp_min(0)
    node_idx = torch.arange(theta.size(-1), device=theta.device)

    if theta.ndim == 2:
        theta[node_idx, node_idx] = diag
        denom = torch.sqrt(diag.unsqueeze(0) * diag.unsqueeze(1))
    else:
        theta[:, node_idx, node_idx] = diag
        denom = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2))

    eps = torch.finfo(theta.dtype).eps
    normalized = theta / denom.clamp_min(eps)
    return torch.where(denom > 0, normalized, torch.zeros_like(theta))


def node_covariance(signal, unbiased=True):
    """Compute centered node covariance for Theta estimation.

    Args:
        signal: either `(B, T, N, C)` for one shared covariance matrix, or
            `(B, K, T, N, C)` for one covariance matrix per batch item.
        unbiased: when True, divide by the number of samples minus one.

    Returns:
        `(N, N)` for a 4-D input, or `(B, N, N)` for a 5-D input.
    """
    if signal.ndim == 4:
        n_nodes = signal.size(2)
        samples = signal.permute(0, 1, 3, 2).reshape(-1, n_nodes)
        samples = samples - samples.mean(dim=0, keepdim=True)
        denom = samples.size(0) - 1 if unbiased else samples.size(0)
        denom = max(denom, 1)
        return torch.einsum("sn,sm->nm", samples, samples) / denom

    if signal.ndim == 5:
        batch_size, _, _, n_nodes, _ = signal.shape
        samples = signal.permute(0, 1, 2, 4, 3).reshape(batch_size, -1, n_nodes)
        samples = samples - samples.mean(dim=1, keepdim=True)
        denom = samples.size(1) - 1 if unbiased else samples.size(1)
        denom = max(denom, 1)
        return torch.einsum("bsn,bsm->bnm", samples, samples) / denom

    raise ValueError("signal must have shape (B, T, N, C) or (B, K, T, N, C)")


def node_signal_matrix(signal):
    """Reshape model signals into Kalofolias node-signal matrices.

    Args:
        signal: `(B, T, N, C)` for one shared graph estimate, or
            `(B, K, T, N, C)` for one graph estimate per batch item.

    Returns:
        `(N, B*T*C)` for a shared graph, or `(B, N, K*T*C)` for batched graphs.
    """
    if signal.ndim == 4:
        return signal.permute(2, 0, 1, 3).reshape(signal.size(2), -1)
    if signal.ndim == 5:
        batch_size, _, _, n_nodes, _ = signal.shape
        return signal.permute(0, 3, 1, 2, 4).reshape(batch_size, n_nodes, -1)
    raise ValueError("signal must have shape (B, T, N, C) or (B, K, T, N, C)")

DEFAULT_GRAPH_INFO = {
    "n_nodes": None,
    "u_edges": None,
    "u_dist": None,
}

DEFAULT_ADMM_INFO = {
    "ADMM_iters": 30,
    "CG_iters": 3,
    "mu_u_init": 3,
    "mu_d1_init": 3,
    "mu_d2_init": 3,
}

DEFAULT_ST_EMB_INFO = {
    "spatial_dim": 5,
    "t_dim": 10,
    "tid_dim": 6,
    "diw_dim": 4,
}


class UnrollingModel(nn.Module):
    """Stacked unrolling model.

    High-level data flow:
        observed y -> extrapolated output -> optional ST embedding
        -> feature extractor -> graph learning -> ADMM block
        -> skip connection -> next block

    Main external tensor convention:
        y:      (B, t_in, N, C_signal)
        t_list: (B, T), integer time indices used by ST embedding
        output: (B, T, N, C_signal) or (B, T, N, 1) when use_one_channel=True
    """

    def __init__(
        self,
        num_blocks,
        device,
        T,
        t_in,
        n_heads,
        interval,
        signal_channels,
        feature_channels,
        k_hop,
        theta_k_hop=None,
        GNN_alpha=0.2,
        graph_info=None,
        ADMM_info=None,
        use_norm=False,
        GNN_layers=2,
        use_st_emb=True,
        st_emb_info=None,
        use_extrapolation=True,
        use_old_extrapolation=False,
        extrapolation_agg_layers=1,
        sigma_ratio=450,
        ablation="None",
        use_one_channel=False,
        sharedM=False,
        sharedQ=True,
        diff_interval=True,
        use_stable_graph_learning=False,
        predict_only=False,
        le_emb=False,
        theta_method="glasso",
        glasso_backend="admm",
        glasso_alpha=0.2,
        glasso_rho=1.0,
        glasso_eps=0.0,
        glasso_eigh_shift=1e-6,
        glasso_eigh_shift_retries=4,
        glasso_fallback=True,
        glasso_max_iter=20,
        glasso_tol=1e-4,
        glasso_allow_backward=False,
        kalofolias_alpha=0.3,
        kalofolias_beta=1.0,
        kalofolias_graph="dense",
        kalofolias_max_iter=200,
        kalofolias_tol=1e-4,
        kalofolias_threshold=1e-4,
        kalofolias_output_mode="laplacian",
        kalofolias_normalize_distances=True,
        use_deflation=True,
        deflation_samples=None,
    ):
        super().__init__()
        graph_info = DEFAULT_GRAPH_INFO if graph_info is None else graph_info
        ADMM_info = DEFAULT_ADMM_INFO if ADMM_info is None else ADMM_info
        st_emb_info = DEFAULT_ST_EMB_INFO if st_emb_info is None else st_emb_info

        self.num_blocks = num_blocks
        self.device = device
        self.T = T
        self.t_in = t_in
        self.n_heads = n_heads
        self.use_norm = use_norm
        self.ablation = ablation
        self.use_one_channel = use_one_channel
        self.predict_only = predict_only
        self.use_extrapolation = use_extrapolation
        self.use_st_emb = use_st_emb
        self.use_stable_graph_learning = use_stable_graph_learning
        self.theta_method = theta_method.lower()
        self.glasso_backend = glasso_backend.lower()
        self.kalofolias_graph = kalofolias_graph.lower()
        if self.theta_method not in {"glasso", "kalofolias"}:
            raise ValueError("theta_method must be one of: glasso, kalofolias")
        if self.glasso_backend not in {"admm", "quic", "sklearn"}:
            raise ValueError("glasso_backend must be one of: admm, quic, sklearn")
        if self.kalofolias_graph not in {"dense", "local"}:
            raise ValueError("kalofolias_graph must be one of: dense, local")
        self.theta_k_hop = k_hop if theta_k_hop is None else theta_k_hop
        self.glasso_alpha = glasso_alpha
        self.glasso_rho = glasso_rho
        self.glasso_eps = glasso_eps
        self.glasso_eigh_shift = glasso_eigh_shift
        self.glasso_eigh_shift_retries = glasso_eigh_shift_retries
        self.glasso_fallback = glasso_fallback
        self.glasso_max_iter = min(glasso_max_iter, 20)
        self.glasso_tol = glasso_tol
        self.glasso_allow_backward = glasso_allow_backward
        self.kalofolias_alpha = kalofolias_alpha
        self.kalofolias_beta = kalofolias_beta
        self.kalofolias_max_iter = kalofolias_max_iter
        self.kalofolias_tol = kalofolias_tol
        self.kalofolias_threshold = kalofolias_threshold
        self.kalofolias_output_mode = kalofolias_output_mode
        self.kalofolias_normalize_distances = kalofolias_normalize_distances
        self.kalofolias_graph_estimator = KalofoliasGraphLearningModule(
            alpha=kalofolias_alpha,
            beta=kalofolias_beta,
            max_iter=kalofolias_max_iter,
            tol=kalofolias_tol,
            threshold=kalofolias_threshold,
            output_mode=kalofolias_output_mode,
            normalize_distances=kalofolias_normalize_distances,
            allow_backward=glasso_allow_backward,
        )
        self.use_deflation = use_deflation
        self.deflation_samples = deflation_samples
        self.last_deflation_multi_x = None
        self.last_glasso_events = []
        self.debug_numerics = False
        self.debug_context = ""
        self.debug_records = []

        # Graph metadata:
        #   u_edges: physical/direct graph edges, shape (E, 2)
        #   u_dist: edge distances, shape (E,)
        #   nearest_nodes: k-hop nearest list, shape (N, k_hop + 1). Column 0
        #       is the node itself; later columns are neighbors or -1 padding.
        #   nearest_dists: same shape as nearest_nodes.
        self.nearest_nodes, self.nearest_dists = find_k_nearest_neighbors(
            graph_info["n_nodes"],
            graph_info["u_edges"],
            graph_info["u_dist"],
            k_hop,
            device=self.device,
        )
        self.theta_nearest_nodes, self.theta_nearest_dists = find_k_nearest_neighbors(
            graph_info["n_nodes"],
            graph_info["u_edges"],
            graph_info["u_dist"],
            self.theta_k_hop,
            device=self.device,
        )
        self.theta_neighbor_list = self._build_theta_neighbor_list(k_hop)
        self.local_kalofolias_graph_estimator = LocalKalofoliasGraphLearning(
            self.theta_neighbor_list,
            alpha=kalofolias_alpha,
            beta=kalofolias_beta,
            max_iter=kalofolias_max_iter,
            tol=kalofolias_tol,
            threshold=kalofolias_threshold,
            normalize_distances=kalofolias_normalize_distances,
            allow_backward=glasso_allow_backward,
        )
        self.connect_list = connect_list(graph_info["n_nodes"], graph_info["u_edges"], self.device)

        if self.use_extrapolation:
            self.use_old_extrapolation = use_old_extrapolation
            if self.use_old_extrapolation:
                # Input: (B, t_in, N, C_signal); output: (B, T, N, C_signal).
                self.linear_extrapolation = GNNExtrapolation(
                    graph_info["n_nodes"],
                    t_in,
                    T,
                    self.nearest_nodes,
                    self.nearest_dists,
                    n_heads,
                    self.device,
                    sigma_ratio=sigma_ratio,
                )
            else:
                # GraphSAGE extrapolates missing future steps before ADMM starts.
                # Input/output shapes match GNNExtrapolation above.
                self.linear_extrapolation = GraphSAGEExtrapolation(
                    graph_info["n_nodes"],
                    t_in,
                    T,
                    self.nearest_nodes,
                    signal_channels,
                    n_heads,
                    device,
                    interval=interval,
                    n_layers=extrapolation_agg_layers,
                )

        if self.use_st_emb:
            # st_emb(t_list) returns (B, T, N, C_st), where
            # C_st = spatial_dim + t_dim + tid_dim + diw_dim.
            self.st_emb = SpatialTemporalEmbedding(
                graph_info["n_nodes"],
                graph_info["u_edges"],
                graph_info["u_dist"],
                sigma_ratio,
                self.device,
                st_emb_info["spatial_dim"],
                st_emb_info["t_dim"],
                st_emb_info["tid_dim"],
                st_emb_info["diw_dim"],
                learnable=le_emb,
            )
            signal_emb_channels = (
                signal_channels
                + st_emb_info["spatial_dim"]
                + st_emb_info["t_dim"]
                + st_emb_info["tid_dim"]
                + st_emb_info["diw_dim"]
            )
        else:
            signal_emb_channels = signal_channels

        if self.use_one_channel:
            # Only the first signal channel is reconstructed by ADMM. The
            # embedding channels are still appended to the single reconstructed
            # signal channel before feature extraction in later blocks.
            signal_rec_channels = 1
            signal_rec_emb_channels = signal_emb_channels - signal_channels + 1
        else:
            signal_rec_channels = signal_channels
            signal_rec_emb_channels = signal_emb_channels

        self.model_blocks = nn.ModuleList([])
        self.skip_connection_weights = Parameter(
            torch.ones((num_blocks,), device=self.device) * 0.95,
            requires_grad=True,
        )

        directed_time_graph = self.ablation != "UT"
        for i in range(self.num_blocks):
            block_input_channels = signal_emb_channels if i == 0 else signal_rec_emb_channels
            self.model_blocks.append(
                nn.ModuleDict(
                    {
                        # FeatureExtractor:
                        #   input:  (B, T, N, block_input_channels)
                        #   output: (B, T, N, H, feature_channels)
                        "feature_extractor": FeatureExtractor(
                            in_features=block_input_channels,
                            out_features=feature_channels,
                            nearest_nodes=self.nearest_nodes,
                            n_heads=n_heads,
                            device=device,
                            interval=interval,
                            n_layers=extrapolation_agg_layers,
                        ),
                        # ADMMBlock:
                        #   input:  (B, T, N, signal_rec_channels)
                        #   output: (B, T, N, signal_rec_channels)
                        # Its graph weights are assigned just before forward.
                        # This constructor is inside the layer loop, so every
                        # unrolled layer owns an independent ADMMBlock and an
                        # independent set of UnrolledCGSolver modules/parameters:
                        #   x_solver, zu_solver, zd_solver
                        "ADMM_block": ADMMBlock(
                            T=T,
                            n_nodes=graph_info["n_nodes"],
                            n_heads=n_heads,
                            n_channels=signal_rec_channels,
                            interval=interval,
                            connect_list=self.connect_list,
                            nearest_nodes=self.nearest_nodes,
                            device=device,
                            ADMM_info=ADMM_info,
                            ablation=self.ablation,
                        ),
                        # GraphLearningModule consumes features and returns:
                        #   u_ew: (B, T, N, K, H)
                        #   d_ew: (B, T - 1, interval, N, H)
                        "graph_learning_module": GraphLearningModule(
                            T=T,
                            n_nodes=graph_info["n_nodes"],
                            connect_list=self.connect_list,
                            nearest_nodes=self.nearest_nodes,
                            n_heads=n_heads,
                            interval=interval,
                            device=device,
                            n_channels=feature_channels,
                            sharedM=sharedM,
                            sharedQ=sharedQ,
                            diff_interval=diff_interval,
                            directed_time=directed_time_graph,
                            use_stable_graph_learning=use_stable_graph_learning,
                        ),
                    }
                )
            )

        self.y_norm_shape = [self.t_in, graph_info["n_nodes"], signal_channels]
        self.norm_shape = [self.T, graph_info["n_nodes"], signal_channels]

    def _build_theta_neighbor_list(self, graph_k_hop):
        """Return the Theta-only local candidate list, excluding the self column.

        `nearest_nodes` is built with column 0 equal to the node itself.  Local
        Kalofolias estimates only off-diagonal candidate edge weights, so ADMM
        receives `theta_neighbor_list` with shape `(N, theta.local_kNN)`.
        """
        uses_local_kalofolias = self._uses_local_kalofolias()
        if uses_local_kalofolias and self.theta_k_hop <= graph_k_hop:
            raise ValueError(
                "model.theta.local_kNN must be larger than model.graph.kNN when "
                "model.theta.method='kalofolias' and model.theta.kalofolias.graph='local'"
            )

        theta_neighbor_list = self.theta_nearest_nodes[:, 1:].to(torch.long)
        if uses_local_kalofolias and theta_neighbor_list.numel() == 0:
            raise ValueError("theta_neighbor_list is empty; set model.theta.local_kNN > 0")
        if uses_local_kalofolias and torch.any(theta_neighbor_list < 0):
            bad_rows = torch.nonzero((theta_neighbor_list < 0).any(dim=1), as_tuple=False).flatten()
            preview = bad_rows[:10].detach().cpu().tolist()
            raise ValueError(
                "theta_neighbor_list contains -1 padding. Local Kalofolias "
                "requires a complete candidate list for every node. "
                f"bad_node_preview={preview}, theta.local_kNN={self.theta_k_hop}"
            )
        return theta_neighbor_list

    def set_debug_numerics(self, enabled, context=""):
        """Enable detailed tensor/gradient diagnostics for one training rerun."""
        self.debug_numerics = enabled
        self.debug_context = context
        self.debug_records = []

    @staticmethod
    def _debug_tensor_summary(name, tensor):
        """Summarize tensor health without keeping autograd history."""
        detached = tensor.detach()
        total = detached.numel()
        if total == 0:
            return f"{name}: shape={tuple(detached.shape)}, numel=0"

        finite_mask = torch.isfinite(detached)
        finite_count = finite_mask.sum().item()
        nan_count = torch.isnan(detached).sum().item()
        inf_count = torch.isinf(detached).sum().item()
        if finite_count == 0:
            return (
                f"{name}: shape={tuple(detached.shape)}, finite=0/{total}, "
                f"nan={nan_count}, inf={inf_count}"
            )

        finite_values = detached[finite_mask]
        return (
            f"{name}: shape={tuple(detached.shape)}, finite={finite_count}/{total}, "
            f"nan={nan_count}, inf={inf_count}, min={finite_values.min().item():.3e}, "
            f"max={finite_values.max().item():.3e}, mean={finite_values.mean().item():.3e}"
        )

    def _debug_tensor(self, name, tensor):
        """Record forward tensor health and attach a backward grad hook."""
        if not self.debug_numerics:
            return tensor

        label = f"{self.debug_context}.{name}" if self.debug_context else name
        self.debug_records.append("forward " + self._debug_tensor_summary(label, tensor))

        if tensor.requires_grad:
            def _grad_hook(grad, grad_label=label):
                summary = "backward " + self._debug_tensor_summary(f"grad({grad_label})", grad)
                self.debug_records.append(summary)
                if not torch.isfinite(grad).all():
                    raise RuntimeError(f"Non-finite gradient first observed at {grad_label}\n{summary}")
                return grad

            tensor.register_hook(_grad_hook)
        return tensor

    def regularized_terms(self, x, t=None):
        """Compute diagnostic regularization magnitudes on a full sequence.

        Args:
            x: full sequence, shape (B, T, N, C_signal).

        Returns:
            Per-block lists/tensors for signal norm, spatial Lu norm, temporal
            Ld L1 norm, and temporal Ld L2 norm.
        """
        assert not self.training, "only on validation and test"
        if self.use_norm:
            x, _, _ = layer_norm_on_data(x, self.norm_shape)

        x_norm_list = []
        x_Lu_norm_list = []
        Ldx_l1_list = []
        Ldx_l2_list = []

        with torch.no_grad():
            for i, block in enumerate(self.model_blocks):
                feature_extractor = block["feature_extractor"]
                graph_learn = block["graph_learning_module"]
                admm_block = block["ADMM_block"]

                # features: (B, T, N, H, feature_channels)
                features = feature_extractor(x)
                # u_ew/d_ew become the graph operators used inside ADMMBlock.
                u_ew, d_ew = graph_learn(features)
                admm_block.u_ew = u_ew
                admm_block.d_ew = d_ew

                p = self.skip_connection_weights[i]
                x_norm_list.append(torch.norm(x, dim=0))
                x_Lu_norm_list.append(torch.sqrt((x * admm_block.apply_op_Lu(x)).sum([1, 2, 3])))
                Ldx_l2_list.append(torch.norm(admm_block.apply_op_Ldr(x)))
                Ldx_l1_list.append(torch.norm(admm_block.apply_op_Ldr(x), p=1))

                x_old = x
                x_new = admm_block(x, self.t_in)
                x = p * x_new + (1 - p) * x_old

        return (
            torch.Tensor(x_norm_list),
            torch.Tensor(x_Lu_norm_list),
            torch.Tensor(Ldx_l1_list),
            torch.Tensor(Ldx_l2_list),
        )

    def clamp_param(self, alpha_max=None, beta_max=None):
        """Clamp learnable CG step sizes after optimizer updates."""
        for block in self.model_blocks:
            admm_block = block["ADMM_block"]
            self._clamp_cg_param(admm_block, "alpha", alpha_max)
            self._clamp_cg_param(admm_block, "beta", beta_max)

    def _clamp_cg_param(self, admm_block, prefix, max_value):
        suffixes = ["x"]
        if self.ablation != "simple":
            suffixes.append("zu")
        if self.ablation not in ["DGLR", "simple"]:
            suffixes.append("zd")

        for suffix in suffixes:
            param = getattr(admm_block, f"{prefix}_{suffix}")
            if max_value is None:
                param.data = torch.clamp(param.data, 0.0)
            else:
                param.data = torch.clamp(param.data, 0.0, max_value)

    def _record_glasso_events(self, block_idx, source, cov_matrix, events):
        """Store GLASSO fallback/eigensolver events from the latest forward."""
        for event in events:
            event = dict(event)
            event["block"] = block_idx
            event["theta_source"] = source
            event["cov_shape"] = tuple(cov_matrix.shape)
            self.last_glasso_events.append(event)

    def _uses_local_kalofolias(self):
        """Whether Theta is learned as local candidate-edge weights."""
        return self.theta_method == "kalofolias" and self.kalofolias_graph == "local"

    def _uses_dense_kalofolias(self):
        """Whether Theta is learned as a dense Kalofolias graph matrix."""
        return self.theta_method == "kalofolias" and self.kalofolias_graph == "dense"

    def _estimate_theta_for_block(self, block_idx, source, signal):
        """Estimate normalized Theta for one unrolled block.

        Args:
            block_idx: index of the current unrolled block.
            source: either ``"output"`` for the current reconstructed signal or
                ``"multi_x"`` for deflation modes from the previous block.
            signal: ``output`` with shape (B, T, N, C), or ``multi_x`` with
                shape (B, K, T, N, C).

        Returns:
            Dense GLASSO/Kalofolias branches return Theta with shape (N, N) for
            ``output`` source or (B, N, N) for ``multi_x`` source. The local
            Kalofolias branch returns candidate-edge weights with shape
            (N, K_theta) or (B, N, K_theta); ADMMBlock applies them through
            `theta_neighbor_list` without materializing a dense matrix.
        """
        if self.theta_method == "kalofolias" and self._uses_local_kalofolias():
            signal_matrix = node_signal_matrix(signal[..., 0:1])
            signal_matrix = self._debug_tensor(f"block_{block_idx}.local_kalofolias_signal_matrix", signal_matrix)
            theta = self.local_kalofolias_graph_estimator(signal_matrix)
            return self._debug_tensor(f"block_{block_idx}.Theta_local", theta)

        if self.theta_method == "kalofolias" and self._uses_dense_kalofolias():
            signal_matrix = node_signal_matrix(signal[..., 0:1])
            signal_matrix = self._debug_tensor(f"block_{block_idx}.kalofolias_signal_matrix", signal_matrix)
            theta = self.kalofolias_graph_estimator(signal_matrix)
            if self.kalofolias_output_mode.lower() == "laplacian":
                theta = normalize_theta(theta)
            return self._debug_tensor(f"block_{block_idx}.Theta", theta)

        if self.theta_method != "glasso":
            raise ValueError(f"Unsupported theta_method={self.theta_method}")

        cov_label = "theta_cov_from_multi_x" if source == "multi_x" else "theta_cov"
        cov_matrix = node_covariance(signal[..., 0:1], unbiased=True)
        cov_matrix = self._debug_tensor(f"block_{block_idx}.{cov_label}", cov_matrix)
        theta, glasso_events = glasso_estimation(
            cov_matrix,
            alpha=self.glasso_alpha,
            method=self.glasso_backend,
            rho=self.glasso_rho,
            eps=self.glasso_eps,
            eigh_shift=self.glasso_eigh_shift,
            eigh_shift_retries=self.glasso_eigh_shift_retries,
            fallback=self.glasso_fallback,
            max_iter=self.glasso_max_iter,
            tol=self.glasso_tol,
            allow_backward=self.glasso_allow_backward,
            return_events=True,
        )
        self._record_glasso_events(block_idx, source, cov_matrix, glasso_events)
        return self._debug_tensor(f"block_{block_idx}.Theta", theta)

    def forward(self, y, t_list, output_graph=False):
        """Run the stacked unrolling model.

        Args:
            y: observed sequence, shape (B, t_in, N, C_signal).
            t_list: time index sequence for the reconstructed horizon,
                shape (B, T).
            output_graph: when True, also return learned graph weights from
                each block.
            If `self.use_deflation` is True, intermediate ADMM blocks also
                compute multi-signal deflation modes. They are stored in
                `self.last_deflation_multi_x` and are not returned by this
                method. The final block skips deflation and returns only the
                normal ADMM-updated signal.

        Returns:
            output: (B, T, N, C_signal) unless use_one_channel=True, in which
                case the reconstructed path uses C=1 internally.
            If output_graph=True:
                undirected_graphs: (B, num_blocks, T, N, K, H)
                directed_graphs: (B, num_blocks, T - 1, interval, N, H)
        """
        batch_size = y.size(0)
        if output_graph:
            directed_graph_list = []
            undirected_graph_list = []
        self.last_deflation_multi_x = [] if self.use_deflation else None
        self.last_glasso_events = []
        multi_x = None  # Used for deflation across blocks.

        if self.use_norm:
            y, mean, std = layer_norm_on_data(y, self.y_norm_shape)

        # Initial guess for the full horizon. Shape: (B, T, N, C_signal).
        if self.use_extrapolation:
            output = self.linear_extrapolation(y)
        else:
            output = LR_guess(y, self.T, self.device)
        output = self._debug_tensor("initial_output", output)

        assert torch.isfinite(output).all(), "initial full-horizon guess contains NaN or Inf"
        if self.use_st_emb:
            # Shared across blocks because t_list and static node embeddings do
            # not depend on the current reconstructed signal.
            shared_output_emb = self.st_emb(t_list)

        for i, block in enumerate(self.model_blocks):
            is_first_block = i == 0
            is_last_block = i == self.num_blocks - 1
            # output_emb is what graph learning sees. It may include raw signal
            # channels plus spatial/temporal embedding channels.
            output_emb = torch.cat((output, shared_output_emb), -1) if self.use_st_emb else output
            output_emb = self._debug_tensor(f"block_{i}.output_emb", output_emb)
            output_old = output[..., 0:1] if self.use_one_channel and is_first_block else output

            feature_extractor = block["feature_extractor"]
            graph_learn = block["graph_learning_module"]
            admm_block = block["ADMM_block"]

            try:
                # features: (B, T, N, H, feature_channels)
                features = feature_extractor(output_emb)
                features = self._debug_tensor(f"block_{i}.features", features)
            except ValueError as exc:
                raise ValueError(
                    f"FeatureExtractor failed in block {i}: {exc}. "
                    f"output_emb_shape={tuple(output_emb.shape)}, ablation={self.ablation}"
                ) from exc

            try:
                # u_ew: (B, T, N, K, H)
                # d_ew: (B, T - 1, interval, N, H)
                u_ew, d_ew = graph_learn(features)
                u_ew = self._debug_tensor(f"block_{i}.u_ew", u_ew)
                d_ew = self._debug_tensor(f"block_{i}.d_ew", d_ew)
            except AssertionError as exc:
                raise ValueError(
                    f"GraphLearningModule failed in block {i}: {exc}. "
                    f"features_shape={tuple(features.shape)}, ablation={self.ablation}"
                ) from exc

            if output_graph:
                undirected_graph_list.append(u_ew.unsqueeze(1))
                directed_graph_list.append(d_ew.unsqueeze(1))

            admm_block.u_ew = u_ew
            admm_block.d_ew = d_ew
            admm_block.theta_neighbor_list = self.theta_neighbor_list if self._uses_local_kalofolias() else None
            admm_block.theta_operator_mode = self.kalofolias_output_mode.lower() if self._uses_local_kalofolias() else "matrix"
            # Theta branch:
            #   output[..., 0:1] is (B, T, N, 1).
            #   GLASSO methods first compute centered node covariance, then a
            #   normalized dense precision matrix. The Kalofolias method uses
            #   the smooth signal itself and returns a graph-derived matrix.
            #   Local Kalofolias also uses smooth signals, but returns only
            #   local edge weights over theta_neighbor_list.
            #   Plain output gives (N, N); deflated multi_x gives one matrix
            #   per batch, (B, N, N). Local Kalofolias gives (N, K_theta) or
            #   (B, N, K_theta). ADMMBlock.apply_op_Theta consumes either
            #   representation over the node dimension.
            if self.ablation == "Theta" or is_first_block:
                # The first block has no previous ADMM-refined signal for Theta
                # estimation. Theta ablation also disables this branch globally.
                admm_block.Theta = None
            elif multi_x is None or len(self.last_deflation_multi_x) == 0:
                admm_block.Theta = self._estimate_theta_for_block(i, "output", output)
            else:
                admm_block.Theta = self._estimate_theta_for_block(i, "multi_x", multi_x)

            try:
                if self.predict_only:
                    # Keep the observed prefix fixed before each ADMM correction.
                    output[:, : self.t_in] = y[..., 0:1] if self.use_one_channel else y

                # ADMM receives signal channels only, not ST embedding channels.
                run_deflation = self.use_deflation and not is_last_block and admm_block.ablation != "Theta"
                if self.use_one_channel:
                    admm_input = self._debug_tensor(f"block_{i}.admm_input", output[..., 0:1])
                    admm_result = admm_block(
                        admm_input,
                        self.t_in,
                        deflation=run_deflation,
                        deflation_samples=self.deflation_samples,
                    )
                else:
                    admm_input = self._debug_tensor(f"block_{i}.admm_input", output)
                    admm_result = admm_block(
                        admm_input,
                        self.t_in,
                        deflation=run_deflation,
                        deflation_samples=self.deflation_samples,
                    )
                if run_deflation:
                    output_new, multi_x = admm_result
                    output_new = self._debug_tensor(f"block_{i}.admm_output", output_new)
                    multi_x = self._debug_tensor(f"block_{i}.multi_x", multi_x)
                    self.last_deflation_multi_x.append(multi_x)
                else:
                    output_new = admm_result
                    output_new = self._debug_tensor(f"block_{i}.admm_output", output_new)
            except AssertionError as exc:
                raise ValueError(
                    f"ADMMBlock assertion failed in block {i}: {exc}. "
                    f"input_shape={tuple(output.shape)}, ablation={self.ablation}, "
                    f"use_deflation={self.use_deflation}, run_deflation={run_deflation}"
                ) from exc

            assert torch.isfinite(output_new).all(), f"output_new contains NaN or Inf in block {i}"
            p = self.skip_connection_weights[i]
            assert torch.isfinite(self.skip_connection_weights).all(), f"skip connection contains NaN or Inf in block {i}"
            # Learned residual/skip blend between the ADMM correction and the
            # previous block output.
            output = p * output_new + (1 - p) * output_old
            output = self._debug_tensor(f"block_{i}.skip_output", output)

        if self.use_norm:
            output = layer_recovery_on_data(output, self.norm_shape, mean, std)

        if output_graph:
            return output, torch.cat(undirected_graph_list, 1), torch.cat(directed_graph_list, 1)
        return output
