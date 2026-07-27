import torch
import torch.nn as nn
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


def glasso_estimation(
    cov_matrix,
    alpha=0.2,
    method="quic",
    max_iter=20,
    tol=1e-4,
    allow_backward=False,
):
    """
    Estimate the precision matrix (inverse covariance) using Graphical Lasso.
    Args:
        cov_matrix: Dense node covariance matrix, shape (N, N).
        alpha: Regularization parameter for Graphical Lasso.
        method: One of "admm", "quic", or "sklearn".
        max_iter: Maximum solver iterations. Capped at 20 for this model.
        tol: Solver tolerance.
        allow_backward: If False, Theta estimation is detached from autograd.
    Returns:
        Dense node precision matrix Theta, shape (N, N).
    """
    device = cov_matrix.device
    dtype = cov_matrix.dtype
    max_iter = min(max_iter, 20)
    method = method.lower()
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.transpose(-1, -2))
    solver_cov = cov_matrix if allow_backward else cov_matrix.detach()

    if method == "admm":
        try:
            from glasso_pytorch import graphical_lasso

            if allow_backward:
                result = graphical_lasso(
                    solver_cov,
                    alpha=alpha,
                    max_iter=max_iter,
                    tol=tol,
                    rtol=tol,
                    return_info=True,
                )
            else:
                with torch.no_grad():
                    result = graphical_lasso(
                        solver_cov,
                        alpha=alpha,
                        max_iter=max_iter,
                        tol=tol,
                        rtol=tol,
                        return_info=True,
                    )
            return result.precision.to(device=device, dtype=dtype)
        except ImportError:
            method = "sklearn"

    if method == "quic":
        try:
            import numpy as np
            from inverse_covariance import quic

            cov_np = solver_cov.detach().cpu().numpy().astype(np.float64, copy=False)
            lam = np.full_like(cov_np, alpha, dtype=np.float64)
            np.fill_diagonal(lam, 0.0)
            precision, _, _, _, _, _ = quic(cov_np, lam, tol=tol, max_iter=max_iter)
            return torch.tensor(precision, dtype=dtype, device=device)
        except ImportError:
            method = "sklearn"

    if method == "sklearn":
        from sklearn.covariance import GraphicalLasso

        model = GraphicalLasso(alpha=alpha, max_iter=max_iter, tol=tol)
        model.fit(solver_cov.detach().cpu().numpy())
        return torch.tensor(model.precision_, dtype=dtype, device=device)

    raise ValueError(f"Unknown graphical lasso method: {method}")

DEFAULT_GRAPH_INFO = {
    "n_nodes": None,
    "u_edges": None,
    "u_dist": None,
}

DEFAULT_ADMM_INFO = {
    "ADMM_iters": 30,
    "CG_iters": 3,
    "PGD_iters": 3,
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
        predict_only=False,
        le_emb=False,
        glasso_method="quic",
        glasso_max_iter=20,
        glasso_tol=1e-4,
        glasso_allow_backward=False,
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
        self.glasso_method = glasso_method
        self.glasso_max_iter = min(glasso_max_iter, 20)
        self.glasso_tol = glasso_tol
        self.glasso_allow_backward = glasso_allow_backward

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
                        # independent set of CGSolver modules/parameters:
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
                        ),
                    }
                )
            )

        self.y_norm_shape = [self.t_in, graph_info["n_nodes"], signal_channels]
        self.norm_shape = [self.T, graph_info["n_nodes"], signal_channels]

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

    def forward(self, y, t_list, output_graph=False):
        """Run the stacked unrolling model.

        Args:
            y: observed sequence, shape (B, t_in, N, C_signal).
            t_list: time index sequence for the reconstructed horizon,
                shape (B, T).
            output_graph: when True, also return learned graph weights from
                each block.

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

        if self.use_norm:
            y, mean, std = layer_norm_on_data(y, self.y_norm_shape)

        # Initial guess for the full horizon. Shape: (B, T, N, C_signal).
        if self.use_extrapolation:
            output = self.linear_extrapolation(y)
        else:
            output = LR_guess(y, self.T, self.device)

        assert not torch.isnan(output).any(), "linear extrapolation has nan"
        if self.use_st_emb:
            # Shared across blocks because t_list and static node embeddings do
            # not depend on the current reconstructed signal.
            shared_output_emb = self.st_emb(t_list)

        for i, block in enumerate(self.model_blocks):
            # output_emb is what graph learning sees. It may include raw signal
            # channels plus spatial/temporal embedding channels.
            output_emb = torch.cat((output, shared_output_emb), -1) if self.use_st_emb else output
            output_old = output[..., 0:1] if self.use_one_channel and i == 0 else output

            feature_extractor = block["feature_extractor"]
            graph_learn = block["graph_learning_module"]
            admm_block = block["ADMM_block"]

            try:
                # features: (B, T, N, H, feature_channels)
                features = feature_extractor(output_emb)
            except ValueError as exc:
                raise ValueError(f"Error in Feature extractor in Block {i}: {exc}") from exc

            try:
                # u_ew: (B, T, N, K, H)
                # d_ew: (B, T - 1, interval, N, H)
                u_ew, d_ew = graph_learn(features)
            except AssertionError as exc:
                raise ValueError(f"Error in Graph Learning Module in Block {i}: {exc}") from exc

            if output_graph:
                undirected_graph_list.append(u_ew.unsqueeze(1))
                directed_graph_list.append(d_ew.unsqueeze(1))

            admm_block.u_ew = u_ew
            admm_block.d_ew = d_ew
            # Theta branch:
            #   output[..., 0:1] is (B, T, N, 1).
            #   cov_matrix aggregates over batch, time and channel -> (N, N).
            #   glasso_estimation returns Theta with shape (N, N), consumed by
            #   ADMMBlock.apply_op_Theta over the node dimension.
            cov_matrix = torch.einsum(
                "btic,btjc->ij",
                output[..., 0:1],
                output[..., 0:1],
            ) / (batch_size * self.T)
            admm_block.Theta = glasso_estimation(
                cov_matrix,
                method=self.glasso_method,
                max_iter=self.glasso_max_iter,
                tol=self.glasso_tol,
                allow_backward=self.glasso_allow_backward,
            )

            try:
                if self.predict_only:
                    # Keep the observed prefix fixed before each ADMM correction.
                    output[:, : self.t_in] = y[..., 0:1] if self.use_one_channel else y

                # ADMM receives signal channels only, not ST embedding channels.
                if self.use_one_channel:
                    output_new = admm_block(output[..., 0:1], self.t_in)
                else:
                    output_new = admm_block(output, self.t_in)
            except AssertionError as exc:
                raise ValueError(f"Assertation Error in ADMM block in block {i} - {exc}") from exc

            assert not torch.isnan(output_new).any(), f"output_new has NaN value in block {i}"
            p = self.skip_connection_weights[i]
            assert not torch.isnan(self.skip_connection_weights).any(), f"skip connection has NaN Values in block {i}"
            # Learned residual/skip blend between the ADMM correction and the
            # previous block output.
            output = p * output_new + (1 - p) * output_old

        if self.use_norm:
            output = layer_recovery_on_data(output, self.norm_shape, mean, std)

        if output_graph:
            return output, torch.cat(undirected_graph_list, 1), torch.cat(directed_graph_list, 1)
        return output
