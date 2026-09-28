"""
Book Graph Neural Network — Treats the order book as a graph of price levels.

Input: Same NPZ files as BookSpatialCNN from `lob_cache_builder --mode book`
  - book_tensors: (n_bars, 20, 4) float32
    20 levels: [10 bid levels (best to worst), 10 ask levels (best to worst)]
    4 features per level: [price_relative_to_mid, depth_lots, num_orders, queue_age_seconds]
  - mid_prices: (n_bars,) float64

Graph Structure (20 nodes, ~56 edges):
  - Each of the 20 price levels is a node
  - Edges encode the book's relational structure:
    1. Adjacent bid levels: bid[i] <-> bid[i+1] for i=0..8 (9 edges)
    2. Adjacent ask levels: ask[i] <-> ask[i+1] for i=0..8 (9 edges)
    3. Cross bid-ask: bid[i] <-> ask[i] for i=0..9 (10 edges, bid[0]-ask[0] = spread)
    4. Skip connections: bid[i] <-> bid[i+2], ask[i] <-> ask[i+2] (16 edges for multi-scale)
  - All edges are bidirectional (undirected graph)

Node Features (5 per node, derived from the 4 raw features + 1 computed):
  - price_distance: price relative to mid (raw feature 0)
  - log_depth: log1p(depth_lots) (raw feature 1, log-transformed)
  - log_orders: log1p(num_orders) (raw feature 2, log-transformed)
  - log_age: log1p(queue_age_seconds) (raw feature 3, log-transformed)
  - size_change: delta of depth from previous snapshot (computed)

Model Architecture:
  - Manual GCN implementation (no torch_geometric dependency)
  - Message passing: h' = ReLU(D^{-1/2} A D^{-1/2} h W + b)
  - 2-3 GCN layers with residual connections
  - Global mean + max pooling -> graph-level embedding
  - Temporal processing across window of snapshots
  - Linear head -> single regression output (mfe_net prediction)

Target: mfe_net_10s (same as CNN, for direct IC comparison)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# Graph Construction (static — same for all samples)
# =============================================================================

def build_order_book_graph(num_bid_levels: int = 10, num_ask_levels: int = 10) -> torch.Tensor:
    """
    Build the adjacency structure for the order book graph.

    Returns:
        edge_index: (2, num_edges) tensor of [source, target] pairs (undirected)

    Node indices:
        0-9:   bid levels (0 = best bid, 9 = worst bid)
        10-19: ask levels (10 = best ask, 19 = worst ask)
    """
    edges = []
    n_bid = num_bid_levels
    n_ask = num_ask_levels

    # 1. Adjacent bid levels: bid[i] <-> bid[i+1]
    for i in range(n_bid - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    # 2. Adjacent ask levels: ask[i] <-> ask[i+1]
    for i in range(n_ask - 1):
        edges.append((n_bid + i, n_bid + i + 1))
        edges.append((n_bid + i + 1, n_bid + i))

    # 3. Cross bid-ask edges: bid[i] <-> ask[i]
    #    bid[0] <-> ask[0] captures the spread (most important edge)
    for i in range(min(n_bid, n_ask)):
        edges.append((i, n_bid + i))
        edges.append((n_bid + i, i))

    # 4. Skip connections: bid[i] <-> bid[i+2], ask[i] <-> ask[i+2]
    #    Multi-scale view of depth structure
    for i in range(n_bid - 2):
        edges.append((i, i + 2))
        edges.append((i + 2, i))
    for i in range(n_ask - 2):
        edges.append((n_bid + i, n_bid + i + 2))
        edges.append((n_bid + i + 2, n_bid + i))

    # Self-loops (important for GCN: each node should aggregate its own features)
    for i in range(n_bid + n_ask):
        edges.append((i, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    return edge_index


def compute_norm_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute the normalized adjacency matrix: D^{-1/2} A D^{-1/2}.

    This is the standard GCN normalization from Kipf & Welling (2017).
    Pre-computed once since the graph structure is static.

    Args:
        edge_index: (2, E) edge indices
        num_nodes: total number of nodes

    Returns:
        norm_adj: (num_nodes, num_nodes) dense normalized adjacency matrix
    """
    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes)
    src, dst = edge_index[0], edge_index[1]
    adj[src, dst] = 1.0

    # Degree matrix
    deg = adj.sum(dim=1)  # (N,)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    # Normalized adjacency: D^{-1/2} A D^{-1/2}
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    return norm_adj


# =============================================================================
# GCN Layer (manual implementation — no torch_geometric needed)
# =============================================================================

class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Network layer.

    Implements: h' = sigma(A_norm @ h @ W + b)
    where A_norm = D^{-1/2} A D^{-1/2} is the pre-computed normalized adjacency.

    Args:
        in_features: input feature dimension per node
        out_features: output feature dimension per node
        bias: whether to include bias
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Glorot initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, in_features) node features
            norm_adj: (num_nodes, num_nodes) normalized adjacency (broadcast over batch)

        Returns:
            (batch, num_nodes, out_features)
        """
        # Linear transform: (B, N, F_in) @ (F_in, F_out) = (B, N, F_out)
        support = x @ self.weight

        # Message passing: (N, N) @ (B, N, F_out) -> need to handle batch dim
        # norm_adj is (N, N), support is (B, N, F_out)
        # torch.matmul handles broadcasting: (N, N) @ (B, N, F_out) via einsum
        out = torch.einsum('ij,bjf->bif', norm_adj, support)

        if self.bias is not None:
            out = out + self.bias

        return out


class GCNResBlock(nn.Module):
    """
    GCN layer with residual connection and layer normalization.

    If in_features != out_features, uses a linear projection for the skip connection.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.gcn = GCNLayer(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        if in_features != out_features:
            self.skip_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, in_features)
            norm_adj: (num_nodes, num_nodes)

        Returns:
            (batch, num_nodes, out_features)
        """
        residual = x if self.skip_proj is None else self.skip_proj(x)
        out = self.gcn(x, norm_adj)
        out = self.norm(out)
        out = F.gelu(out + residual)
        out = self.dropout(out)
        return out


# =============================================================================
# Graph Attention Layer (GAT — optional, for experiments)
# =============================================================================

class GATLayer(nn.Module):
    """
    Graph Attention Network layer (Velickovic et al., 2018).

    Learns attention coefficients per edge instead of using fixed adjacency weights.
    More expressive than GCN but slightly slower.

    Args:
        in_features: input feature dimension per node
        out_features: output feature dimension per node
        num_heads: number of attention heads (output is concatenated)
        dropout: attention dropout rate
    """
    def __init__(self, in_features: int, out_features: int,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention: a^T [Wh_i || Wh_j] decomposed as a_src^T Wh_i + a_dst^T Wh_j
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, in_features)
            edge_index: (2, E) edge indices
            num_nodes: N

        Returns:
            (batch, num_nodes, out_features)
        """
        B, N, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        # Project: (B, N, out_features) -> (B, N, H, D)
        h = self.W(x).view(B, N, H, D)

        src, dst = edge_index[0], edge_index[1]  # (E,)

        # Attention scores: e_{ij} = a_src^T h_i + a_dst^T h_j
        # h[:, src] -> (B, E, H, D), a_src -> (H, D)
        e_src = (h[:, src] * self.a_src).sum(dim=-1)  # (B, E, H)
        e_dst = (h[:, dst] * self.a_dst).sum(dim=-1)  # (B, E, H)
        e = F.leaky_relu(e_src + e_dst, negative_slope=0.2)  # (B, E, H)

        # Softmax over incoming edges per destination node
        # Build sparse attention using scatter
        E = edge_index.shape[1]
        alpha = torch.full((B, N, H), float('-inf'), device=x.device)

        # For each edge, compute attention for destination node
        # We need to do softmax per destination node
        # Simple approach: dense attention matrix (N is only 20, so this is fine)
        attn_matrix = torch.zeros(B, N, N, H, device=x.device)
        attn_matrix[:, dst, src, :] = e  # note: reversed for "message from src to dst"

        # Mask non-edges to -inf
        mask = torch.ones(N, N, dtype=torch.bool, device=x.device)
        mask[dst, src] = False
        attn_matrix[:, mask, :] = float('-inf')

        attn_matrix = F.softmax(attn_matrix, dim=2)  # softmax over source dim
        attn_matrix = self.dropout(attn_matrix)

        # Replace NaN from all-inf rows with 0
        attn_matrix = torch.nan_to_num(attn_matrix, nan=0.0)

        # Aggregate: out[i] = sum_j alpha[i,j] * h[j]
        # attn: (B, N, N, H), h: (B, N, H, D)
        # out[b,i,h,d] = sum_j attn[b,i,j,h] * h[b,j,h,d]
        out = torch.einsum('bijh,bjhd->bihd', attn_matrix, h)  # (B, N, H, D)
        out = out.reshape(B, N, H * D)  # (B, N, out_features)

        return out


# =============================================================================
# Book GNN Model
# =============================================================================

class BookGNN(nn.Module):
    """
    Graph Neural Network for order book prediction.

    Treats each snapshot's 20 price levels as a graph, processes with GCN layers,
    pools to a graph-level embedding, then processes the temporal sequence.

    Architecture:
        1. Node feature expansion (4 raw -> node_features with derived features)
        2. GCN layers with residual connections (process graph structure)
        3. Global mean+max pooling (graph -> vector)
        4. Temporal Conv1D across window (capture dynamics)
        5. Linear regression head -> 1 output (mfe_net prediction)

    Args:
        window_size: Number of consecutive bars per sample (default: 20)
        num_nodes: Number of price levels (default: 20 = 10 bid + 10 ask)
        in_features: Raw features per node from NPZ (default: 4)
        node_features: Expanded features per node after enrichment (default: 5)
        hidden_dim: GCN hidden dimension (default: 64)
        num_gcn_layers: Number of GCN layers (default: 2)
        temporal_dim: Temporal conv channels (default: 128)
        dropout: Dropout rate (default: 0.2)
        num_classes: Output dimension (default: 1 for regression)
        use_gat: If True, use GAT instead of GCN (default: False)
        gat_heads: Number of GAT attention heads (default: 4)
    """
    def __init__(
        self,
        window_size: int = 20,
        num_nodes: int = 20,
        in_features: int = 4,
        node_features: int = 5,
        hidden_dim: int = 64,
        num_gcn_layers: int = 2,
        temporal_dim: int = 128,
        dropout: float = 0.2,
        num_classes: int = 1,
        use_gat: bool = False,
        gat_heads: int = 4,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat

        # --- Build static graph structure ---
        self.edge_index = build_order_book_graph(num_bid_levels=10, num_ask_levels=10)
        # Pre-compute normalized adjacency (registered as buffer so it moves to GPU)
        norm_adj = compute_norm_adj(self.edge_index, num_nodes)
        self.register_buffer('norm_adj', norm_adj)

        # --- Node feature projection ---
        # Input: 5 features (4 raw + 1 computed size_change)
        self.node_proj = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # --- GCN layers ---
        if use_gat:
            self.gcn_layers = nn.ModuleList()
            for i in range(num_gcn_layers):
                in_dim = hidden_dim
                out_dim = hidden_dim
                self.gcn_layers.append(GATLayer(in_dim, out_dim,
                                                num_heads=gat_heads, dropout=dropout))
            self.gcn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim)
                                            for _ in range(num_gcn_layers)])
            self.gcn_dropouts = nn.ModuleList([nn.Dropout(dropout)
                                               for _ in range(num_gcn_layers)])
        else:
            self.gcn_layers = nn.ModuleList()
            for i in range(num_gcn_layers):
                self.gcn_layers.append(GCNResBlock(hidden_dim, hidden_dim, dropout=dropout))

        # --- Graph-level readout ---
        # mean + max pooling across nodes -> 2 * hidden_dim
        graph_embed_dim = 2 * hidden_dim

        # --- Temporal processing ---
        # Process sequence of graph embeddings across the window
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(graph_embed_dim, temporal_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(temporal_dim),
            nn.GELU(),
        )

        self.temporal_conv1 = nn.Sequential(
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.temporal_conv2 = nn.Sequential(
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(temporal_dim),
        )
        self.temporal_dropout = nn.Dropout(dropout)

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # --- Prediction head ---
        self.head = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim // 2, num_classes),
        )

    def _extract_node_features(
        self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract 5 node features from raw 4-feature book tensor.

        Input x: (batch, 20, 4) — one snapshot
            [0] price_relative_to_mid (already in ticks)
            [1] depth_lots (raw, NOT log-transformed — we do it here)
            [2] num_orders (raw)
            [3] queue_age_seconds (raw)

        NOTE: The training script (train_walkforward.py) log-transforms features 1,2,3
        BEFORE passing to the dataset. So by the time we see the data, features 1,2,3
        are already log1p-transformed. We do NOT re-transform.

        Computed feature:
            [4] size_change: depth[t] - depth[t-1] (0 if no previous snapshot)

        Returns: (batch, 20, 5)
        """
        B, N, Fin = x.shape

        if x_prev is not None:
            # size_change = current depth - previous depth (both already log-transformed)
            size_change = x[:, :, 1:2] - x_prev[:, :, 1:2]  # (B, N, 1)
        else:
            size_change = torch.zeros(B, N, 1, device=x.device, dtype=x.dtype)

        # Concatenate: 4 raw features + 1 computed = 5
        node_feats = torch.cat([x, size_change], dim=-1)  # (B, N, 5)
        return node_feats

    def _process_single_snapshot(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Process one graph snapshot through GCN layers and pooling.

        Args:
            node_feats: (batch, num_nodes, node_features) — features for one timestep

        Returns:
            graph_embed: (batch, 2*hidden_dim) — graph-level embedding
        """
        # Project node features to hidden dim
        h = self.node_proj(node_feats)  # (B, N, hidden_dim)

        # GCN message passing
        if self.use_gat:
            for gcn, norm, drop in zip(self.gcn_layers, self.gcn_norms, self.gcn_dropouts):
                residual = h
                h = gcn(h, self.edge_index, self.num_nodes)
                h = norm(h)
                h = F.gelu(h + residual)
                h = drop(h)
        else:
            for gcn_block in self.gcn_layers:
                h = gcn_block(h, self.norm_adj)  # (B, N, hidden_dim)

        # Global readout: mean + max pooling across nodes
        h_mean = h.mean(dim=1)  # (B, hidden_dim)
        h_max = h.max(dim=1).values  # (B, hidden_dim)
        graph_embed = torch.cat([h_mean, h_max], dim=-1)  # (B, 2*hidden_dim)

        return graph_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full GNN model.

        Args:
            x: (batch, window_size, 20, 4) book snapshot window
               Features are already log-transformed by the data loader.

        Returns:
            output: (batch, num_classes) predictions
        """
        B, T, N, Fin = x.shape

        # Process each timestep through the GNN
        graph_embeds = []
        prev_snapshot = None
        for t in range(T):
            snapshot = x[:, t, :, :]  # (B, 20, 4)
            node_feats = self._extract_node_features(snapshot, prev_snapshot)  # (B, 20, 5)
            graph_embed = self._process_single_snapshot(node_feats)  # (B, 2*hidden_dim)
            graph_embeds.append(graph_embed)
            prev_snapshot = snapshot

        # Stack temporal sequence: (B, T, 2*hidden_dim)
        temporal_seq = torch.stack(graph_embeds, dim=1)

        # Temporal processing: (B, 2*hidden_dim, T) for Conv1d
        temporal_seq = temporal_seq.permute(0, 2, 1)

        out = self.temporal_stem(temporal_seq)  # (B, temporal_dim, T)

        # Temporal residual block
        residual = out.clone()
        out = self.temporal_conv1(out)
        out = self.temporal_conv2(out)
        out = F.gelu(out + residual)  # residual connection
        out = self.temporal_dropout(out)

        out = self.temporal_pool(out).squeeze(-1)  # (B, temporal_dim)

        # Predict
        output = self.head(out)  # (B, num_classes)
        return output


# =============================================================================
# Convenience: count parameters
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("BookGNN Architecture Verification")
    print("=" * 60)

    # Test GCN variant
    model = BookGNN(
        window_size=20, num_nodes=20, in_features=4, node_features=5,
        hidden_dim=64, num_gcn_layers=2, temporal_dim=128,
        dropout=0.2, num_classes=1, use_gat=False,
    )
    n_params = count_parameters(model)
    print(f"\nGCN variant parameters: {n_params:,}")

    # Test forward pass
    B, T = 4, 20
    x = torch.randn(B, T, 20, 4)
    out = model(x)
    print(f"Input shape:  ({B}, {T}, 20, 4)")
    print(f"Output shape: {out.shape}")  # Should be (4, 1)
    print(f"Output: {out.squeeze(-1).detach()}")

    # Test GAT variant
    model_gat = BookGNN(
        window_size=20, num_nodes=20, in_features=4, node_features=5,
        hidden_dim=64, num_gcn_layers=2, temporal_dim=128,
        dropout=0.2, num_classes=1, use_gat=True, gat_heads=4,
    )
    n_params_gat = count_parameters(model_gat)
    print(f"\nGAT variant parameters: {n_params_gat:,}")
    out_gat = model_gat(x)
    print(f"GAT Output shape: {out_gat.shape}")
    print(f"GAT Output: {out_gat.squeeze(-1).detach()}")

    # Print graph structure
    edge_index = build_order_book_graph()
    print(f"\nGraph: {20} nodes, {edge_index.shape[1]} edges (directed, including self-loops)")
    print(f"Edge index shape: {edge_index.shape}")
