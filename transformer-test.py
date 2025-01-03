import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float
import wandb
import einops


class Transformer(nn.Module):
    """transformer model"""

    def __init__(self, input_dim=10, hidden_dim=128, n_heads=8):
        super().__init__()

        # self.input_mlp = nn.Linear(input_dim, hidden_dim)
        self.attention = Attention(hidden_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.output_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: Float[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len"]:
        x = self.attention(x) + x
        x = self.norm1(x)
        x = self.output_mlp(x) + x
        x = self.norm2(x)

        return x


class Attention(nn.Module):
    """attention module"""

    def __init__(self, hidden_dim=128, n_heads=8):
        super().__init__()

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

    def forward(
        self, x: Float[Tensor, "batch seq_len hdim"]
    ) -> Float[Tensor, "batch seq_len hdim"]:
        # TODO: annotate shapes

        Q, K, V = self.q(x), self.k(x), self.v(x)
        Q = einops.rearrange(
            Q,
            "batch seq_len (n_heads hdim) -> batch n_heads seq_len hdim",
            n_heads=self.n_heads,
        )
        K = einops.rearrange(
            K,
            "batch seq_len (n_heads hdim) -> batch n_heads seq_len hdim",
            n_heads=self.n_heads,
        )
        V = einops.rearrange(
            V,
            "batch seq_len (n_heads hdim) -> batch n_heads seq_len hdim",
            n_heads=self.n_heads,
        )

        QK_1 = torch.matmul(Q, K.transpose(-2, -1)) / (self.n_heads**0.5)
        QK_2 = torch.einsum("b n i j, b n d j -> b n i d", Q, K) / (self.n_heads**0.5)

        QK = QK_1
        softmax_QK = F.softmax(QK, dim=-1)
        # attention = torch.matmul(softmax_QK, V)
        attention = torch.einsum("b n s S, b n S d -> b n s d", softmax_QK, V)
        QKV = attention
        QKV = einops.rearrange(
            QKV, "batch n_heads seq_len hdim -> batch seq_len (n_heads hdim)"
        )
        output = self.output(QKV)

        return output


if __name__ == "__main__":
    # mlp -> attention * N -> mlp ->
    TestAttention = False
    if TestAttention:
        attn = Attention(64, 8)
        x = torch.rand(32, 10, 64)
        print(attn(x).shape)

    TestTransformer = True
    if TestTransformer:
        batch_size = 32
        seq_len = 10
        n_dim = 64
        n_heads = 8

        x = torch.rand(batch_size, seq_len, n_dim)
        transformer = Transformer(seq_len, n_dim, n_heads=8)
        out = transformer(x)
        print(out.shape)

# write unit tests for the model!
