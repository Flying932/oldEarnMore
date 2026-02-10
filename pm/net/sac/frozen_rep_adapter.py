import torch.nn as nn

from pm.registry import NET


@NET.register_module(force=True)
class ResidualLightweightAdapter(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        x = self.drop(x)
        return residual + x


@NET.register_module(force=True)
class FrozenRepAdapterHierarchical(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bottom_adapter = ResidualLightweightAdapter(
            d_model=d_model,
            rank=rank,
            dropout=dropout,
        )
        self.mid_adapter = ResidualLightweightAdapter(
            d_model=d_model,
            rank=rank,
            dropout=dropout,
        )
        self.top_adapter = ResidualLightweightAdapter(
            d_model=d_model,
            rank=rank,
            dropout=dropout,
        )

    def forward(self, rep_state):
        rep_state = self.bottom_adapter(rep_state)
        rep_state = self.mid_adapter(rep_state)
        rep_state = self.top_adapter(rep_state)
        return rep_state
