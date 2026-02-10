import torch
from typing import List
from functools import partial
from pm.registry import NET
from pm.net import MAE
from einops import rearrange, repeat
from typing import Final, Set, Optional, Union, Tuple
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, use_fused_attn


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, k, v):
        q = q + self.drop_path1(self.ls1(self.attn(self.norm1(q), k, v)))
        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q

@NET.register_module(force=True)
class MaskTimeState(MAE):
    def __init__(self,
                *args,
                embed_type: str = "TimesEmbed",
                feature_size: List[int] = (10, 99),
                patch_size: List[int] = (10, 99),
                t_patch_size: int = 1,
                num_stocks: int = 420,
                pred_num_stocks: int = 420,
                in_chans: int = 1,
                embed_dim: int = 128,
                depth: int = 2,
                num_heads: int = 4,
                decoder_embed_dim: int = 64,
                decoder_depth: int = 1,
                decoder_num_heads: int = 8,
                mlp_ratio: float = 4.0,
                norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                norm_pix_loss: bool = False,
                cls_embed: bool = True,
                sep_pos_embed: bool = True,
                trunc_init: bool = False,
                no_qkv_bias: bool = False,
                mask_ratio_min: float = 0.5,
                mask_ratio_max: float = 1.0,
                mask_ratio_mu: float = 0.55,
                mask_ratio_std:float = 0.25,
                contrastive_dim: int = None,
                contrastive_temperature: float = 0.2,
                corr_positive_weight: float = 0.5,
                ** kwargs
                ):
        super(MaskTimeState, self).__init__(
            *args,
            embed_type=embed_type,
            feature_size = feature_size,
            patch_size = patch_size,
            t_patch_size = t_patch_size,
            num_stocks = num_stocks,
            pred_num_stocks = pred_num_stocks,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            decoder_embed_dim = decoder_embed_dim,
            decoder_depth = decoder_depth,
            decoder_num_heads = decoder_num_heads,
            mlp_ratio = mlp_ratio,
            norm_layer = norm_layer,
            norm_pix_loss = norm_pix_loss,
            cls_embed = cls_embed,
            sep_pos_embed = sep_pos_embed,
            trunc_init = trunc_init,
            no_qkv_bias = no_qkv_bias,
            mask_ratio_min = mask_ratio_min,
            mask_ratio_max = mask_ratio_max,
            mask_ratio_mu = mask_ratio_mu,
            mask_ratio_std = mask_ratio_std,
            **kwargs,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                CrossBlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )
        self.contrastive_dim = embed_dim if contrastive_dim is None else contrastive_dim
        self.contrastive_temperature = contrastive_temperature
        self.corr_positive_weight = corr_positive_weight
        self.contrastive_head = nn.Sequential(
            nn.Linear(embed_dim, self.contrastive_dim),
            nn.GELU(),
            nn.Linear(self.contrastive_dim, self.contrastive_dim),
        )

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        if getattr(self.patch_embed, "proj", None) is not None:
            w = self.patch_embed.proj.weight.data
            if self.trunc_init:
                torch.nn.init.trunc_normal_(w)
                torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_full_ids_restore(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    def forward_encoder(self, x, mask = None, ids_restore = None, if_mask = True):
        """
        b, c, n, d, f = x.shape # batch size, in chans, num stocks, days, features
        """
        x = self.patch_embed(x)

        B, L, C = x.shape

        if if_mask:
            if mask is None:
                mask_ratio = self.mask_ratio_generator.rvs(1)[0]
                x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
                x = x.view(B, -1, C)
            else:
                ids_keep = torch.argsort(ids_restore, dim=1)[:, :(mask[0, :] == 0).sum().item()]
                x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
        else:
            ids_keep = torch.arange(0, L).unsqueeze(0).repeat(B, 1).to(x.device)
            if mask is None:
                mask = torch.zeros((B, L), device=x.device, dtype=torch.float32)
            if ids_restore is None:
                ids_restore = self._build_full_ids_restore(B, L, x.device)

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1],
                dim=1,
            )

            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )

        x = x.view([B, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    def forward_decoder(self, x, kv, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        kv = self.decoder_embed(kv)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])

        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)
            kv = torch.cat((decoder_cls_tokens, kv), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])
            kv = kv.view([N, T, H * W, C])

        k = v = kv

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, k, v)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_state(self, x, mask=None, ids_restore=None, if_mask: bool = True):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        x, mask, ids_restore = self.forward_encoder(x,
                                                    mask = mask,
                                                    ids_restore = ids_restore,
                                                    if_mask = if_mask)

        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size
        if ids_restore is None:
            ids_restore = self._build_full_ids_restore(N, T * H * W, x.device)

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])

        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    @staticmethod
    def nt_xent_loss(z1, z2, temperature=0.2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        z = torch.cat([z1, z2], dim=0)  # [2B, C]
        logits = torch.matmul(z, z.t()) / temperature
        bsz = z1.shape[0]
        labels = torch.arange(2 * bsz, device=z.device)
        labels = (labels + bsz) % (2 * bsz)
        self_mask = torch.eye(2 * bsz, dtype=torch.bool, device=z.device)
        mask_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(self_mask, mask_value)
        return F.cross_entropy(logits.float(), labels)

    def _flatten_contrastive_tokens(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, z.shape[-1])

    def forward_pretrain(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor = None,
        lambda_recon: float = 1.0,
        lambda_contrastive: float = 0.7,
        temperature: float = None,
        corr_pos_index: torch.Tensor = None,
        corr_positive_weight: float = None,
        mask: torch.Tensor = None,
        ids_restore: torch.Tensor = None,
    ):
        if temperature is None:
            temperature = self.contrastive_temperature
        if corr_positive_weight is None:
            corr_positive_weight = self.corr_positive_weight
        corr_positive_weight = float(min(max(corr_positive_weight, 0.0), 1.0))

        if x2 is None:
            x2 = x1

        if len(x1.shape) == 4:
            x1 = x1.unsqueeze(1)
        if len(x2.shape) == 4:
            x2 = x2.unsqueeze(1)

        recon_loss, recon_mask, recon_ids_restore = self.forward(
            x1,
            mask=mask,
            ids_restore=ids_restore,
        )

        h1, _, _ = self.forward_encoder(x1, if_mask=False)
        h2, _, _ = self.forward_encoder(x2, if_mask=False)
        z1 = self.contrastive_head(h1)
        z2 = self.contrastive_head(h2)

        z1_flat = self._flatten_contrastive_tokens(z1)
        z2_flat = self._flatten_contrastive_tokens(z2)
        contrastive_same = self.nt_xent_loss(z1_flat, z2_flat, temperature=temperature)

        contrastive_corr = torch.zeros_like(contrastive_same)
        if corr_pos_index is not None:
            if not torch.is_tensor(corr_pos_index):
                corr_pos_index = torch.as_tensor(corr_pos_index, dtype=torch.long)
            corr_pos_index = corr_pos_index.to(z2.device, dtype=torch.long)
            corr_pos_index = corr_pos_index.clamp(0, z2.shape[1] - 1)
            z2_corr = torch.index_select(z2, dim=1, index=corr_pos_index)
            z2_corr_flat = self._flatten_contrastive_tokens(z2_corr)
            contrastive_corr = self.nt_xent_loss(z1_flat, z2_corr_flat, temperature=temperature)
            contrastive_loss = (
                (1.0 - corr_positive_weight) * contrastive_same
                + corr_positive_weight * contrastive_corr
            )
        else:
            contrastive_loss = contrastive_same

        total_loss = lambda_recon * recon_loss + lambda_contrastive * contrastive_loss
        return {
            "loss_total": total_loss,
            "loss_recon": recon_loss,
            "loss_contrastive": contrastive_loss,
            "loss_contrastive_same": contrastive_same,
            "loss_contrastive_corr": contrastive_corr,
            "mask": recon_mask,
            "ids_restore": recon_ids_restore,
        }

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        _imgs = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.num_stocks,
            )
            .long()
            .to(imgs.device),
        )

        target = self.patchify(_imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask = None, ids_restore = None):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        latent, mask, ids_restore = self.forward_encoder(x,
                                                         mask = mask,
                                                         ids_restore = ids_restore)

        kv = self.forward_encoder(x,if_mask=False)[0]

        pred = self.forward_decoder(latent, kv, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, mask, ids_restore

if __name__ == '__main__':
    device = torch.device("cpu")

    model = MaskTimeState(
        feature_size=(10, 102),
        patch_size=(10, 102),
        frames=420,
        t_patch_size=1,
        input_dim=102,
        temporal_dim=3,
        embed_dim=128
    ).to(device)
    print(model)

    feature = torch.randn(4, 1, 420, 10, 99)
    temporal = torch.zeros(4, 1, 420, 10, 3)

    batch = torch.cat([feature, temporal], dim=-1)

    loss, mask, ids_restore = model(batch)
    print(loss, mask.shape, ids_restore.shape)

    x, mask, ids_restore = model.forward_state(batch)
    print(x.shape, mask.shape, ids_restore.shape)
