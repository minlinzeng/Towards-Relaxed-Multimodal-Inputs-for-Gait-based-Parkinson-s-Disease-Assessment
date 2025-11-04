from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_encoder import (
    SkeletonMLP, SensorEncoder, SharedBackbone, TaskHead
)

class FOCALSharedLatentBaseline(nn.Module):
    """
    Shared-latent baseline WITHOUT pretraining.
    - Two encoders (skeleton, sensor)
    - Four projectors (shared/private per modality)
    - Fixed-width fuse vector: concat[SHARED_fused, PRIVATE_skel, PRIVATE_imu]
    - If synced pair: fuse shared (mean), concat privates, one backbone → one head
    - If async/unpaired: no cross-mix; run each modality separately through the same backbone → two heads
    Shapes:
      S = skel_enc(x_skel) : (B, T, Cs)
      M = sens_enc(x_imu)  : (B, T, Cm)
      S_sh = P_sk_sh(S)    : (B, T, d_sh)
      S_pr = P_sk_pr(S)    : (B, T, d_pr)
      M_sh = P_im_sh(M)    : (B, T, d_sh)
      M_pr = P_im_pr(M)    : (B, T, d_pr)
      F_synced   = concat( 0.5*(S_sh+M_sh), S_pr, M_pr ) : (B, T, d_sh+2*d_pr)
      F_skelOnly = concat( S_sh,            S_pr, zeros ) : (B, T, d_sh+2*d_pr)
      F_imuOnly  = concat( M_sh,            zeros, M_pr ) : (B, T, d_sh+2*d_pr)
    Returns:
      if synced=True and both present: (logits_sync, None)
      else: (logits_skel or None, logits_imu or None)
    """
    def __init__(
        self,
        *,
        skeleton_input_dim: int,
        skeleton_output_dim: int,
        sensor_in_channels: int,
        sensor_out_channels: int,
        sensor_length: int,
        d_shared: int = 128,
        d_private: int = 64,
        shared_out_channels: int = 16,
        backbone_dim: int = 8,
        num_classes: int = 2,
        use_norm_head: bool = False,
        use_cosine_head: bool = False
    ):
        super().__init__()

        # Encoders
        self.skel_enc = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.sens_enc = SensorEncoder(
            in_channels=sensor_in_channels,
            out_channels=sensor_out_channels,
            sensor_length=sensor_length
        )

        # Projectors: shared & private per modality
        self.sk_sh = nn.Linear(skeleton_output_dim, d_shared)
        self.sk_pr = nn.Linear(skeleton_output_dim, d_private)
        self.im_sh = nn.Linear(sensor_out_channels,  d_shared)
        self.im_pr = nn.Linear(sensor_out_channels,  d_private)

        # Fixed input width for the shared backbone
        in_channels_backbone = d_shared + 2 * d_private
        self.backbone = SharedBackbone(
            in_channels=in_channels_backbone,
            shared_out_channels=shared_out_channels,
            backbone_dim=backbone_dim
        )
        feat_dim = backbone_dim * shared_out_channels

        # Heads
        self.head_sync   = TaskHead(feat_dim, num_classes, use_norm_head, use_cosine_head)
        self.head_skel   = TaskHead(feat_dim, num_classes, use_norm_head, use_cosine_head)
        self.head_sensor = TaskHead(feat_dim, num_classes, use_norm_head, use_cosine_head)

    @staticmethod
    def _concat_fixed(z_shared: torch.Tensor,
                      z_pr_s: torch.Tensor,
                      z_pr_m: torch.Tensor) -> torch.Tensor:
        """Concat along channel dim keeping fixed width."""
        return torch.cat([z_shared, z_pr_s, z_pr_m], dim=-1)

    def _zeros_like_private(self, ref: torch.Tensor, d_private: int) -> torch.Tensor:
        B, T = ref.shape[0], ref.shape[1]
        return torch.zeros(B, T, d_private, device=ref.device, dtype=ref.dtype)

    def forward(
        self,
        x_skel: Optional[torch.Tensor] = None,
        x_sensor: Optional[torch.Tensor] = None,
        *,
        synced: bool = False
    ):
        """
        x_skel   : (B,T_skel,D_skel) or None
        x_sensor : (B,T_imu,C_imu)  or None
        synced   : True only if same subject & time window; else False
        """
        out_sync = out_skel = out_imu = None

        # Encode present modalities
        S = self.skel_enc(x_skel) if x_skel is not None else None   # (B,T,Cs)
        M = self.sens_enc(x_sensor) if x_sensor is not None else None  # (B,T,Cm)

        # Short-circuit if nothing is provided
        if (S is None) and (M is None):
            raise ValueError("Both x_skel and x_sensor are None.")

        # Project to shared/private
        S_sh = self.sk_sh(S) if S is not None else None
        S_pr = self.sk_pr(S) if S is not None else None
        M_sh = self.im_sh(M) if M is not None else None
        M_pr = self.im_pr(M) if M is not None else None

        # Synced pair: fuse shared by mean, keep both privates
        if synced and (S is not None) and (M is not None):
            SH = 0.5 * (S_sh + M_sh)
            F = self._concat_fixed(SH, S_pr, M_pr)                     # (B,T,d_sh+2*d_pr)
            repr_ = self.backbone(F).flatten(1)                         # (B, D)
            out_sync = self.head_sync(repr_)
            return out_sync, None

        # Async / unpaired: run each available modality independently (no cross-mix)
        if S is not None:
            zeros_pr_m = self._zeros_like_private(S, M_pr.shape[-1] if M_pr is not None else self.sk_pr.out_features)
            F_s = self._concat_fixed(S_sh, S_pr, zeros_pr_m)
            repr_s = self.backbone(F_s).flatten(1)
            out_skel = self.head_skel(repr_s)

        if M is not None:
            zeros_pr_s = self._zeros_like_private(M, S_pr.shape[-1] if S_pr is not None else self.sk_pr.out_features)
            F_m = self._concat_fixed(M_sh, zeros_pr_s, M_pr)
            repr_m = self.backbone(F_m).flatten(1)
            out_imu = self.head_sensor(repr_m)

        return out_skel, out_imu


# ---------- minimal usage ----------
# from focal_shared_latent_baseline import FOCALSharedLatentBaseline
# model = FOCALSharedLatentBaseline(
#     skeleton_input_dim=21, skeleton_output_dim=6,
#     sensor_in_channels=6, sensor_out_channels=6, sensor_length=426,
#     d_shared=128, d_private=64,
#     shared_out_channels=16, backbone_dim=8, num_classes=3,
#     use_norm_head=False, use_cosine_head=False
# )
# # synced pair:
# logits_sync, _ = model(x_skel, x_sensor, synced=True)
# # unpaired:
# logit_s, logit_m = model(x_skel, None, synced=False)
# logit_s, logit_m = model(None, x_sensor, synced=False)


# models/focal.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
class Shared_Backbone(nn.Module):
        """Tiny 1D conv + GELU + AdaptiveAvgPool backbone used by FOCAL."""
        def __init__(self, in_channels: int, shared_out_channels: int = 16, backbone_dim: int = 8):
            super().__init__()
            self.conv = nn.Conv1d(in_channels, shared_out_channels, kernel_size=3, padding=1)
            self.act  = nn.GELU()
            self.pool = nn.AdaptiveAvgPool1d(backbone_dim)

        def forward(self, x):              # x: (B,T,Cin)
            x = x.permute(0, 2, 1)         # (B,Cin,T)
            x = self.act(self.conv(x))     # (B,Cout,T)
            x = self.pool(x)               # (B,Cout,bdim)
            return x.permute(0, 2, 1)      # (B,bdim,Cout)

class Task_Head(nn.Module):
    """LayerNorm(optional) + Linear or Cosine-like Linear head."""
    def __init__(self, input_dim: int, num_classes: int,
                    use_norm: bool = False, use_cosine: bool = False):
        super().__init__()
        self.use_cosine = use_cosine
        self.norm = nn.LayerNorm(input_dim) if (use_norm or use_cosine) else None
        if use_cosine:
            self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
            nn.init.xavier_uniform_(self.weight)
            self.eps = 1e-8
        else:
            self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):              # x: (B,D)
        if self.norm is not None:
            x = self.norm(x)
        if self.use_cosine:
            x = torch.nn.functional.normalize(x, p=2, dim=1, eps=self.eps)
            w = torch.nn.functional.normalize(self.weight, p=2, dim=1, eps=self.eps)
            return x @ w.t()
        return self.fc(x)

class FOCALSharedLatent3(nn.Module):
    """
    3-modality FOCAL (walkway, insole, imu).
    • Per-modality encoder is expected to be outside; here we project + fuse.
    • We directly accept raw windows (B,T,D) and do linear projectors per modality.
    • Sync: fuse shared parts by mean, concat private parts, 1 head (shared label)
    • Async: no cross-mix, 3 heads (independent labels)
    Returns: (logits_walkway, logits_insole, logits_imu)
             In sync mode: all three returned (same head weights), trainer will pick how to use.
    """

    def __init__(
        self,
        *,
        w_in_dim: int = 2,     # walkway D
        i_in_dim: int = 13,    # insole  D
        m_in_dim: int = 24,    # imu     D
        d_shared: int = 128,
        d_private: int = 64,
        shared_out_ch: int = 16,
        backbone_dim: int = 8,
        num_classes: int = 2,
        synchronized: bool = True,   # True→“sync head”; False→3 heads
        use_norm_head: bool = False,
        use_cosine_head: bool = False,
    ):
        super().__init__()
        self.synchronized = synchronized
        Dsh, Dpr = d_shared, d_private

        # per-modality shared/private projectors
        self.w_sh, self.w_pr = nn.Linear(w_in_dim, Dsh), nn.Linear(w_in_dim, Dpr)
        self.i_sh, self.i_pr = nn.Linear(i_in_dim, Dsh), nn.Linear(i_in_dim, Dpr)
        self.m_sh, self.m_pr = nn.Linear(m_in_dim, Dsh), nn.Linear(m_in_dim, Dpr)

        # shared backbone after concat([mean(shared)], priv_w, priv_i, priv_m)
        in_ch_backbone = Dsh + 3*Dpr
        self.backbone = Shared_Backbone(
            in_channels=in_ch_backbone,
            shared_out_channels=shared_out_ch,
            backbone_dim=backbone_dim
        )
        feat_dim = backbone_dim * shared_out_ch

        # heads
        if synchronized:
            self.head = Task_Head(feat_dim, num_classes, use_norm_head, use_cosine_head)
            self.head_w = self.head_i = self.head_m = self.head  # alias for uniform return
        else:
            self.head_w = Task_Head(feat_dim, num_classes, use_norm_head, use_cosine_head)
            self.head_i = Task_Head(feat_dim, num_classes, use_norm_head, use_cosine_head)
            self.head_m = Task_Head(feat_dim, num_classes, use_norm_head, use_cosine_head)

    @staticmethod
    def _cat_fixed(z_sh_mean, z_pr_w, z_pr_i, z_pr_m):
        return torch.cat([z_sh_mean, z_pr_w, z_pr_i, z_pr_m], dim=-1)  # (B,T, Dsh+3*Dpr)

    def forward(
        self,
        x_walk: Optional[torch.Tensor],   # (B,T,2)
        x_insole: Optional[torch.Tensor], # (B,T,13)
        x_imu: Optional[torch.Tensor],    # (B,T,24)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ---- project to shared / private (only for present streams) ----
        Wsh = self.w_sh(x_walk)   if x_walk   is not None else None
        Wpr = self.w_pr(x_walk)   if x_walk   is not None else None
        Ish = self.i_sh(x_insole) if x_insole is not None else None
        Ipr = self.i_pr(x_insole) if x_insole is not None else None
        Msh = self.m_sh(x_imu)    if x_imu    is not None else None
        Mpr = self.m_pr(x_imu)    if x_imu    is not None else None

        # pick a reference tensor for (B,T,device,dtype) without using `or` on tensors
        ref_in = next(t for t in (x_walk, x_insole, x_imu) if t is not None)
        B, T = ref_in.size(0), ref_in.size(1)
        device, dtype = ref_in.device, ref_in.dtype

        # dims for zero-fill
        Dpr = next(z.size(-1) for z in (Wpr, Ipr, Mpr) if z is not None)
        Dsh = next(z.size(-1) for z in (Wsh, Ish, Msh) if z is not None)

        def zzeros(TD: int) -> torch.Tensor:
            return torch.zeros(B, T, TD, device=device, dtype=dtype)

        # ---- SYNC: average available shared parts; keep all privates ----
        if self.synchronized:
            shared_list = [z for z in (Wsh, Ish, Msh) if z is not None]
            # if sync loader is correct, all three exist; still guard just in case
            z_sh_mean = torch.stack(shared_list, dim=0).mean(dim=0) if len(shared_list) > 0 else zzeros(Dsh)

            Wpr_f = Wpr if Wpr is not None else zzeros(Dpr)
            Ipr_f = Ipr if Ipr is not None else zzeros(Dpr)
            Mpr_f = Mpr if Mpr is not None else zzeros(Dpr)

            F = self._cat_fixed(z_sh_mean, Wpr_f, Ipr_f, Mpr_f)   # (B,T, Dsh+3*Dpr)
            rep = self.backbone(F).flatten(1)
            y  = self.head(rep)  # one shared head under sync
            # return 3 identical logits so trainer code path stays unchanged
            return y, y, y

        # ---- ASYNC: per-modality, no cross-mix; others' privates are zeros ----
        def head_one(zsh: Optional[torch.Tensor],
                    zpr: Optional[torch.Tensor],
                    which: str) -> Optional[torch.Tensor]:
            if (zsh is None) or (zpr is None):
                return None
            zW = zpr if which == "w" else zzeros(Dpr)
            zI = zpr if which == "i" else zzeros(Dpr)
            zM = zpr if which == "m" else zzeros(Dpr)
            F  = self._cat_fixed(zsh, zW, zI, zM)                 # (B,T, Dsh+3*Dpr)
            rep= self.backbone(F).flatten(1)
            if which == "w": return self.head_w(rep)
            if which == "i": return self.head_i(rep)
            return self.head_m(rep)

        lw = head_one(Wsh, Wpr, "w")
        li = head_one(Ish, Ipr, "i")
        lm = head_one(Msh, Mpr, "m")

        # never return None: use zero-logits placeholders for missing branches
        # (class dim from walkway head; in async we always have 3 separate heads)
        C = self.head_w.fc.out_features
        def safe(y: Optional[torch.Tensor]) -> torch.Tensor:
            return y if y is not None else torch.zeros(B, C, device=device, dtype=dtype)

        return safe(lw), safe(li), safe(lm)
