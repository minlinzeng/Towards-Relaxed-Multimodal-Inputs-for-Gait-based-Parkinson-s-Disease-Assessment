# feature_encoder_weargait.py
"""
Three encoders + ONE shared backbone. No fusion.
- Walkway  (B,64, 2): tiny MLP + depthwise Conv1D → out_ch
- Insole   (B,64,13): shallow Conv1D              → out_ch
- IMU      (B,64,24): shallow Conv1D              → out_ch
All three feed a SINGLE SharedBackbone (same channel width).
Heads:
- synchronized=True  → one shared head for all streams
- synchronized=False → three separate heads
forward(...) → (logits_walkway, logits_insole, logits_imu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Heads ----------------
class CosineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1, eps=self.eps)
        w = F.normalize(self.weight, p=2, dim=1, eps=self.eps)
        return (x @ w.t()).clamp(-1.0 + self.eps, 1.0 - self.eps)

class TaskHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, use_norm: bool = False, use_cosine: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if (use_norm or use_cosine) else None
        self.fc   = CosineLinear(input_dim, num_classes) if use_cosine else nn.Linear(input_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None: x = self.norm(x)
        return self.fc(x)

# -------------- Encoders ---------------
class WalkwayEncoder(nn.Module):
    """Very shallow: walkway (B,64,2) → (B,64,enc_out_ch)."""
    def __init__(self, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(2, out_ch, kernel_size=3, padding=1)
        self.act  = nn.GELU()
        self.ln   = nn.LayerNorm(out_ch)

    def forward(self, x):                 # x: (B,T=64,2)
        x = x.permute(0,2,1)              # (B,2,T)
        x = self.act(self.conv(x))        # (B,C,T)
        x = x.permute(0,2,1)              # (B,T,C)
        return self.ln(x)                 # (B,T,C)

class IMUEncoderShallow(nn.Module):
    """Shallow: IMU (B,64,24) → (B,64,enc_out_ch)."""
    def __init__(self, in_ch: int, out_ch: int, pool_len: int | None = None):
        super().__init__()
        self.pool_len = pool_len
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act  = nn.GELU()
        self.ln   = nn.LayerNorm(out_ch)
        self.pool = nn.AdaptiveAvgPool1d(pool_len) if pool_len else None

    def forward(self, x):                 # x: (B,T,24)
        x = x.permute(0,2,1)              # (B,24,T)
        x = self.act(self.conv(x))        # (B,C,T)
        if self.pool: x = self.pool(x)    # (B,C,T')
        x = x.permute(0,2,1)              # (B,T',C)
        return self.ln(x)

class InsoleEncoderDeep(nn.Module):
    """Deeper/richer: Insole (B,64,13) → (B,64,enc_out_ch) with 2 conv blocks + residual."""
    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int | None = None, pool_len: int | None = None):
        super().__init__()
        self.pool_len = pool_len
        h = hidden_ch or max(out_ch, 2*out_ch)  # widen first block

        # Block 1
        self.conv1 = nn.Conv1d(in_ch, h, kernel_size=5, padding=2)
        self.act1  = nn.GELU()
        self.ln1   = nn.LayerNorm(h)

        # Block 2 (project to out_ch)
        self.conv2 = nn.Conv1d(h, out_ch, kernel_size=3, padding=1)
        self.act2  = nn.GELU()
        self.ln2   = nn.LayerNorm(out_ch)

        # Residual projection (in case h != out_ch)
        self.skip  = nn.Conv1d(h, out_ch, kernel_size=1) if h != out_ch else nn.Identity()

        self.pool  = nn.AdaptiveAvgPool1d(pool_len) if pool_len else None

    def forward(self, x):                 # x: (B,T,13)
        x = x.permute(0,2,1)              # (B,13,T)
        h = self.act1(self.conv1(x))      # (B,h,T)
        y = self.conv2(h)                 # (B,out_ch,T)
        y = self.act2(y + (self.skip(h) if not isinstance(self.skip, nn.Identity) else h))
        if self.pool: y = self.pool(y)    # (B,out_ch,T')
        y = y.permute(0,2,1)              # (B,T',out_ch)
        y = self.ln2(y)
        return y

class SharedBackbone(nn.Module):
    """Single shallow Conv1D + adaptive pool for ALL streams. (B,T,out_ch) → (B,Bdim,Cbb)."""
    def __init__(self, in_ch: int, out_ch: int = 16, bdim: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act  = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(bdim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.conv(x.permute(0,2,1)))  # (B,out_ch,T)
        z = self.pool(z)                           # (B,out_ch,bdim)
        return z.permute(0,2,1)                    # (B,bdim,out_ch)

# --------------- Multimodal Models -----------------
class WearGaitThreeModal(nn.Module):
    def __init__(self, *, enc_out_ch=12, backbone_dim=8, shared_out_ch=16,
                 num_classes=2, use_norm=False, use_cosine=False,
                 synchronized=True, pool_len=None):
        super().__init__()
        self.enc_w = WalkwayEncoder(out_ch=enc_out_ch)                         # (B,64,2)
        self.enc_i = InsoleEncoderDeep(in_ch=13, out_ch=enc_out_ch,            # (B,64,13) → deeper
                                       hidden_ch=enc_out_ch*2, pool_len=pool_len)
        self.enc_m = IMUEncoderShallow(in_ch=24, out_ch=enc_out_ch,            # (B,64,24) → shallow
                                       pool_len=pool_len)

        self.backbone = SharedBackbone(in_ch=enc_out_ch,
                                       out_ch=shared_out_ch,
                                       bdim=backbone_dim)
        feat_dim = shared_out_ch * backbone_dim
        self.synchronized = synchronized

        if synchronized:
            shared = TaskHead(feat_dim, num_classes, use_norm=use_norm, use_cosine=use_cosine)
            self.head_w = self.head_i = self.head_m = shared
            self._shared_head = shared
        else:
            self.head_w = TaskHead(feat_dim, num_classes, use_norm=use_norm, use_cosine=use_cosine)
            self.head_i = TaskHead(feat_dim, num_classes, use_norm=use_norm, use_cosine=use_cosine)
            self.head_m = TaskHead(feat_dim, num_classes, use_norm=use_norm, use_cosine=use_cosine)
            self._shared_head = None
    
    def _forward_stream(self, x: torch.Tensor, enc: nn.Module) -> torch.Tensor:
        f = enc(x)                 # (B,T,enc_out_ch)
        r = self.backbone(f)       # (B,Bdim,shared_out_ch)
        return r.flatten(1)        # (B,feat_dim)

    def forward(self, x_walk: torch.Tensor, x_insole: torch.Tensor, x_imu: torch.Tensor):
        fw = self._forward_stream(x_walk, self.enc_w)
        fi = self._forward_stream(x_insole, self.enc_i)
        fm = self._forward_stream(x_imu, self.enc_m)

        lw = self.head_w(fw)
        li = self.head_i(fi)
        lm = self.head_m(fm)
        return lw, li, lm

        # --- private parameter groups (exclude shared backbone/head) ---
    def walkway_parameters(self):
        """
        Params unique to the walkway stream.
        - Always includes walkway encoder.
        - In async mode, also include its own head.
        - In sync mode, the head is shared → DO NOT include it here.
        """
        params = list(self.enc_w.parameters())
        if not self.synchronized:  # three independent heads
            params += list(self.head_w.parameters())
        return params

    def insole_parameters(self):
        """Params unique to the insole stream (encoder + head if async)."""
        params = list(self.enc_i.parameters())
        if not self.synchronized:
            params += list(self.head_i.parameters())
        return params

    def imu_parameters(self):
        """Params unique to the IMU stream (encoder + head if async)."""
        params = list(self.enc_m.parameters())
        if not self.synchronized:
            params += list(self.head_m.parameters())
        return params

    def get_shared_parameters(self):
        params = list(self.backbone.parameters())
        if self._shared_head is not None:
            params += list(self._shared_head.parameters())
        return params


# ---------------------------- Single modal Baselines ----------------------------
# ---------------------------- Utilities used by baselines ----------------------------

def _flatten_backbone(out: torch.Tensor) -> torch.Tensor:
    # out: (B, Bdim, C) -> (B, Bdim*C)
    return out.flatten(1)

def _shared_or_three_heads(feat_dim, num_classes, synchronized, use_norm, use_cosine):
    if synchronized:
        shared = TaskHead(feat_dim, num_classes, use_norm=use_norm, use_cosine=use_cosine)
        return shared, shared, shared, shared  # (shared, w, i, m) all refs to same module
    else:
        hw = TaskHead(feat_dim, num_classes, use_norm=use_norm, use_cosine=use_cosine)
        hi = TaskHead(feat_dim, num_classes, use_norm=use_norm, use_cosine=use_cosine)
        hm = TaskHead(feat_dim, num_classes, use_norm=use_norm, use_cosine=use_cosine)
        return None, hw, hi, hm

class EarlyFusion3(nn.Module):
    """
    Encodes each stream to the SAME temporal length and feature width,
    then stacks along feature/channel dim and runs ONE shared backbone.

    Fusion: concat along feature dim (T, Cw+Ci+Cm).
    Head(s): synchronized -> 1 shared head; async -> 3 separate heads but SAME fused repr.
    """
    def __init__(self, enc_out_ch, backbone_dim, shared_out_ch, num_classes,
                 use_norm=False, use_cosine=False, synchronized=True):
        super().__init__()
        self.synchronized = synchronized

        self.enc_w = WalkwayEncoder(out_ch=enc_out_ch)                  # (B,64,2)  -> (B,64,enc_out_ch)
        self.enc_i = InsoleEncoderDeep(in_ch=13, out_ch=enc_out_ch)     # (B,64,13) -> (B,64,enc_out_ch)
        self.enc_m = IMUEncoderShallow(in_ch=24, out_ch=enc_out_ch)     # (B,64,24) -> (B,64,enc_out_ch)

        in_ch = enc_out_ch * 3
        self.backbone = SharedBackbone(in_ch=in_ch, out_ch=shared_out_ch, bdim=backbone_dim)

        feat_dim = shared_out_ch * backbone_dim
        self._shared_head, self.head_w, self.head_i, self.head_m = _shared_or_three_heads(
            feat_dim, num_classes, synchronized, use_norm, use_cosine
        )

    def forward(self, xw, xi, xm):
        fw = self.enc_w(xw)  # (B,T,C)
        fi = self.enc_i(xi)
        fm = self.enc_m(xm)
        fused = torch.cat([fw, fi, fm], dim=-1)            # (B,T,Cw+Ci+Cm)
        repr_ = _flatten_backbone(self.backbone(fused))    # (B,feat_dim)

        if self.synchronized:
            logits = self._shared_head(repr_)
            return logits, logits, logits
        else:
            return self.head_w(repr_), self.head_i(repr_), self.head_m(repr_)

class LateFusion3(nn.Module):
    """
    Each stream: Encoder -> shared backbone (weights shared across streams),
    then fuse the *latent* vectors (here we use element-wise MEAN to keep dim).
    """
    def __init__(self, enc_out_ch, backbone_dim, shared_out_ch, num_classes,
                 use_norm=False, use_cosine=False, synchronized=True):
        super().__init__()
        self.synchronized = synchronized

        self.enc_w = WalkwayEncoder(out_ch=enc_out_ch)
        self.enc_i = InsoleEncoderDeep(in_ch=13, out_ch=enc_out_ch)
        self.enc_m = IMUEncoderShallow(in_ch=24, out_ch=enc_out_ch)

        # One shared backbone reused 3x
        self.backbone = SharedBackbone(in_ch=enc_out_ch, out_ch=shared_out_ch, bdim=backbone_dim)
        feat_dim = shared_out_ch * backbone_dim

        self._shared_head, self.head_w, self.head_i, self.head_m = _shared_or_three_heads(
            feat_dim, num_classes, synchronized, use_norm, use_cosine
        )

    def _branch(self, x, enc):
        return _flatten_backbone(self.backbone(enc(x)))  # (B,feat_dim)

    def forward(self, xw, xi, xm):
        rw = self._branch(xw, self.enc_w)
        ri = self._branch(xi, self.enc_i)
        rm = self._branch(xm, self.enc_m)

        fused = (rw + ri + rm) / 3.0  # keep dim; avoid huge concat
        if self.synchronized:
            logits = self._shared_head(fused)
            return logits, logits, logits
        else:
            return self.head_w(rw), self.head_i(ri), self.head_m(rm)

class SharedLatent3(nn.Module):
    """
    Encoders -> per-stream linear projection to shared channel width ->
    shared backbone per stream -> heads.
    For sync: we also return fused (mean) logits replicated for 3 outputs.
    """
    def __init__(self, enc_out_ch, proj_ch, backbone_dim, shared_out_ch, num_classes,
                 use_norm=False, use_cosine=False, synchronized=True):
        super().__init__()
        self.synchronized = synchronized

        self.enc_w = WalkwayEncoder(out_ch=enc_out_ch)
        self.enc_i = InsoleEncoderDeep(in_ch=13, out_ch=enc_out_ch)
        self.enc_m = IMUEncoderShallow(in_ch=24, out_ch=enc_out_ch)

        self.proj_w = nn.Linear(enc_out_ch, proj_ch)
        self.proj_i = nn.Linear(enc_out_ch, proj_ch)
        self.proj_m = nn.Linear(enc_out_ch, proj_ch)

        self.backbone = SharedBackbone(in_ch=proj_ch, out_ch=shared_out_ch, bdim=backbone_dim)
        feat_dim = shared_out_ch * backbone_dim

        self._shared_head, self.head_w, self.head_i, self.head_m = _shared_or_three_heads(
            feat_dim, num_classes, synchronized, use_norm, use_cosine
        )

    def _branch(self, x, enc, proj):
        lat = proj(enc(x))                         # (B,T,proj_ch)
        return _flatten_backbone(self.backbone(lat))

    def forward(self, xw, xi, xm):
        rw = self._branch(xw, self.enc_w, self.proj_w)
        ri = self._branch(xi, self.enc_i, self.proj_i)
        rm = self._branch(xm, self.enc_m, self.proj_m)

        out_w = self.head_w(rw)
        out_i = self.head_i(ri)
        out_m = self.head_m(rm)
        return out_w, out_i, out_m

class CheapCrossAttention(nn.Module):
    """
    Zero-parameter attention as in your 2-mod baseline; extended helper.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A,B: (B,T,d)
        sim = (A @ B.transpose(1, 2)) * self.scale      # (B,T,T)
        attn = torch.softmax(sim, dim=-1)
        return attn @ B                                  # (B,T,d)

class CheapXAttn3(nn.Module):
    """
    Encode each modality to same (T,d), then do pairwise cheap cross-attn,
    average the three cross-attended streams, and use shared backbone.

    Sync: single shared head on fused repr (replicated outputs).
    Async: three heads, each gets its own *self-attn-enriched* repr.
    """
    def __init__(self, enc_out_ch, backbone_dim, shared_out_ch, num_classes,
                 use_norm=False, use_cosine=False, synchronized=True):
        super().__init__()
        self.synchronized = synchronized
        d = enc_out_ch

        self.enc_w = WalkwayEncoder(out_ch=d)
        self.enc_i = InsoleEncoderDeep(in_ch=13, out_ch=d)
        self.enc_m = IMUEncoderShallow(in_ch=24, out_ch=d)

        self.xattn = CheapCrossAttention(dim=d)   # reused for all pairs
        self.backbone = SharedBackbone(in_ch=d, out_ch=shared_out_ch, bdim=backbone_dim)
        feat_dim = shared_out_ch * backbone_dim

        self._shared_head, self.head_w, self.head_i, self.head_m = _shared_or_three_heads(
            feat_dim, num_classes, synchronized, use_norm, use_cosine
        )

    def _repr(self, X: torch.Tensor) -> torch.Tensor:
        return _flatten_backbone(self.backbone(X))  # (B,feat_dim)

    def forward(self, xw, xi, xm):
        # encode to (B,T,d)
        W = self.enc_w(xw)
        I = self.enc_i(xi)
        M = self.enc_m(xm)

        # pairwise cheap x-attn
        W_i = self.xattn(W, I); I_w = self.xattn(I, W)
        W_m = self.xattn(W, M); M_w = self.xattn(M, W)
        I_m = self.xattn(I, M); M_i = self.xattn(M, I)

        # self-enriched per stream
        W_star = (W_i + W_m) * 0.5
        I_star = (I_w + I_m) * 0.5
        M_star = (M_w + M_i) * 0.5

        # Always output per-branch logits; eval will ensemble in SYNC
        lw = self.head_w(self._repr(W_star))
        li = self.head_i(self._repr(I_star))
        lm = self.head_m(self._repr(M_star))
        return lw, li, lm
