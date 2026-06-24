# feature_encoder.py (original “shallow, regularization‐free” version)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineLinear(nn.Module):
    """
    x:   (B, D) features
    W:   (C, D) learnable weight vectors
    out: (B, C) cosine similarities
    """
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize features and weights
        x_norm = F.normalize(x, p=2, dim=1, eps=self.eps)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=self.eps)
        cosine = x_norm @ w_norm.t()
        return cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)


class SensorEncoder(nn.Module):
    """
    Processes sensor data with a single Conv1D → (optional pooling).
    Input:  (B, T_in, C_in)      e.g. (B,65,3) or (B,426,6)
    Output: (B, T_out, C_out)    e.g. (B,65,3) or (B,101,6)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sensor_length: int = None,
        output_length: int = 101
    ):
        super().__init__()
        self.d2_sensor_length = sensor_length
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool = nn.AdaptiveAvgPool1d(output_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, C_in)
        x = x.permute(0, 2, 1)         # → (B, C_in, T_in)
        x = self.conv1d(x)            # → (B, C_out, T_in)
        if x.shape[2] == self.d2_sensor_length:
            x = self.pool(x)          # → (B, C_out, output_length)
        x = x.permute(0, 2, 1)         # → (B, T_out, C_out)
        return x


class SkeletonMLP(nn.Module):
    """
    Processes skeleton (pose) data with a single Linear → LayerNorm → ReLU.
    Input:  (B, T, D_in)    e.g. (B,101,51) or (B,101,21)
    Output: (B, T, D_out)   e.g. (B,101,3)   or (B,101,6)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, output_dim)
        self.ln1  = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        h = self.fc1(x)           # → (B, T, output_dim)
        h = self.ln1(h)           # → (B, T, output_dim)
        return self.relu(h)       # → (B, T, output_dim)


class SharedBackbone(nn.Module):
    """
    A shallow 2‐block Conv1D backbone (no dropout, no BatchNorm).
    Input:  (B, T, in_channels)   e.g. (B,101,6)
    Output: (B, backbone_dim, shared_out_channels) e.g. (B,8,16)
    """
    def __init__(
        self,
        in_channels: int,
        shared_out_channels: int = 16,
        backbone_dim: int = 8
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=shared_out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu  = nn.ReLU()
        self.pool  = nn.AdaptiveAvgPool1d(backbone_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_channels)
        x = x.permute(0, 2, 1)   # → (B, in_channels, T)
        x = self.conv1d(x)      # → (B, shared_out_channels, T)
        x = self.relu(x)
        x = self.pool(x)        # → (B, shared_out_channels, backbone_dim)
        return x.permute(0, 2, 1)# → (B, backbone_dim, shared_out_channels)


class TaskHead(nn.Module):
    """
    Final classification head. Depending on 'use_norm' / 'use_cosine', 
    uses either LayerNorm+Linear (LDAM), LayerNorm+CosineLinear (GCL), or
    plain Linear (CE).
    Always expects input of shape (B, D), outputs (B, num_classes).
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_norm: bool = False,
        use_cosine: bool = False
    ):
        super().__init__()
        self.use_cosine = use_cosine

        if use_cosine:
            # for GCL: normalize features before cosine similarity
            self.norm = nn.LayerNorm(input_dim)
            self.fc   = CosineLinear(input_dim, num_classes)
        elif use_norm:
            # for LDAM: just apply LayerNorm before a normal Linear
            self.norm = nn.LayerNorm(input_dim)
            self.fc   = nn.Linear(input_dim, num_classes)
        else:
            # for plain CE: no normalization
            self.norm = None
            self.fc   = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        if self.use_cosine or self.norm is not None:
            x = self.norm(x)
        return self.fc(x)  # → (B, num_classes)


class MultiModalMultiTaskModel(nn.Module):
    """
    Multi-task model that processes both skeleton + sensor:
      • Skeleton path: SkeletonMLP → SharedBackbone → TaskHead_skel
      • Sensor path:   SensorEncoder → SharedBackbone → TaskHead_sensor
      • If synchronized_loading=True, share a single TaskHead for both reprs
    No dropout, no BatchNorm—i.e., regularization‐free.
    """
    def __init__(
        self,
        skeleton_input_dim: int,
        skeleton_output_dim: int,
        sensor_in_channels: int,
        sensor_out_channels: int,
        sensor_length: int,
        shared_out_channels: int,
        backbone_dim: int,
        taskhead_input_dim: int,
        num_classes: int,
        use_norm: bool = False,
        use_cosine: bool = False,
        synchronized_loading: bool = False
    ):
        super().__init__()

        # Encoders
        self.skeleton_encoder = SkeletonMLP(
            input_dim  = skeleton_input_dim,   # e.g. 21 for “turn”
            output_dim = skeleton_output_dim,  # e.g. 6 for “turn”
        )
        self.sensor_encoder   = SensorEncoder(
            in_channels   = sensor_in_channels,   # e.g. 6
            out_channels  = sensor_out_channels,  # e.g. 6
            sensor_length = sensor_length,        # e.g. 426
        )

        # Shared backbone (shallow, reg-free)
        self.backbone = SharedBackbone(
            in_channels         = sensor_out_channels,  # must match skeleton_output_dim too
            shared_out_channels = shared_out_channels,  # e.g. 16
            backbone_dim        = backbone_dim          # e.g. 8
        )

        self.synchronized_loading = synchronized_loading

        # Task heads
        if synchronized_loading:
            # one shared head for both skeleton & sensor
            self.task_head_shared = TaskHead(
                input_dim   = taskhead_input_dim,  # = shared_out_channels * backbone_dim, e.g. 16×8=128
                num_classes = num_classes,         # e.g. 3
                use_norm    = use_norm,
                use_cosine  = use_cosine
            )
        else:
            # two separate heads
            self.task_head_skel   = TaskHead(
                input_dim   = taskhead_input_dim,
                num_classes = num_classes,
                use_norm    = use_norm,
                use_cosine  = use_cosine
            )
            self.task_head_sensor = TaskHead(
                input_dim   = taskhead_input_dim,
                num_classes = num_classes,
                use_norm    = use_norm,
                use_cosine  = use_cosine
            )

        # Flags for single-modality inference
        self.use_skeleton_only = False
        self.use_sensor_only   = False

    def forward(
        self,
        x_skel: torch.Tensor,
        x_sensor: torch.Tensor
    ):
        """
        x_skel:   (B, T_skel, D_skel_in)
        x_sensor: (B, T_sens, C_sens_in)
        """
        # 1) Encode each modality
        skel_feat = self.skeleton_encoder(x_skel)      # → (B, T_skel, C) 
        sens_feat = self.sensor_encoder(x_sensor)      # → (B, T_sens, C)

        # 2) Shared backbone on each repr
        #    SharedBackbone expects (B, T, C) → permutes internally
        skel_repr = self.backbone(skel_feat).flatten(1)  # → (B, C_out * backbone_dim)
        sens_repr = self.backbone(sens_feat).flatten(1)  # → (B, C_out * backbone_dim)

        # 3) Single⇒two‐head logic
        if self.use_skeleton_only:
            return self.task_head_skel(skel_repr), None
        if self.use_sensor_only:
            return None, self.task_head_sensor(sens_repr)

        # 4) Multimodal output
        if self.synchronized_loading:
            logits_skel = self.task_head_shared(skel_repr)
            logits_sens = self.task_head_shared(sens_repr)
            return logits_skel, logits_sens
        else:
            logits_skel = self.task_head_skel(skel_repr)
            logits_sens = self.task_head_sensor(sens_repr)
            return logits_skel, logits_sens

    def get_shared_parameters(self):
        """
        Return list of parameters “shared” by gradient‐methods:
          • Async mode: only the backbone’s weights
          • Sync mode: backbone + shared head
        """
        shared = list(self.backbone.parameters())
        if self.synchronized_loading:
            shared += list(self.task_head_shared.parameters())
        return shared


class SensorModalityModel(nn.Module):
    """
    Single‐modality sensor‐only model (no regularization).
    """
    def __init__(
        self,
        sensor_in_channels: int,
        sensor_out_channels: int,
        sensor_length: int,
        shared_out_channels: int,
        backbone_dim: int,
        taskhead_input_dim: int,
        num_classes: int,
        use_norm: bool = True,  # if using LDAM in sensor-only
        use_cosine: bool = False
    ):
        super().__init__()
        self.encoder  = SensorEncoder(
            in_channels   = sensor_in_channels,
            out_channels  = sensor_out_channels,
            sensor_length = sensor_length
        )
        self.backbone = SharedBackbone(
            in_channels         = sensor_out_channels,
            shared_out_channels = shared_out_channels,
            backbone_dim        = backbone_dim
        )
        self.task_head = TaskHead(
            input_dim   = taskhead_input_dim,
            num_classes = num_classes,
            use_norm    = use_norm,
            use_cosine  = use_cosine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)              # → (B, T, C)
        repr = self.backbone(feat)          # → (B, backbone_dim, C_out)
        repr = repr.flatten(start_dim=1)    # → (B, D = C_out * backbone_dim)
        return self.task_head(repr)         # → (B, num_classes)


class SkelModalityModel(nn.Module):
    """
    Single‐modality skeleton‐only model (no regularization).
    """
    def __init__(
        self,
        skeleton_input_dim: int,
        skeleton_output_dim: int,
        sensor_out_channels: int,  # passed as “in_channels” to backbone
        shared_out_channels: int,
        backbone_dim: int,
        taskhead_input_dim: int,
        num_classes: int,
        use_norm: bool = True,  # if using LDAM in skel-only
        use_cosine: bool = False
    ):
        super().__init__()
        self.encoder  = SkeletonMLP(
            input_dim  = skeleton_input_dim,
            output_dim = skeleton_output_dim,
        )
        self.backbone = SharedBackbone(
            in_channels         = sensor_out_channels,  # must match skeleton_output_dim
            shared_out_channels = shared_out_channels,
            backbone_dim        = backbone_dim
        )
        self.task_head = TaskHead(
            input_dim   = taskhead_input_dim,
            num_classes = num_classes,
            use_norm    = use_norm,
            use_cosine  = use_cosine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)               # → (B, T, C)
        repr = self.backbone(feat)           # → (B, backbone_dim, C_out)
        repr = repr.flatten(start_dim=1)     # → (B, D = C_out * backbone_dim)
        return self.task_head(repr)          # → (B, num_classes)

# ——— EARLY FUSION ———
class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        skeleton_input_dim: int,
        skeleton_output_dim: int,
        sensor_in_channels: int,
        sensor_out_channels: int,
        sensor_length: int,
        shared_out_channels: int,
        backbone_dim: int,
        num_classes: int,
        synchronized_loading: bool = False):
        
        super().__init__()
        self.synchronized_loading = synchronized_loading

        # raw‐input encoders
        self.skel_enc = SkeletonMLP(
            skeleton_input_dim,
            skeleton_output_dim,
        )
        self.sens_enc = SensorEncoder(
            in_channels=sensor_in_channels,
            out_channels=sensor_out_channels,
            sensor_length=sensor_length,
        )
        # shared backbone after concatenation
        self.backbone = SharedBackbone(
            in_channels = skeleton_output_dim + sensor_out_channels,
            shared_out_channels = shared_out_channels,
            backbone_dim = backbone_dim
        )

        feature_dim = backbone_dim * shared_out_channels
        if synchronized_loading:
            # single head for the shared label
            self.head = nn.Linear(feature_dim, num_classes)
        else:
            # two task heads
            self.head_skel  = nn.Linear(feature_dim, num_classes)
            self.head_sens  = nn.Linear(feature_dim, num_classes)

    def forward(self, x_skel, x_sens):
        # stack raw features along channel dim
        sk_feat = self.skel_enc(x_skel)    # (B, T, C1)
        se_feat = self.sens_enc(x_sens)    # (B, T, C2)
        fused_in = torch.cat([sk_feat, se_feat], dim=-1)

        repr = self.backbone(fused_in).flatten(1)
        if self.synchronized_loading:
            return self.head(repr)
        else:
            return self.head_skel(repr), self.head_sens(repr)

# ——— LATE FUSION ———
class LateFusionModel(nn.Module):
    def __init__(
        self,
        skeleton_input_dim: int,
        skeleton_output_dim: int,
        sensor_in_channels: int,
        sensor_out_channels: int,
        sensor_length: int,
        shared_out_channels: int,
        backbone_dim: int,
        num_classes: int,
        synchronized_loading: bool = False,
    ):
        super().__init__()
        self.synchronized_loading = synchronized_loading

        self.skel_enc = SkeletonMLP(
            skeleton_input_dim,
            skeleton_output_dim,
        )
        self.sens_enc = SensorEncoder(
            in_channels=sensor_in_channels,
            out_channels=sensor_out_channels,
            sensor_length=sensor_length,
        )
        self.backbone = SharedBackbone(
            in_channels = skeleton_output_dim,
            shared_out_channels = shared_out_channels,
            backbone_dim = backbone_dim
        )

        branch_dim = backbone_dim * shared_out_channels
        fused_dim  = branch_dim * 2

        if synchronized_loading:
            self.head = nn.Linear(fused_dim, num_classes)
        else:
            self.head_skel  = nn.Linear(fused_dim, num_classes)
            self.head_sens  = nn.Linear(fused_dim, num_classes)

    def forward(self, x_skel, x_sens):
        sk_repr = self.backbone(self.skel_enc(x_skel)).flatten(1)
        se_repr = self.backbone(self.sens_enc(x_sens)).flatten(1)
        fused   = torch.cat([sk_repr, se_repr], dim=1)

        if self.synchronized_loading:
            return self.head(fused)
        else:
            return self.head_skel(fused), self.head_sens(fused)

# ——— SHARED‐LATENT FUSION ———
class ShareLatentModel(nn.Module):
    def __init__(
        self,
        skeleton_input_dim: int,
        skeleton_output_dim: int,
        sensor_in_channels: int,
        sensor_out_channels: int,
        sensor_length: int,
        shared_out_channels: int,
        backbone_dim: int,
        taskhead_input_dim: int,
        num_classes: int,
        synchronized_loading: bool = False,
    ):
        super().__init__()
        self.synchronized_loading = synchronized_loading

        self.skel_enc  = SkeletonMLP(
            skeleton_input_dim,
            skeleton_output_dim,
        )
        self.sens_enc  = SensorEncoder(
            in_channels=sensor_in_channels,
            out_channels=sensor_out_channels,
            sensor_length=sensor_length,
        )
        # project each modality into common latent
        self.proj_skel = nn.Linear(skeleton_output_dim, shared_out_channels)
        self.proj_sens = nn.Linear(sensor_out_channels, shared_out_channels)

        self.backbone = SharedBackbone(
            in_channels = shared_out_channels,
            shared_out_channels = shared_out_channels,
            backbone_dim = backbone_dim
        )

        feature_dim = backbone_dim * shared_out_channels
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x_skel, x_sens):
        sk_lat = self.proj_skel(self.skel_enc(x_skel))   # (B, T, shared_out)
        se_lat = self.proj_sens(self.sens_enc(x_sens))   # (B, T, shared_out)
        # fused  = sk_lat + se_lat                         # element-wise fusion
        # repr   = self.backbone(fused).flatten(1)
        sk_repr = self.backbone(sk_lat).flatten(1)  # (B, feature_dim)
        se_repr = self.backbone(se_lat).flatten(1)  # (B, feature_dim)
        
        logit_sk = self.head(sk_repr)  # (B, num_classes)
        logit_se = self.head(se_repr)  # (B, num_classes)
        # logits = self.head(repr)
        return logit_sk, logit_se


class CheapCrossAttention(nn.Module):
    """
    Symmetric zero-parameter cross-attention that *fuses* two same-dimensional sequences.

    Inputs
        S : (B, T, d)   skeleton latent sequence
        G : (B, T, d)   sensor   latent sequence
    Returns
        fused : (B, T, d)  element-wise average of S* and G*
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5   # 1 / sqrt(d)

    def forward(self, S: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        # S, G shape = (B, T, d)
        # 1) compute pairwise similarity
        sim_sg = (S @ G.transpose(1, 2)) * self.scale    # (B, T, T)
        sim_gs = sim_sg.transpose(1, 2)                  # (B, T, T)

        # 2) attention weights
        attn_sg = torch.softmax(sim_sg, dim=-1)          # (B, T, T)
        attn_gs = torch.softmax(sim_gs, dim=-1)          # (B, T, T)

        # 3) cross-attended sequences
        S_star = attn_sg @ G                             # (B, T, d)
        G_star = attn_gs @ S                             # (B, T, d)

        # 4) fuse them (requires same T and d)
        fused = (S_star + G_star) * 0.5                  # (B, T, d)
        return fused


class CheapXAttnModel(nn.Module):
    """
    • Encodes each modality separately  
    • Fuses via symmetric cross-attention + element-wise average  
    • Shared backbone → two task heads
    """
    def __init__(
        self,
        skeleton_input_dim: int,
        skeleton_output_dim: int,
        sensor_in_channels: int,
        sensor_out_channels: int,
        sensor_length: int,
        shared_out_channels: int,
        backbone_dim: int,
        num_classes: int,
        synchronized_loading: bool = False,
    ):
        super().__init__()
        assert skeleton_output_dim == sensor_out_channels, \
            "For cross-attention we need same feature dim on both modalities"

        self.synchronized_loading = synchronized_loading

        # modality encoders
        self.skel_enc = SkeletonMLP(
            skeleton_input_dim,
            skeleton_output_dim,
        )
        self.sens_enc = SensorEncoder(
            in_channels=sensor_in_channels,
            out_channels=sensor_out_channels,
            sensor_length=sensor_length,
        )

        # cross-attention fusion block
        self.cross_attn = CheapCrossAttention(dim=skeleton_output_dim)

        # shared backbone
        self.backbone = SharedBackbone(
            in_channels = skeleton_output_dim,
            shared_out_channels = shared_out_channels,
            backbone_dim = backbone_dim
        )

        feature_dim = backbone_dim * shared_out_channels
        if synchronized_loading:
            # single-task head
            self.head = nn.Linear(feature_dim, num_classes)
        else:
            # two heads for two different labels
            self.head_skel = nn.Linear(feature_dim, num_classes)
            self.head_sens = nn.Linear(feature_dim, num_classes)

    def forward(self, x_skel: torch.Tensor, x_sens: torch.Tensor):
        # 1) encode
        sk_feat = self.skel_enc(x_skel)   # (B, T, d)
        se_feat = self.sens_enc(x_sens)   # (B, T, d)

        # 2) fuse via cross-attention
        fused_feat = self.cross_attn(sk_feat, se_feat)  # (B, T, d)

        # 3) shared backbone + flatten
        repr = self.backbone(fused_feat).flatten(1)     # (B, feature_dim)

        # 4) heads
        if self.synchronized_loading:
            return self.head(repr)                      # (B, num_classes)
        else:
            return self.head_skel(repr), self.head_sens(repr)

# ==================== WearGait Models ====================
# -------------- WearGait Encoders ---------------
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

class WearGaitSharedBackbone(nn.Module):
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

        self.backbone = WearGaitSharedBackbone(in_ch=enc_out_ch,
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
        self.backbone = WearGaitSharedBackbone(in_ch=in_ch, out_ch=shared_out_ch, bdim=backbone_dim)

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
        self.backbone = WearGaitSharedBackbone(in_ch=enc_out_ch, out_ch=shared_out_ch, bdim=backbone_dim)
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

        self.backbone = WearGaitSharedBackbone(in_ch=proj_ch, out_ch=shared_out_ch, bdim=backbone_dim)
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

class WearGaitCheapCrossAttention(nn.Module):
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

        self.xattn = WearGaitCheapCrossAttention(dim=d)   # reused for all pairs
        self.backbone = WearGaitSharedBackbone(in_ch=d, out_ch=shared_out_ch, bdim=backbone_dim)
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
