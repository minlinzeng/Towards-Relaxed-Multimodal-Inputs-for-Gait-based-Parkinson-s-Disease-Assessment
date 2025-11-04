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
            output_dim = skeleton_output_dim   # e.g. 6 for “turn”
        )
        self.sensor_encoder   = SensorEncoder(
            in_channels   = sensor_in_channels,   # e.g. 6
            out_channels  = sensor_out_channels,  # e.g. 6
            sensor_length = sensor_length         # e.g. 426
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
        use_norm: bool = True  # if using LDAM in sensor-only
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
            use_cosine  = False
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
        use_norm: bool = True  # if using LDAM in skel-only
    ):
        super().__init__()
        self.encoder  = SkeletonMLP(
            input_dim  = skeleton_input_dim,
            output_dim = skeleton_output_dim
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
            use_cosine  = False
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
        self.skel_enc = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.sens_enc = SensorEncoder(
            in_channels=sensor_in_channels,
            out_channels=sensor_out_channels,
            sensor_length=sensor_length
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
        synchronized_loading: bool = False
    ):
        super().__init__()
        self.synchronized_loading = synchronized_loading

        self.skel_enc = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.sens_enc = SensorEncoder(
            in_channels=sensor_in_channels,
            out_channels=sensor_out_channels,
            sensor_length=sensor_length
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
        synchronized_loading: bool = False
    ):
        super().__init__()
        self.synchronized_loading = synchronized_loading

        self.skel_enc  = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.sens_enc  = SensorEncoder(
            in_channels=sensor_in_channels,
            out_channels=sensor_out_channels,
            sensor_length=sensor_length
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
        synchronized_loading: bool = False
    ):
        super().__init__()
        assert skeleton_output_dim == sensor_out_channels, \
            "For cross-attention we need same feature dim on both modalities"

        self.synchronized_loading = synchronized_loading

        # modality encoders
        self.skel_enc = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.sens_enc = SensorEncoder(
            in_channels=sensor_in_channels,
            out_channels=sensor_out_channels,
            sensor_length=sensor_length
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

