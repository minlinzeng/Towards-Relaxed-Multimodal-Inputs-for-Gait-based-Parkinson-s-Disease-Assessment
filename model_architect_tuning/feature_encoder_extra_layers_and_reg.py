import torch
import torch.nn as nn
import numpy as np
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
        cosine = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)
        return cosine

class SensorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, sensor_length=None, output_length=101):
        super().__init__()
        self.d2_sensor_length = sensor_length
        self.conv = nn.Conv1d(
            in_channels=in_channels,    # e.g. 6
            out_channels=out_channels,  # e.g. 6
            kernel_size=3, padding=1
        )
        self.bn   = nn.BatchNorm1d(out_channels, momentum=0.01)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(output_length)

    def forward(self, x):
        x = x.permute(0, 2, 1)           # → (B, in_channels, T)
        x = self.conv(x)                 # → (B, out_channels, T)
        x = self.bn(x)
        x = self.relu(x)
        if x.shape[2] == self.d2_sensor_length:
            x = self.pool(x)             # → (B, out_channels, output_length)
        return x.permute(0, 2, 1)         

class SkeletonMLP(nn.Module):
    """
    Processes skeleton (pose) data with an MLP to produce a fixed-length representation.
    Expected input: a flattened pose vector per frame, e.g., shape (101, 51)
    Input: shape (B, 101, 51) or (B, 101, 21)
    Output: shape (B, 65, 3) or (B, 101, 6)
    """
    def __init__(self, input_dim, output_dim):
        super(SkeletonMLP, self).__init__()
        # single per-frame projection — keeps your original time dimension
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, T, input_dim)
        return self.relu(self.ln1(self.fc1(x))) # (B,T,D_out)
    
class SharedBackbone(nn.Module):
    """
    Lighter SharedBackbone with exactly 2 Conv1D layers,
    each using `shared_out_channels` feature maps.
    We pool the time dimension down to `backbone_dim`.
    """
    def __init__(self, in_channels, shared_out_channels=8, backbone_dim=4, p_drop=0.05):
        super(SharedBackbone, self).__init__()

        # Block #1: conv from `in_channels` → `shared_out_channels`
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=shared_out_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(shared_out_channels, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p_drop)
        )

        # Block #2: conv from `shared_out_channels` → `shared_out_channels`
        self.block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=shared_out_channels,
                out_channels=shared_out_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(shared_out_channels, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p_drop)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(shared_out_channels, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p_drop)
        )

        # Finally, pool time → `backbone_dim`
        self.pool = nn.AdaptiveAvgPool1d(backbone_dim)

    def forward(self, x):
        # x comes in as (B, T, in_channels)
        x = x.permute(0, 2, 1)   # → (B, in_channels, T)
        x = self.block1(x)      # → (B, shared_out_channels, T)
        x = self.block2(x)      # → (B, shared_out_channels, T)
        x = self.block3(x)
        x = self.pool(x)        # → (B, shared_out_channels, backbone_dim)
        return x.permute(0, 2, 1)  # → (B, backbone_dim, shared_out_channels)

# -- Task Heads --
class TaskHead(nn.Module):
    """
    Final classification/regression head for a given task.
    Takes the backbone output and produces predictions.
    input: (B, 8*16)
    output: (B, 8, 3)
    """
    def __init__(self, input_dim, num_classes, use_norm: bool = False, use_cosine: bool=False, dropout_p: float = 0.05):
        super(TaskHead, self).__init__()
        self.use_cosine = use_cosine
        self.drop      = nn.Dropout(p=dropout_p)

        if use_cosine:
            self.norm = nn.LayerNorm(input_dim)
            self.fc   = CosineLinear(input_dim, num_classes)
        elif use_norm:
            self.norm = nn.LayerNorm(input_dim)
            self.fc   = nn.Linear(input_dim, num_classes)
        else:
            self.norm   = None
            self.fc      = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.drop(x)
        if self.use_cosine:
            x = self.norm(x)
        return self.fc(x)

class MultiModalMultiTaskModel(nn.Module):
    """
    Processes two modalities (skeleton + sensor) via:
      - SkeletonMLP → (B, T_s, C) per-frame skeleton features
      - SensorEncoder → (B, T_g, C) per-frame sensor features
      - SharedBackbone (in_channels = C) → (B, backbone_dim, shared_out_channels)
      - Flatten → (B, D = backbone_dim*shared_out_channels)
      - Two TaskHeads (or one if sync): each (B, D) → (B, num_classes)

    All dimensions are controlled by MODALITY_PARAMS:
      skeleton_output_dim == sensor_out_channels == C
      shared_out_channels, backbone_dim chosen accordingly
      taskhead_input_dim = shared_out_channels * backbone_dim
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
            input_dim  = skeleton_input_dim,
            output_dim = skeleton_output_dim
        )
        self.sensor_encoder   = SensorEncoder(
            in_channels    = sensor_in_channels,
            out_channels   = sensor_out_channels,
            sensor_length  = sensor_length,
            output_length  = sensor_length  # keep original temporal length
        )

        # Backbone expects in_channels = C (same for both modalities)
        self.backbone = SharedBackbone(
            in_channels         = sensor_out_channels,  # equals skeleton_output_dim
            shared_out_channels = shared_out_channels,
            backbone_dim        = backbone_dim,
            p_drop              = 0.05
        )

        self.synchronized_loading = synchronized_loading

        # Task heads
        if synchronized_loading:
            # Single head for both modalities (shared head)
            self.task_head_shared = TaskHead(
                input_dim   = taskhead_input_dim,
                num_classes = num_classes,
                use_norm    = use_norm,
                use_cosine  = use_cosine,
                dropout_p   = 0.05
            )
        else:
            # Two separate heads (one per modality)
            self.task_head_skel   = TaskHead(
                input_dim   = taskhead_input_dim,
                num_classes = num_classes,
                use_norm    = use_norm,
                use_cosine  = use_cosine,
                dropout_p   = 0.05
            )
            self.task_head_sensor = TaskHead(
                input_dim   = taskhead_input_dim,
                num_classes = num_classes,
                use_norm    = use_norm,
                use_cosine  = use_cosine,
                dropout_p   = 0.05
            )

        # Flags for single‐modality inference
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
        skel_feat = self.skeleton_encoder(x_skel)    # → (B, T_skel, C)
        sens_feat = self.sensor_encoder(x_sensor)    # → (B, T_sens, C)

        # 2) Shared backbone on per-frame features
        skel_repr = self.backbone(skel_feat).flatten(1)  # → (B, D = shared_out*backbone_dim)
        sens_repr = self.backbone(sens_feat).flatten(1)  # → (B, D)

        # 3) Allow single‐modality inference if desired
        if self.use_skeleton_only:
            return self.task_head_skel(skel_repr), None
        if self.use_sensor_only:
            return None, self.task_head_sensor(sens_repr)

        # 4) Multimodal output
        if self.synchronized_loading:
            # Both modalities share the same head
            logits_skel = self.task_head_shared(skel_repr)  # → (B, num_classes)
            logits_sens = self.task_head_shared(sens_repr)  # → (B, num_classes)
            return logits_skel, logits_sens
        else:
            # Two separate heads
            logits_skel = self.task_head_skel(skel_repr)     # → (B, num_classes)
            logits_sens = self.task_head_sensor(sens_repr)   # → (B, num_classes)
            return logits_skel, logits_sens

    def get_shared_parameters(self):
        """
        Return the list of parameters that are truly 'shared' by the gradient method:
          - In async mode: just the backbone
          - In sync mode: backbone + shared head
        """
        shared = list(self.backbone.parameters())
        if self.synchronized_loading:
            shared += list(self.task_head_shared.parameters())
        return shared
    
        
class SensorModalityModel(nn.Module):
    """
    Processes only one modality (skeleton or sensor) using the corresponding encoder,
    a shared backbone, and a task head.
    """
    def __init__(self, sensor_in_channels, sensor_out_channels, sensor_length, shared_out_channels, backbone_dim, taskhead_input_dim, num_classes, use_norm: bool = True):
        super(SensorModalityModel, self).__init__()
        self.encoder = SensorEncoder(in_channels=sensor_in_channels, out_channels=sensor_out_channels, sensor_length=sensor_length)
        self.backbone = SharedBackbone(in_channels=sensor_out_channels, shared_out_channels=shared_out_channels, backbone_dim=backbone_dim)
        self.task_head = TaskHead(input_dim=taskhead_input_dim, num_classes=num_classes, use_norm=use_norm)
    
    def forward(self, x):
        feat = self.encoder(x)   # => (B, 65, 3) / (B, 101, 6)
        repr = self.backbone(feat)  # => (B, 8, 16) / (B, 8, 16)
        repr = repr.flatten(start_dim=1)  # Flatten to (B, 8*16)
        pred = self.task_head(repr)
        return pred
        
        
class SkelModalityModel(nn.Module):
    """
    Processes only one modality (skeleton or sensor) using the corresponding encoder,
    a shared backbone, and a task head.
    """
    def __init__(self, skeleton_input_dim, skeleton_output_dim, sensor_out_channels, shared_out_channels, backbone_dim, taskhead_input_dim, num_classes=3, use_norm: bool = True):
        super(SkelModalityModel, self).__init__()
        self.encoder = SkeletonMLP(input_dim=skeleton_input_dim, output_dim=skeleton_output_dim)
        self.backbone = SharedBackbone(in_channels=sensor_out_channels, shared_out_channels=shared_out_channels, backbone_dim=backbone_dim)
        self.task_head = TaskHead(input_dim=taskhead_input_dim, num_classes=num_classes, use_norm=use_norm)
    
    def forward(self, x):
        feat = self.encoder(x)   # => (B, 65, 3) / (B, 101, 6)
        repr = self.backbone(feat)  # => (B, 8, 16)
        repr = repr.flatten(start_dim=1)  # Flatten to (B, 8*16)
        pred = self.task_head(repr)
        return pred
    
  
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
        if synchronized_loading:
            self.head = nn.Linear(feature_dim, num_classes)
        else:
            self.head_skel  = nn.Linear(feature_dim, num_classes)
            self.head_sens  = nn.Linear(feature_dim, num_classes)

    def forward(self, x_skel, x_sens):
        sk_lat = self.proj_skel(self.skel_enc(x_skel))   # (B, T, shared_out)
        se_lat = self.proj_sens(self.sens_enc(x_sens))   # (B, T, shared_out)
        fused  = sk_lat + se_lat                         # element-wise fusion
        repr   = self.backbone(fused).flatten(1)

        if self.synchronized_loading:
            return self.head(repr)
        else:
            return self.head_skel(repr), self.head_sens(repr)


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
