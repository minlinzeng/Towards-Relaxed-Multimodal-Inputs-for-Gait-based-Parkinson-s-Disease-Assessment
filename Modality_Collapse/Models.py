import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, sensor_length, output_length=101):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(output_length)
        self.sensor_length = sensor_length
    def forward(self, x):
        x = x.permute(0,2,1)                   # (B, C_in, T)
        x = self.conv(x)                       # (B, C_out, T)
        if x.shape[2] == self.sensor_length:
            x = self.pool(x)                   # (B, C_out, output_length)
        return x.permute(0,2,1)                # (B, T_out, C_out)

class SkeletonMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.fc(x)                         # (B, T, output_dim)
        return self.relu(self.ln(h))           # (B, T, output_dim)

class SharedBackbone(nn.Module):
    def __init__(self, in_channels, shared_out=16, backbone_dim=8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, shared_out, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(backbone_dim)
    def forward(self, x):
        x = x.permute(0,2,1)                   # (B, C, T)
        x = self.relu(self.conv(x))            # (B, shared_out, T)
        x = self.pool(x)                       # (B, shared_out, backbone_dim)
        return x.permute(0,2,1)                # (B, backbone_dim, shared_out)

class TaskHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)                      # (B, num_classes)

class MultiModalMultiTaskModel(nn.Module):
    def __init__(self, skeleton_input_dim, skeleton_output_dim,
                 sensor_in_channels, sensor_out_channels, sensor_length,
                 shared_out, backbone_dim, taskhead_dim, num_classes,
                 synchronized_loading=False):
        super().__init__()
        self.skel_enc = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.sens_enc = SensorEncoder(sensor_in_channels, sensor_out_channels, sensor_length)
        self.backbone = SharedBackbone(sensor_out_channels, shared_out, backbone_dim)
        self.sync = synchronized_loading
        if self.sync:
            self.head_shared = TaskHead(taskhead_dim, num_classes)
        else:
            self.head_skel = TaskHead(taskhead_dim, num_classes)
            self.head_sens = TaskHead(taskhead_dim, num_classes)
        self.use_skel_only = False
        self.use_sens_only = False

    def forward(self, x_skel, x_sens):
        sk = self.backbone(self.skel_enc(x_skel)).flatten(1)
        se = self.backbone(self.sens_enc(x_sens)).flatten(1)
        if self.use_skel_only:
            return self.head_skel(sk), None
        if self.use_sens_only:
            return None, self.head_sens(se)
        if self.sync:
            return self.head_shared(sk), self.head_shared(se)
        return self.head_skel(sk), self.head_sens(se)

    def get_shared_parameters(self):
        params = list(self.backbone.parameters())
        if self.sync:
            params += list(self.head_shared.parameters())
        return params

class EarlyFusionModel(nn.Module):
    def __init__(self, skeleton_input_dim, skeleton_output_dim,
                 sensor_in_channels, sensor_out_channels, sensor_length,
                 shared_out, backbone_dim, num_classes, synchronized_loading=False):
        super().__init__()
        self.sync = synchronized_loading
        self.skel_enc = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.sens_enc = SensorEncoder(sensor_in_channels, sensor_out_channels, sensor_length)
        self.backbone = SharedBackbone(skeleton_output_dim+sensor_out_channels, shared_out, backbone_dim)
        feat_dim = shared_out*backbone_dim
        self.head = nn.Linear(feat_dim, num_classes)
        self.use_skel_only = False
        self.use_sens_only = False

    def forward(self, x_skel, x_sens):
        sk = self.skel_enc(x_skel)
        se = self.sens_enc(x_sens)
        
        # mask before concatenation
        if self.use_skel_only:
            se = torch.zeros_like(se)
        elif self.use_sens_only:
            sk = torch.zeros_like(sk)
        
        fused = torch.cat([sk, se], dim=-1)
        rep = self.backbone(fused).flatten(1)
        return self.head(rep)

class LateFusionModel(nn.Module):
    def __init__(self, skeleton_input_dim, skeleton_output_dim,
                 sensor_in_channels, sensor_out_channels, sensor_length,
                 shared_out, backbone_dim, num_classes, synchronized_loading=False):
        super().__init__()
        self.sync = synchronized_loading
        self.skel_enc = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.sens_enc = SensorEncoder(sensor_in_channels, sensor_out_channels, sensor_length)
        self.backbone = SharedBackbone(skeleton_output_dim, shared_out, backbone_dim)
        branch_dim = shared_out*backbone_dim
        fused_dim = 2*branch_dim
        self.head = nn.Linear(fused_dim, num_classes)
        self.use_skel_only = False
        self.use_sens_only = False
        
    def forward(self, x_skel, x_sens):
        a = self.backbone(self.skel_enc(x_skel)).flatten(1)
        b = self.backbone(self.sens_enc(x_sens)).flatten(1)
        
        # mask before concatenation
        if self.use_skel_only:
            se = torch.zeros_like(se)
        elif self.use_sens_only:
            sk = torch.zeros_like(sk)
        
        f = torch.cat([a,b], dim=1)
        return self.head(f)

class ShareLatentModel(nn.Module):
    def __init__(self, skeleton_input_dim, skeleton_output_dim,
                 sensor_in_channels, sensor_out_channels, sensor_length,
                 shared_out, backbone_dim, num_classes, synchronized_loading=False):
        super().__init__()
        self.sync = synchronized_loading
        self.skel_enc = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.sens_enc = SensorEncoder(sensor_in_channels, sensor_out_channels, sensor_length)
        self.proj_s = nn.Linear(skeleton_output_dim, shared_out)
        self.proj_g = nn.Linear(sensor_out_channels, shared_out)
        self.backbone = SharedBackbone(shared_out, shared_out, backbone_dim)
        feat_dim = shared_out*backbone_dim
        self.head = nn.Linear(feat_dim, num_classes)
        self.use_skel_only = False
        self.use_sens_only = False
            
    def forward(self, x_skel, x_sens):
        sk_feat = self.skel_enc(x_skel)
        se_feat = self.sens_enc(x_sens)
        sk_lat  = self.proj_s(sk_feat)
        se_lat  = self.proj_g(se_feat)

        if self.use_skel_only:
            # only skeleton branch
            rep = self.backbone(sk_lat).flatten(1)
            p_skel = self.head(rep)
            p_sens = None

        elif self.use_sens_only:
            # only sensor branch
            rep = self.backbone(se_lat).flatten(1)
            p_skel = None
            p_sens = self.head(rep)

        else:
            # full fusion
            fused = sk_lat + se_lat
            rep   = self.backbone(fused).flatten(1)
            p     = self.head(rep)
            # under sync mode these are the same head, but we return both
            p_skel, p_sens = p, p

        return p_skel, p_sens

class CheapCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5
    def forward(self, S, G):
        sim = (S @ G.transpose(1,2))*self.scale
        attn_sg = F.softmax(sim, dim=-1)
        attn_gs = F.softmax(sim.transpose(1,2), dim=-1)
        S_star = attn_sg @ G; G_star = attn_gs @ S
        return 0.5*(S_star+G_star)

class CheapXAttnModel(nn.Module):
    def __init__(self, skeleton_input_dim, skeleton_output_dim,
                 sensor_in_channels, sensor_out_channels, sensor_length,
                 shared_out, backbone_dim, num_classes, synchronized_loading=False):
        super().__init__()
        assert skeleton_output_dim==sensor_out_channels
        self.sync = synchronized_loading
        self.skel_enc = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.sens_enc = SensorEncoder(sensor_in_channels, sensor_out_channels, sensor_length)
        self.cross = CheapCrossAttention(skeleton_output_dim)
        self.backbone = SharedBackbone(skeleton_output_dim, shared_out, backbone_dim)
        feat_dim = shared_out*backbone_dim
        self.head = nn.Linear(feat_dim, num_classes)
        self.use_skel_only = False
        self.use_sens_only = False
        
    def forward(self, x_skel, x_sens):
        sk_feat = self.skel_enc(x_skel)   # (B,T,d)
        se_feat = self.sens_enc(x_sens)   # (B,T,d)

        # ---- masking ablation: skip the cross-attn if one branch is off ----
        if self.use_skel_only:
            # just run skeleton branch through backbone
            rep    = self.backbone(sk_feat).flatten(1)
            p_skel = self.head(rep)
            p_sens = None

        elif self.use_sens_only:
            rep    = self.backbone(se_feat).flatten(1)
            p_skel = None
            p_sens = self.head(rep)

        else:
            # full cross-attention fusion
            fus    = self.cross(sk_feat, se_feat)
            rep    = self.backbone(fus).flatten(1)
            p      = self.head(rep)
            p_skel, p_sens = p, p

        return p_skel, p_sens

# at bottom of feature_encoder.py (or in its own baselines.py)

class SkelOnlyModel(nn.Module):
    def __init__(self,
                 skeleton_input_dim:int,
                 skeleton_output_dim:int,
                 shared_out:int, backbone_dim:int,
                 num_classes:int):
        super().__init__()
        self.encoder  = SkeletonMLP(skeleton_input_dim, skeleton_output_dim)
        self.backbone = SharedBackbone(skeleton_output_dim, shared_out, backbone_dim)
        self.head     = nn.Linear(shared_out*backbone_dim, num_classes)

    def forward(self, x_skel, _=None):
        feat = self.encoder(x_skel)            # (B,T, C)
        rep  = self.backbone(feat).flatten(1)  # (B, D)
        return self.head(rep)                  # (B, num_classes)


class SensorOnlyModel(nn.Module):
    def __init__(self,
                 sensor_in_channels:int,
                 sensor_out_channels:int,
                 sensor_length:int,
                 shared_out:int, backbone_dim:int,
                 num_classes:int):
        super().__init__()
        self.encoder  = SensorEncoder(sensor_in_channels, sensor_out_channels, sensor_length)
        self.backbone = SharedBackbone(sensor_out_channels, shared_out, backbone_dim)
        self.head     = nn.Linear(shared_out*backbone_dim, num_classes)

    def forward(self, _, x_sens):
        feat = self.encoder(x_sens)            # (B,T, C)
        rep  = self.backbone(feat).flatten(1)  # (B, D)
        return self.head(rep)                  # (B, num_classes)
