# file: models/deepav.py
# DeepAV-Lite: minimal early-fusion with factorized interactions, with
# optional bottlenecked attention and weight sharing to shrink params.

import math, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- tiny building blocks ----------------
class PatchEmbed1D(nn.Module):
    def __init__(self, in_dim, embed_dim, patch=16, stride=16):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, embed_dim, kernel_size=patch, stride=stride, padding=0)
        self.ln   = nn.LayerNorm(embed_dim)
    def forward(self, x):                  # x: [B, T, Din]
        x = x.transpose(1, 2)              # [B, Din, T]
        z = self.proj(x).transpose(1, 2)   # [B, L, E]
        return self.ln(z)

class MLP(nn.Module):
    def __init__(self, d, r=4.0, p=0.0):
        super().__init__()
        h = int(d * r)
        self.fc1, self.fc2 = nn.Linear(d,h), nn.Linear(h,d)
        self.act, self.drop = nn.GELU(), nn.Dropout(p)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class MHSA(nn.Module):
    def __init__(self, d, h=4, p=0.0, d_att=None):
        super().__init__()
        self.d, self.h = d, h
        self.da = d_att or d
        assert self.da % h == 0
        self.dk = self.da // h
        self.q = nn.Linear(d, self.da, bias=False)
        self.k = nn.Linear(d, self.da, bias=False)
        self.v = nn.Linear(d, self.da, bias=False)
        self.o = nn.Linear(self.da, d, bias=False)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        B,T,_ = x.shape
        q = self.q(x).view(B,T,self.h,self.dk).transpose(1,2)   # [B,h,T,dk]
        k = self.k(x).view(B,T,self.h,self.dk).transpose(1,2)
        v = self.v(x).view(B,T,self.h,self.dk).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / (self.dk ** 0.5)
        z = (att.softmax(-1) @ v).transpose(1,2).contiguous().view(B,T,self.da)
        return self.drop(self.o(z))

class XAttn(nn.Module):
    """Cross-attention: queries attend to kv."""
    def __init__(self, d, h=4, p=0.0, d_att=None):
        super().__init__()
        self.d, self.h = d, h
        self.da = d_att or d
        assert self.da % h == 0
        self.dk = self.da // h
        self.q = nn.Linear(d, self.da, bias=False)
        self.k = nn.Linear(d, self.da, bias=False)
        self.v = nn.Linear(d, self.da, bias=False)
        self.o = nn.Linear(self.da, d, bias=False)
        self.drop = nn.Dropout(p)
    def forward(self, q_in, kv_in):        # q_in: [B,Nq,E], kv_in: [B,Nk,E]
        B,Nq,_ = q_in.shape; Nk = kv_in.size(1)
        q = self.q(q_in).view(B,Nq,self.h,self.dk).transpose(1,2)  # [B,h,Nq,dk]
        k = self.k(kv_in).view(B,Nk,self.h,self.dk).transpose(1,2) # [B,h,Nk,dk]
        v = self.v(kv_in).view(B,Nk,self.h,self.dk).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / (self.dk ** 0.5)
        z = (att.softmax(-1) @ v).transpose(1,2).contiguous().view(B,Nq,self.da)
        return self.drop(self.o(z))

class Block(nn.Module):
    """Standard Transformer block with optional attention bottleneck."""
    def __init__(self, d, h=4, r=4.0, p=0.0, d_att=None):
        super().__init__()
        self.ln1, self.sa = nn.LayerNorm(d), MHSA(d,h,p,d_att)
        self.ln2, self.ff = nn.LayerNorm(d), MLP(d,r,p)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ---------------- DeepAV-Lite ----------------

class DeepAVLite(nn.Module):
    """
    Early fusion with factorized interactions:
      - Per-modality self-attn stacks (Step A)
      - Per-modality aggregation cross-attn (Step B)
      - Fusion tokens attending to concatenated aggregates (Step C)
      - Synced: joint head on fusion CLS; Async: per-branch heads
    """
    def __init__(self,
                 skel_in_dim, sens_in_dim, num_classes,
                 embed_dim=None, depth=3, heads=3, mlp_ratio=2.0,
                 skel_patch=16, sens_patch=16, stride=16, drop=0.1,
                 n_agg=4, n_fusion=4,
                 use_cls=True, pool="cls",
                 share_blocks=False, share_unimodal=False, attn_bottleneck=None):
        super().__init__()
        E = embed_dim
        self.use_cls, self.pool = use_cls, pool
        self.E, self.n_agg, self.n_fusion = E, n_agg, n_fusion
        self.depth = depth
        self.share_blocks = share_blocks
        self.share_unimodal = share_unimodal
        self.d_att = attn_bottleneck or embed_dim

        # tokenizers
        self.tk_s = PatchEmbed1D(skel_in_dim, E, skel_patch, stride)
        self.tk_e = PatchEmbed1D(sens_in_dim, E, sens_patch, stride)

        # unimodal stacks (with optional sharing)
        if self.share_unimodal:
            base = Block(E, heads, mlp_ratio, drop, d_att=self.d_att)
            if self.share_blocks:
                # one block shared across both modalities and all layers
                self.skel_blocks = self.sens_blocks = nn.ModuleList([base])
            else:
                # same weights for both modalities, repeated L times
                self.skel_blocks = self.sens_blocks = nn.ModuleList([base for _ in range(depth)])
        else:
            if self.share_blocks:
                # one block per modality, shared across its layers
                s_block = Block(E, heads, mlp_ratio, drop, d_att=self.d_att)
                e_block = Block(E, heads, mlp_ratio, drop, d_att=self.d_att)
                self.skel_blocks = nn.ModuleList([s_block])
                self.sens_blocks = nn.ModuleList([e_block])
            else:
                # independent blocks per modality and layer
                self.skel_blocks = nn.ModuleList([Block(E, heads, mlp_ratio, drop, d_att=self.d_att) for _ in range(depth)])
                self.sens_blocks = nn.ModuleList([Block(E, heads, mlp_ratio, drop, d_att=self.d_att) for _ in range(depth)])

        # aggregation queries (learnable)
        self.agg_s_q = nn.Parameter(torch.randn(n_agg, E) * 0.02)
        self.agg_e_q = nn.Parameter(torch.randn(n_agg, E) * 0.02)
        self.xattn_s = XAttn(E, heads, drop, d_att=self.d_att)  # agg_s <- skel_tokens
        self.xattn_e = XAttn(E, heads, drop, d_att=self.d_att)  # agg_e <- sens_tokens

        # fusion tokens and fusion block
        self.fus_tok   = nn.Parameter(torch.randn(n_fusion + (1 if use_cls else 0), E) * 0.02)
        self.fuse_xattn= XAttn(E, heads, drop, d_att=self.d_att)  # fusion <- [agg_s, agg_e]
        self.fuse_ff   = MLP(E, mlp_ratio, drop)
        self.ln_fuse   = nn.LayerNorm(E)

        # heads
        self.head_joint = nn.Linear(E, num_classes)
        self.head_skel  = nn.Linear(E, num_classes)
        self.head_sens  = nn.Linear(E, num_classes)

        # modality/type embeddings
        self.type_s = nn.Parameter(torch.randn(1,1,E)*0.02)
        self.type_e = nn.Parameter(torch.randn(1,1,E)*0.02)
        self.type_c = nn.Parameter(torch.randn(1,1,E)*0.02) if use_cls else None

    def _pos_enc(self, B, L, device):
        # fixed sinusoid
        pos = torch.arange(L, device=device).float()
        dim = torch.arange(self.E, device=device).float()
        div = torch.exp(dim//2 * (-math.log(10000.0)/max(1,(self.E//2))))
        pe  = torch.zeros(L, self.E, device=device)
        pe[:, 0::2] = torch.sin(pos[:,None]*div[0::2])
        pe[:, 1::2] = torch.cos(pos[:,None]*div[0::2])
        return pe.unsqueeze(0).expand(B, -1, -1)

    def forward_feats(self, skel, sens):
        B = skel.size(0)
        z_s = self.tk_s(skel) + self.type_s                  # [B,Ls,E]
        z_e = self.tk_e(sens) + self.type_e                  # [B,Lg,E]
        z_s = z_s + self._pos_enc(B, z_s.size(1), z_s.device)
        z_e = z_e + self._pos_enc(B, z_e.size(1), z_e.device)

        # fusion tokens per batch
        F = self.fus_tok.unsqueeze(0).repeat(B, 1, 1).contiguous()  # [B,Nf(+CLS),E]
        if self.use_cls:
            cls_bias = torch.zeros_like(F)
            cls_bias[:, 0, :] = self.type_c.expand(B, 1, self.E).squeeze(1)
            F = F + cls_bias

        for li in range(self.depth):
            blk_s = self.skel_blocks[0] if self.share_blocks else self.skel_blocks[li]
            blk_e = self.sens_blocks[0] if self.share_blocks else self.sens_blocks[li]

            # Step A: unimodal temporal mixing
            z_s = blk_s(z_s)
            z_e = blk_e(z_e)

            # Step B: per-modality aggregation via cross-attn
            qs = self.agg_s_q.unsqueeze(0).expand(B, -1, -1)
            qe = self.agg_e_q.unsqueeze(0).expand(B, -1, -1)
            agg_s = self.xattn_s(qs, z_s)   # [B,n_agg,E]
            agg_e = self.xattn_e(qe, z_e)   # [B,n_agg,E]

            # Step C: factorized cross-modal fusion via fusion tokens
            av = torch.cat([agg_s, agg_e], dim=1)            # [B,2*n_agg,E]
            F = F + self.fuse_xattn(F, av)                   # fusion <- aggregated AV
            F = F + self.fuse_ff(self.ln_fuse(F))            # refine fusion tokens

        # readouts
        joint  = F[:,0,:] if (self.use_cls and self.pool == "cls") else F.mean(1)
        sk_pool= z_s.mean(1)
        se_pool= z_e.mean(1)
        return joint, sk_pool, se_pool

    def forward(self, skel, sens, synced: bool = True):
        joint, sk_pool, se_pool = self.forward_feats(skel, sens)
        if synced:
            return self.head_joint(joint), None
        else:
            return self.head_skel(sk_pool), self.head_sens(se_pool)
              
# ---- N-modality thin extension (keeps original DeepAVLite intact) ----
class DeepAVLiteN(nn.Module):
    """
    Generic N-modality DeepAV-Lite.
    - modal_dims: Ordered dict {name: input_dim}, e.g. {"walkway":2,"insole":13,"imu":24}
    - synchronized=True: one joint head (replicated N times to fit your trainers)
      synchronized=False: per-modality heads
    """
    def __init__(self, modal_dims: dict, num_classes: int,
                 *, embed_dim=96, depth=3, heads=3, mlp_ratio=2.0,
                 patch=8, stride=8, drop=0.1, n_agg=4, n_fusion=4,
                 use_cls=True, pool="cls",
                 share_blocks=False, share_unimodal=False, attn_bottleneck=None,
                 synchronized=True):
        super().__init__()
        from collections import OrderedDict
        self.modal_names = list(OrderedDict(modal_dims).keys())
        self.synchronized = synchronized
        self.E, self.depth, self.n_agg, self.n_fusion = embed_dim, depth, n_agg, n_fusion
        self.use_cls, self.pool = use_cls, pool
        self.share_blocks, self.share_unimodal = share_blocks, share_unimodal
        self.d_att = attn_bottleneck or embed_dim

        # tokenizers & type embeddings
        self.tokenizers = nn.ModuleDict({
            m: PatchEmbed1D(in_dim, embed_dim, patch, stride) for m, in_dim in modal_dims.items()
        })
        self.type_embed = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1,1,embed_dim)*0.02) for m in modal_dims
        })

        # unimodal stacks
        def make_stack():
            if self.share_blocks:
                blk = Block(embed_dim, heads, mlp_ratio, drop, d_att=self.d_att)
                return nn.ModuleList([blk])
            else:
                return nn.ModuleList([Block(embed_dim, heads, mlp_ratio, drop, d_att=self.d_att) for _ in range(depth)])
        if self.share_unimodal:
            base = Block(embed_dim, heads, mlp_ratio, drop, d_att=self.d_att)
            if self.share_blocks:
                stacks = nn.ModuleList([base])
            else:
                stacks = nn.ModuleList([base for _ in range(depth)])
            self.blocks = nn.ModuleDict({m: stacks for m in modal_dims})
        else:
            self.blocks = nn.ModuleDict({m: make_stack() for m in modal_dims})

        # per-mod aggregation
        self.agg_q   = nn.ParameterDict({m: nn.Parameter(torch.randn(n_agg, embed_dim)*0.02) for m in modal_dims})
        self.xattn_a = nn.ModuleDict({m: XAttn(embed_dim, heads, drop, d_att=self.d_att) for m in modal_dims})

        # fusion
        self.fus_tok    = nn.Parameter(torch.randn(n_fusion + (1 if use_cls else 0), embed_dim)*0.02)
        self.type_cls   = nn.Parameter(torch.randn(1,1,embed_dim)*0.02) if use_cls else None
        self.fuse_xattn = XAttn(embed_dim, heads, drop, d_att=self.d_att)
        self.fuse_ff    = MLP(embed_dim, mlp_ratio, drop)
        self.ln_fuse    = nn.LayerNorm(embed_dim)

        # heads
        self.head_joint = nn.Linear(embed_dim, num_classes)
        if not synchronized:
            self.heads = nn.ModuleDict({m: nn.Linear(embed_dim, num_classes) for m in modal_dims})

    def _pos_enc(self, B, L, device):
        pos = torch.arange(L, device=device).float()
        dim = torch.arange(self.E, device=device).float()
        div = torch.exp((dim//2) * (-math.log(10000.0)/max(1,(self.E//2))))
        pe  = torch.zeros(L, self.E, device=device)
        pe[:,0::2] = torch.sin(pos[:,None]*div[0::2]); pe[:,1::2] = torch.cos(pos[:,None]*div[0::2])
        return pe.unsqueeze(0).expand(B, -1, -1)

    def forward(self, inputs: dict):
        """
        inputs: {name: tensor(B,T,D)} in the same keys/order as modal_dims
        returns (lw, li, lm, ...) in the declared modal order.
        """
        B = next(iter(inputs.values())).size(0)

        # tokenize + add types + pos
        Z = {}
        for m in self.modal_names:
            z = self.tokenizers[m](inputs[m]) + self.type_embed[m]
            Z[m] = z + self._pos_enc(B, z.size(1), z.device)

        # fusion tokens
        F = self.fus_tok.unsqueeze(0).repeat(B, 1, 1).contiguous()
        if self.use_cls:
            F[:,0,:] = F[:,0,:] + self.type_cls.expand(B,1,self.E).squeeze(1)

        # depth loop
        for li in range(self.depth):
            # A) unimodal mixing
            for m in self.modal_names:
                blk = self.blocks[m][0] if self.share_blocks else self.blocks[m][li]
                Z[m] = blk(Z[m])

            # B) per-mod aggregation
            aggs = []
            for m in self.modal_names:
                q = self.agg_q[m].unsqueeze(0).expand(B, -1, -1)
                aggs.append(self.xattn_a[m](q, Z[m]))  # [B,n_agg,E]
            av = torch.cat(aggs, dim=1)               # [B, N_mod*n_agg, E]

            # C) fusion tokens attend aggregates
            F = F + self.fuse_xattn(F, av)
            F = F + self.fuse_ff(self.ln_fuse(F))

        # readouts
        joint = F[:,0,:] if (self.use_cls and self.pool=="cls") else F.mean(1)
        if self.synchronized:
            j = self.head_joint(joint)
            # replicate to match your (lw,li,lm) interface
            return tuple(j for _ in self.modal_names)
        else:
            outs = []
            for m in self.modal_names:
                pm = Z[m].mean(1)
                outs.append(self.heads[m](pm))
            return tuple(outs)

# ---- tiny 3-mod wrapper to match your trainer signature ----
class DeepAVLite3(nn.Module):
    """
    Wrapper: keeps (x_walk, x_insole, x_imu) → (lw, li, lm) signature.
    """
    def __init__(self, num_classes, *, embed_dim=96, depth=3, heads=3, mlp_ratio=2.0,
                 patch=8, stride=8, drop=0.1, n_agg=4, n_fusion=4, use_cls=True, pool="cls",
                 share_blocks=False, share_unimodal=False, attn_bottleneck=None,
                 synchronized=True):
        super().__init__()
        self.core = DeepAVLiteN(
            {"walkway":2, "insole":13, "imu":24}, num_classes,
            embed_dim=embed_dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio,
            patch=patch, stride=stride, drop=drop, n_agg=n_agg, n_fusion=n_fusion,
            use_cls=use_cls, pool=pool,
            share_blocks=share_blocks, share_unimodal=share_unimodal, attn_bottleneck=attn_bottleneck,
            synchronized=synchronized
        )

    def forward(self, x_walk, x_insole, x_imu):
        return self.core({"walkway": x_walk, "insole": x_insole, "imu": x_imu})
