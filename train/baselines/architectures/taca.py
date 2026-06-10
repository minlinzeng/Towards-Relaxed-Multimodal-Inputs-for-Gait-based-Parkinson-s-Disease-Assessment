# file: taca.py
import math, torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Tokenizer", "TACA", "TACAWrapper"]

# ---------------- Tokenizer ----------------

class Tokenizer(nn.Module):
    def __init__(self, D_frame: int, d_model: int, n_tokens: int, use_time_shared: bool = True):
        super().__init__()
        self.use_time_shared = use_time_shared
        self.n_tokens, self.d = n_tokens, d_model
        if use_time_shared:
            self.frame_proj = nn.Linear(D_frame, d_model, bias=False)  # shared across time
        else:
            self.proj = nn.Sequential(
                nn.Linear(D_frame, d_model * n_tokens, bias=False),
                nn.LayerNorm(d_model * n_tokens),
                nn.GELU()
            )

    def forward(self, x_flat: torch.Tensor, T_frames: int, D_frame: int) -> torch.Tensor:
        # x_flat: [B, T_frames * D_frame]
        B = x_flat.size(0)
        x = x_flat.view(B, T_frames, D_frame)
        if self.use_time_shared:
            z = self.frame_proj(x)  # [B, T_frames, d]
            stride = max(1, T_frames // self.n_tokens)
            z = z[:, ::stride, :][:, :self.n_tokens, :]  # [B, T, d]
            return z
        y = self.proj(x_flat.view(B, -1)).view(B, self.n_tokens, self.d)
        return y


# ---------------- TACA core ----------------

class TACA(nn.Module):
    def __init__(self, d, n_heads=4, tau=1.0, gamma=1.5, schedule='const', depth_id=0, num_depths=1, dropout=0.0):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.h, self.dk = d, n_heads, d // n_heads
        self.tau, self.gamma0 = tau, gamma
        self.schedule, self.depth_id = schedule, depth_id
        self.num_depths = max(1, num_depths)

        # skel->sens
        self.q_s2e = nn.Linear(d, d, bias=False)
        self.k_e   = nn.Linear(d, d, bias=False)
        self.v_e   = nn.Linear(d, d, bias=False)
        # sens->skel
        self.q_e2s = nn.Linear(d, d, bias=False)
        self.k_s   = nn.Linear(d, d, bias=False)
        self.v_s   = nn.Linear(d, d, bias=False)

        self.o_s, self.o_e = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('epoch_frac', torch.tensor(0.0))

    @torch.no_grad()
    def set_epoch_frac(self, frac: float):
        self.epoch_frac.fill_(float(max(0.0, min(1.0, frac))))

    def _gamma(self):
        if self.schedule == 'const': return self.gamma0
        if self.schedule == 'depth':
            alpha = 1.0 - (self.depth_id / max(1, self.num_depths - 1))
            return 1.0 + alpha * (self.gamma0 - 1.0)
        if self.schedule == 'epoch':
            alpha = 1.0 - float(self.epoch_frac)
            return 1.0 + alpha * (self.gamma0 - 1.0)
        return self.gamma0

    def _proj(self, x, lin, B, T):
        return lin(x).view(B, T, self.h, self.dk).transpose(1, 2)  # [B,h,T,dk]

    def _cross(self, q_lin, k_lin, v_lin, x_q, x_kv, mask_kv, scale):
        B, Tq, _ = x_q.shape
        Tk = x_kv.shape[1]
        q = self._proj(x_q, q_lin, B, Tq)
        k = self._proj(x_kv, k_lin, B, Tk)
        v = self._proj(x_kv, v_lin, B, Tk)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask_kv is not None:
            logits = logits.masked_fill(mask_kv[:, None, None, :] == 0, float('-inf'))
        attn = F.softmax(scale * logits, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Tq, self.d)
        return out

    def forward(self, skel_seq, sens_seq, mask_skel=None, mask_sens=None):
        scale = self._gamma() / self.tau
        s2e = self._cross(self.q_s2e, self.k_e, self.v_e, skel_seq, sens_seq, mask_sens, scale)
        s2e = self.drop(self.o_e(s2e))
        e2s = self._cross(self.q_e2s, self.k_s, self.v_s, sens_seq, skel_seq, mask_skel, scale)
        e2s = self.drop(self.o_s(e2s))
        return e2s, s2e  # (sens→skel enhanced, skel→sens enhanced)


# ---------------- Wrapper with tokenizers + heads ----------------

class TACAWrapper(nn.Module):
    """
    Call:
      logits_skel, logits_sens = model(x_skel_flat, x_sens_flat, synced: bool)

    You MUST provide per-modality frame shapes to build the time-shared tokenizers.
    """
    def __init__(self, *,
                 skel_T_frames: int, skel_D_frame: int,
                 sens_T_frames: int, sens_D_frame: int,
                 num_classes: int,
                 d_model=128, n_heads=4, n_tok_s=8, n_tok_e=8,
                 tau=1.0, gamma=1.5, schedule='const', depth_id=0, num_depths=1, dropout=0.1,
                 use_time_shared=True):
        super().__init__()
        # Store shapes
        self.skel_T, self.skel_D = int(skel_T_frames), int(skel_D_frame)
        self.sens_T, self.sens_D = int(sens_T_frames), int(sens_D_frame)

        # Tokenizers (time-shared by default)
        self.tk_s = Tokenizer(D_frame=self.skel_D, d_model=d_model, n_tokens=n_tok_s, use_time_shared=use_time_shared)
        self.tk_e = Tokenizer(D_frame=self.sens_D, d_model=d_model, n_tokens=n_tok_e, use_time_shared=use_time_shared)

        # Fuser + heads
        self.fuser = TACA(d=d_model, n_heads=n_heads, tau=tau, gamma=gamma,
                          schedule=schedule, depth_id=depth_id, num_depths=num_depths, dropout=dropout)
        self.head_joint = nn.Linear(d_model, num_classes)
        self.head_skel  = nn.Linear(d_model, num_classes)
        self.head_sens  = nn.Linear(d_model, num_classes)

    @torch.no_grad()
    def set_epoch_frac(self, frac: float):
        self.fuser.set_epoch_frac(frac)

    @staticmethod
    def _pool(seq):  # [B,T,D] -> [B,D]
        return seq.mean(dim=1)

    def _check_shape(self, x_flat: torch.Tensor, T: int, D: int, name: str):
        if x_flat is None: return
        exp = T * D
        got = int(x_flat.size(-1))
        if got != exp:
            raise ValueError(f"{name}: expected last dim {exp} = T({T})*D({D}), got {got}")

    def forward(self, x_skel: torch.Tensor, x_sens: torch.Tensor, synced: bool):
        has_s = x_skel is not None
        has_e = x_sens is not None

        if has_s: self._check_shape(x_skel, self.skel_T, self.skel_D, "skeleton")
        if has_e: self._check_shape(x_sens, self.sens_T, self.sens_D, "sensor")

        z_s = self.tk_s(x_skel, self.skel_T, self.skel_D) if has_s else None   # [B,Ts,Dm]
        z_e = self.tk_e(x_sens, self.sens_T, self.sens_D) if has_e else None   # [B,Te,Dm]

        if synced and has_s and has_e:
            y_sens, y_skel = self.fuser(z_s, z_e, None, None)
            z = 0.5 * (self._pool(y_skel) + self._pool(y_sens))
            return self.head_joint(z), None

        if has_s and has_e:
            y_sens, y_skel = self.fuser(z_s, z_e, None, None)
            ps = self.head_skel(self._pool(y_skel))
            pe = self.head_sens(self._pool(y_sens))
            return ps, pe
        if has_s:
            return self.head_skel(self._pool(z_s)), None
        if has_e:
            return None, self.head_sens(self._pool(z_e))
        return None, None



class TACA3TriWrapper(nn.Module):
    def __init__(self, *,
                 walk_T, walk_D, insole_T, insole_D, imu_T, imu_D,
                 num_classes, d_model=128, n_heads=4,
                 n_tok_w=8, n_tok_i=8, n_tok_m=8, tau=1.0, gamma=1.5,
                 schedule='const', dropout=0.1, use_time_shared=True,
                 allow_async_cross: bool = False):   # NEW: default off
        super().__init__()
        self.allow_async_cross = allow_async_cross
        
        self.wT,self.wD = walk_T, walk_D
        self.iT,self.iD = insole_T, insole_D
        self.mT,self.mD = imu_T,    imu_D
        
        # tokenizers
        self.tk_w = Tokenizer(self.wD, d_model, n_tok_w, use_time_shared)
        self.tk_i = Tokenizer(self.iD, d_model, n_tok_i, use_time_shared)
        self.tk_m = Tokenizer(self.mD, d_model, n_tok_m, use_time_shared)
        # pairwise TACAs
        self.wi = TACA(d_model, n_heads, tau, gamma, schedule, dropout=dropout)  # W↔I
        self.wm = TACA(d_model, n_heads, tau, gamma, schedule, dropout=dropout)  # W↔M
        self.im = TACA(d_model, n_heads, tau, gamma, schedule, dropout=dropout)  # I↔M
        # heads
        self.head_joint = nn.Linear(d_model, num_classes)
        self.head_w = nn.Linear(d_model, num_classes)
        self.head_i = nn.Linear(d_model, num_classes)
        self.head_m = nn.Linear(d_model, num_classes)
        
    @staticmethod
    def _pool(z): return z.mean(1)
    @staticmethod
    def _unflat(x, T, D): return x.view(x.size(0), T, D)

    def forward(self, xw_flat, xi_flat, xm_flat, *, synced: bool):
        has_w, has_i, has_m = xw_flat is not None, xi_flat is not None, xm_flat is not None
        ref = xw_flat if has_w else (xi_flat if has_i else xm_flat)
        B = ref.size(0); C = self.head_w.out_features
        device = ref.device
        def zlogits(): return torch.zeros(B, C, device=device)
        
        Zw = self.tk_w(xw_flat, self.wT, self.wD) if has_w else None
        Zi = self.tk_i(xi_flat, self.iT, self.iD) if has_i else None
        Zm = self.tk_m(xm_flat, self.mT, self.mD) if has_m else None

        if synced and (Zw is not None) and (Zi is not None) and (Zm is not None):
            wi_e2w, wi_w2i = self.wi(Zw, Zi, None, None)
            wm_e2w, wm_w2m = self.wm(Zw, Zm, None, None)
            im_e2i, im_i2m = self.im(Zi, Zm, None, None)

            W_enh = (wi_e2w + wm_e2w) * 0.5
            I_enh = (wi_w2i + im_e2i) * 0.5
            M_enh = (wm_w2m + im_i2m) * 0.5

            z = (W_enh.mean(1) + I_enh.mean(1) + M_enh.mean(1)) / 3.0
            y = self.head_joint(z)
            return y, y, y

        # ---- ASYNC ----
        if self.allow_async_cross and (Zw is not None) and (Zi is not None):
            _, wi_w2i = self.wi(Zw, Zi, None, None)
        else:
            wi_w2i = Zi
        if self.allow_async_cross and (Zw is not None) and (Zm is not None):
            _, wm_w2m = self.wm(Zw, Zm, None, None)
        else:
            wm_w2m = Zm
        if self.allow_async_cross and (Zi is not None) and (Zm is not None):
            im_e2i, im_i2m = self.im(Zi, Zm, None, None)
        else:
            im_e2i, im_i2m = Zi, Zm

        yw = self.head_w(Zw.mean(1)) if Zw is not None else torch.zeros((xi_flat or xm_flat).size(0), self.head_w.out_features, device=(xi_flat or xm_flat).device)
        yi = self.head_i((wi_w2i if self.allow_async_cross and Zi is not None else Zi).mean(1)) if Zi is not None else torch.zeros_like(yw)
        ym = self.head_m((wm_w2m if self.allow_async_cross and Zm is not None else Zm).mean(1)) if Zm is not None else torch.zeros_like(yw)
        return yw, yi, ym
