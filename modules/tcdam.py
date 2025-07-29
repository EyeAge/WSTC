import torch
import torch.nn as nn

TASK_DIM = 256      # 你想统一的 task 维度
PROJ_DIM = 256

class TCDAM(nn.Module):
    def __init__(self,
                 image_feature_dim=512,
                 task_embedding_dim=TASK_DIM,       # ← ① 改成 512
                 projection_dim=PROJ_DIM,
                 proj_token_dim=None):         # ← ② 允许自定义 ProjToken 长度
        super().__init__()

        if proj_token_dim is None:             # ③ 默认跟 task_dim 一样
            proj_token_dim = task_embedding_dim

        # --------- 参数 ----------
        self.image_feature_dim = image_feature_dim
        self.projection_dim     = projection_dim
        self.task_dim           = task_embedding_dim
        self.proj_dim           = proj_token_dim

        # ---------- ProjToken ----------
        self.proj_token = nn.Parameter(
            torch.randn(1, proj_token_dim)
        )                                       # [1 , P_dim]

        # ---------- MLP ----------
        cond_dim   = self.task_dim + self.proj_dim           # 512+512(或256)=?
        hidden_dim = cond_dim                                # 用同宽隐藏层最省事

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, image_feature_dim * projection_dim)
        )

    # ----------------------------
    def forward(self, z_t):                     # z_t: [B , task_dim]
        B = z_t.size(0)

        # ----- 拼接 -----
        cond = torch.cat(                       # [B , task_dim + P_dim]
            [z_t, self.proj_token.expand(B, -1)], dim=1
        )

        # ==========================debugger==========================
        # print("[DBG] input to self.mlp (cond):", cond.shape)
        # =========================================================

        # ----- 预测投影矩阵 -----
        W_flat = self.mlp(cond)                 # [B , D*P]
        D, P   = self.image_feature_dim, self.projection_dim
        W      = W_flat.view(B, D, P)           # [B , D , P]
        W      = W / (W.norm(dim=1, keepdim=True).clamp_min(1e-6))
        return W
