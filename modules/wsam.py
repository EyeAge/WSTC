import torch
import torch.nn as nn

TASK_DIM = 256      # 你想统一的 task 维度

class WSAM(nn.Module):
    def __init__(self, image_feature_dim=512, task_embedding_dim=TASK_DIM, num_heads=4, num_layers=2, ff_dim=1024, new_interface=False):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, image_feature_dim))  # [1, 1, D]
        self.new_interface = new_interface

        encoder_layer = nn.TransformerEncoderLayer(d_model=image_feature_dim, nhead=num_heads, batch_first=True, dim_feedforward=ff_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(nn.LayerNorm(image_feature_dim), nn.Linear(image_feature_dim, 512), nn.ReLU(),
                                 nn.Linear(512, task_embedding_dim))

    def forward(self, patch_feats):
        """
        patch_feats: [B, N, D] or [B, 1, D] ← 输入为 CLIP 的图像特征，unsqueeze 处理过
        Returns:
            z_t: [B, task_embedding_dim]
        """
        patch_feats = patch_feats.to(self.cls_token.dtype)  # 保证同 dtype
        B = patch_feats.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_token, patch_feats], dim=1)  # [B, 1+N, D]
        x = self.transformer(x)  # [B, 1+N, D]
        # 直接用第 0 个位置的 CLS token 作为输出
        cls_out = x[:, 0, :]  # [B, D]
        z_t = self.mlp(cls_out)  # [B, task_embedding_dim]

        assert z_t.shape[-1] == 256, 'WSAM 输出维度应当是 256'
        return z_t
