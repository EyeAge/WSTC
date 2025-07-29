# meta_conditioning.py
import torch
import torch.nn as nn

class MetaConditioning(nn.Module):
    def __init__(self, task_dim=256, cond_dim=64, num_tasks=5):
        super().__init__()
        self.embed = nn.Embedding(num_tasks, cond_dim)
        self.fuse = nn.Sequential(
            nn.Linear(task_dim + cond_dim, task_dim),
            nn.ReLU(inplace=True)
        )
        # self.task_id = task_id
        # self.task_embed = nn.Embedding(num_tasks, task_dim)

    def forward(self, task_feat, task_id=0):
        """
        task_feat : [B, task_dim]
        task_id   : int or [B] – 当前任务 ID
        """
        if not torch.is_tensor(task_id):
            task_id = torch.tensor(task_id).to(task_feat.device)
        if task_id.ndim == 0:
            task_id = task_id.expand(task_feat.shape[0])
        cond_vec = self.embed(task_id)  # [B, cond_dim]
        x = torch.cat([task_feat, cond_vec], dim=-1)  # [B, task_dim + cond_dim]
        return self.fuse(x)  # [B, task_dim]
