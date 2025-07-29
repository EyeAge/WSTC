import sys
import os
import torch
import torch.nn as nn

from wsam import WSAM
from tcdam import TCDAM
from text_projector import SmallMLPProjector
from meta_conditioning import MetaConditioning
from cross_modal import CrossModalFusion

TASK_DIM = 256              # ➟ 想改 512 也行，但三处要一致
PROJ_DIM = 256

import torch.nn.functional as F

def temperature_scaling(logits, temperature=1.0):
    """应用温度缩放调整输出"""
    logits = logits / temperature
    return F.softmax(logits, dim=-1)


class UniMedCLIP_WSTC(nn.Module):
    def __init__(self,
                 base_model,
                 # clip_model,
                 num_classes: int = 15,
                 task_dim: int = TASK_DIM,
                 proj_dim: int = PROJ_DIM,
                 *,
                 use_wsam: bool = True,
                 use_tcdam: bool = True,
                 with_cls_head: bool = True,
                 task_id = None,
                 use_cross_modal: bool = False
                 ):
        super().__init__()
        self.base = base_model
        self.clip = base_model
        self.use_wsam = use_wsam
        self.use_tcdam = use_tcdam
        self.with_cls = with_cls_head
        self.task_id = task_id
        num_tasks = 10  # ✅ 你目前最多支持几个任务（可后续用 config）
        self.task_embed = nn.Embedding(num_tasks, task_dim)
        self._skip_proj = False  # True ⇒ 直接用 512 维 CLS / Text
        # self.use_cross_modal = use_cross_modal  # 新增一个 flag
        # self.cross_fuser = CrossModalFusion(dim=proj_dim)

        feat_dim = self.clip.visual.output_dim  # typically 512
        # ✅ 冻结原始 CLIP 模型
        for p in self.clip.parameters():
            p.requires_grad_(False)

        if hasattr(self.base, "text_encoder"):  # HuggingFace BERT
            text_emb_dim = self.base.text_encoder.config.hidden_size  # 通常 768
        else:  # open-clip CLIP
            text_emb_dim = (self.clip.text_projection.out_features
                            if hasattr(self.clip, "text_projection")
                            else feat_dim)  # 通常 512

        # ✅ 初始化 WSAM 模块
        if use_wsam:
            self.wsam = WSAM(image_feature_dim=feat_dim, task_embedding_dim=task_dim)

        # ✅ 初始化 TCDAM 模块
        if use_tcdam:
            self.tcdam = TCDAM(task_embedding_dim=task_dim, image_feature_dim=feat_dim, projection_dim=proj_dim)
            self.image_proj = nn.Identity()
            feat_dim = self.clip .visual.output_dim

            self.text_proj = SmallMLPProjector(  # 768→256 / 512→256
                input_dim=text_emb_dim,
                hidden_dim=text_emb_dim,
                output_dim=proj_dim
            )
        else:
            self.image_proj = nn.Identity()
            # self.text_proj = nn.Identity()
            # ---- WSAM-only 需要把 768 → 512（与视觉对齐） ----
            if text_emb_dim != feat_dim:  # 768 vs 512
                self.text_proj = nn.Linear(text_emb_dim, feat_dim, bias=False)
                torch.nn.init.normal_(self.text_proj.weight, std=0.02)
            else:
                self.text_proj = nn.Identity()

        # ✅ 分类头（可选）
        out_dim = proj_dim if use_tcdam else task_dim if use_wsam else feat_dim

        if with_cls_head:
            assert num_classes is not None, "with_cls_head=True 但未设置 num_classes"
            self.cls = nn.Linear(out_dim, num_classes)

        print('task_id', task_id)
        if task_id in [3, 8]:  # monuseg 分割任务
            self.seg_decoder = nn.Sequential(
                nn.Linear(out_dim, 256),
                nn.ReLU(),
                nn.Unflatten(1, (16, 4, 4)),  # 256 → [16, 4, 4]
                nn.Upsample(scale_factor=2),  # [16,8,8]
                nn.Conv2d(16, 8, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=4),  # [8,32,32]
                nn.Conv2d(8, 1, 1),
                nn.Sigmoid()
            )

        self.l2 = nn.functional.normalize
        self.fallback_proj = nn.Linear(feat_dim, task_dim, bias=False)

        # ✅ 所有模块移动到 CLIP 的设备上
        tgt_device = next(self.clip.parameters()).device
        self.to(tgt_device)
        if self.text_proj is not None:
            self.text_proj = self.text_proj.to(tgt_device)

        self.meta_condition = MetaConditioning(task_dim=task_dim, num_tasks=num_tasks)
        # print("[INFO] feat_dim:", feat_dim)
        print("CLIP visual backbone type:", type(self.clip.visual))

    @property
    def text_encoder(self):
        return self.base.text_encoder  # 让外部能直接访问

    @property
    def encode_text(self):  # 兼容 open_clip 接口
        return self.base.encode_text

    @property
    def tokenizer(self):
        return getattr(self.base, "tokenizer", None)

    def forward(self, pixel_values, question_ids: torch.LongTensor = None,
            question_mask: torch.LongTensor = None,
            answer_ids: torch.LongTensor   = None,
            answer_mask: torch.LongTensor  = None):
        """
        pixel_values : [B, 3, H, W]
        return       : logits  [B, num_classes]   (with_cls=True)
                      or       image_feats       (otherwise)
        """
        dbg = hasattr(self.clip.visual, "forward_features")
        # print(f"[DBG] new_interface={dbg}")

        with torch.no_grad():
            vis_dict = self.clip.encode_image(pixel_values)  # ← 不传 flag
            cls_token = vis_dict["pooled_output"]  # [B,512]
            patch_tokens = vis_dict["patch_tokens"]  # [B,P,512] or None

            # ② 若 use_wsam 但 patch_tokens 仍为 None，直接报错
            if self.use_wsam and patch_tokens is None:
                raise RuntimeError(
                    "patch_tokens is None — backbone 没返回 patch。"
                    "请确认 open_clip 版本 ≥ 2.23 或换用 hook 方案。")

            # ③ patch_tokens 可能是 [B,D] → 统一成 [B,1,D]
            if patch_tokens is not None and patch_tokens.ndim == 2:
                patch_tokens = patch_tokens.unsqueeze(1)
            # ---------- ① 取得视觉输出 ----------
            elif isinstance(vis_dict, dict):
                def pick(d, *keys):
                    for k in keys:
                        v = d.get(k, None)
                        if v is not None:  # 只看 None，不对张量做 bool 运算
                            return v
                    return None

                cls_token = pick(vis_dict,
                                 "image_features",
                                 "pooled_output",
                                 "pooled",
                                 "cls_embedding")
                patch_tokens = pick(vis_dict, "patch_tokens", "x")
            else:
                def pick_attr(obj, *names):
                    for n in names:
                        v = getattr(obj, n, None)
                        if v is not None:
                            return v
                    return None

                cls_token = pick_attr(vis_dict,
                                      "pooled_output",
                                      "pooler_output",
                                      "last_hidden_state")
                # last_hidden_state → [B,N,D]，取 CLS
                if cls_token is not None and cls_token.ndim == 3:
                    cls_token = cls_token[:, 0, :]

                patch_tokens = pick_attr(vis_dict, "patch_tokens", "x")

            # 若仍拿不到 CLS，则用 patch[0] 退路
            if cls_token is None and patch_tokens is not None and patch_tokens.ndim == 3:
                cls_token, patch_tokens = patch_tokens[:, 0, :], patch_tokens[:, 1:, :]
            # 若还是拿不到 CLS，就直接把 patch_tokens 的第 1 个 token 当 CLS
            if cls_token is None and patch_tokens is not None and patch_tokens.ndim == 3:
                cls_token, patch_tokens = patch_tokens[:, 0, :], patch_tokens[:, 1:, :]

            # 标准化形状
            if patch_tokens is not None and patch_tokens.ndim == 2:  # [B,D] → [B,1,D]
                patch_tokens = patch_tokens.unsqueeze(1)
            if cls_token.ndim == 3:  # [B,1,D] → [B,D]
                cls_token = cls_token.squeeze(1)


            # 全部搬回同一设备
            cls_token = cls_token.to(pixel_values.device)
            if patch_tokens is not None:
                patch_tokens = patch_tokens.to(pixel_values.device)

            cls_token = self.image_proj(cls_token)  # [B,512]
            if patch_tokens is not None:
                patch_tokens = self.image_proj(patch_tokens)  # [B,L,512]

        if patch_tokens is not None and patch_tokens.shape[0] != cls_token.shape[0]:
            # hook 来自 block，形状是 [Seq, B, D]，把前两维换回来
            patch_tokens = patch_tokens.transpose(0, 1)  # → [B, Seq, D]
        # ---------- ③ WSAM / TCDAM 分支 ----------
        if self.use_wsam:
            task_feat = self.wsam(patch_tokens)  # [B, task_dim]
            task_feat = self.meta_condition(task_feat, task_id=self.task_id)
            task_feat = task_feat.mean(dim=1) if task_feat.ndim == 3 else task_feat
            if task_feat.ndim == 3:
                task_feat = task_feat.mean(dim=1)
            if task_feat.ndim == 3:
                task_feat = task_feat.mean(dim=1)

        else:
            task_feat = self.text_proj(cls_token)

        if self.use_tcdam:
            proj = self.tcdam(task_feat)
            image_feats = torch.bmm(cls_token.unsqueeze(1), proj).squeeze(1)
            image_feats = self.image_proj(image_feats)  # [B, proj_dim]
        else:
            image_feats = cls_token

        image_feats = self.l2(image_feats, dim=-1)  # L2-norm
        # 如果传入了 question_ids，就做 VQA 流程
        # ─── 在最尾部，把现有的 “if question_ids …” 全删，换成：
        if question_ids is not None and self.use_cross_modal:
            print("[DEBUG] 🔥 Running cross-modal fusion")
            vis = self.clip.encode_image(pixel_values, return_patch_tokens=True)
            cls_tok = vis["image_features"]
            patch_tokens = vis["patch_tokens"]


            # 1) 拿 pooled + patch tokens
            # if hasattr(self.clip.visual, "forward_features"):
            #     vis = self.clip.visual.forward_features(
            #         pixel_values, return_patch_tokens=True
            #     )
            #     cls_tok = vis["pooled_output"]
            #     patch_tokens = vis["patch_tokens"]
            # else:
            #     # HF Transformers CLIP 没有 forward_features，只能 plain forward
            #     vis = self.clip.visual(pixel_values)  # 不要 return_dict
            #     if isinstance(vis, torch.Tensor):
            #         # 极旧接口直接给 [B,D] 或 [B,1,D]
            #         cls_tok = vis.squeeze(1) if vis.ndim == 3 else vis
            #     else:
            #         # HF 有可能返回一个 ModelOutput
            #         cls_tok = getattr(vis, "pooler_output", None) \
            #                   or vis.last_hidden_state[:, 0]
            #     patch_tokens = None
            # [B,512]         [B,P,512] 或 [B*P,512]
            if patch_tokens.ndim == 2:
                patch_tokens = patch_tokens.view(
                    cls_tok.size(0), -1, cls_tok.size(1)
                )
            # 2) 投到 proj_dim（同 text_proj 输出维度）
            patch_tokens = self.patch_proj(patch_tokens)  # [B,P,256]
            q_feat = self.base.text_encoder(
                question_ids, attention_mask=question_mask
            ).pooler_output  # [B,512]
            q_feat = self.text_proj(q_feat)  # [B,256]
            # 3) cross‐modal fuse
            q_seq = q_feat.unsqueeze(1)  # [B,1,256]
            fused = self.cross_fuser(q_seq, patch_tokens)  # [B,1,256]
            q_feat = fused.squeeze(1)  # [B,256]
            # 4) 答案 embedding
            out_a = self.clip.text_encoder(
                answer_ids, attention_mask=answer_mask
            ).pooler_output  # [A,512]
            ans_emb = self.text_proj(out_a)  # [A,256]
            # 5) joint & logits
            image_feats = self.l2(self.image_proj(cls_tok), dim=-1)  # [B,256]
            joint = image_feats * q_feat
            return joint @ ans_emb.T

        # return self.cls(image_feats) if self.with_cls else image_feats
        if self.with_cls and self.task_id not in [3, 8]:  # 非 segmentation 任务（如 ChestXray、PCam）
            return self.cls(image_feats)
        elif self.task_id in [3, 8]:  # 3 = monuseg，分割任务
            seg_logits = self.seg_decoder(image_feats)  # [B, 1024]
            seg_mask = seg_logits.view(-1, 1, 32, 32)  # → [B,1,32,32]
            seg_mask = nn.functional.interpolate(seg_mask, size=(256, 256), mode="bilinear", align_corners=False)

            return seg_mask
        else:
            return image_feats


    def forward_seg(self, pixel_values):
        return self.forward(pixel_values)

    @torch.no_grad()
    def encode_image(self,
                     pixel_values,
                     prompt_ids: torch.LongTensor = None,
                     prompt_mask: torch.LongTensor = None):
        """
        如果 prompt_ids 不为 None，就做 cross-modal 融合后返回新的 image_feats；
        否则，沿用原来 WSAM/TCDAM → L2-norm 流程。
        """
        if getattr(self, "_skip_proj", False):
            out = self.clip.encode_image(pixel_values)
            cls = out["pooled_output"] if isinstance(out, dict) else out
            return self.l2(cls.float(), dim=-1)

        # 1) 取视觉输出
        if hasattr(self.clip.visual, "forward_features"):
            vis = self.clip.visual.forward_features(pixel_values, return_patch_tokens=True)
            cls_token = vis["pooled_output"]
            patch_tokens = vis["patch_tokens"]
        else:
            # —— 老接口：可能返回 Tensor / dict / ModelOutput / NamedTuple
            out = self.clip.visual(pixel_values)

            if isinstance(out, torch.Tensor):  # ① 直接给 CLS
                cls_token = out.squeeze() if out.ndim == 3 else out  # [B,D]
                patch_tokens = None
            else:  # ② dict / ModelOutput
                cls_token = (
                        getattr(out, "pooler_output", None) or
                        getattr(out, "pooled_output", None) or
                        getattr(out, "cls_embedding", None) or
                        getattr(out, "last_hidden_state", None)[:, 0]
                )
                patch_tokens = (
                        getattr(out, "patch_tokens", None) or
                        getattr(out, "x", None)
                )

            # 若 patch_tokens 是 [B,D] → [B,1,D]
            if patch_tokens is not None and patch_tokens.ndim == 2:
                patch_tokens = patch_tokens.unsqueeze(1)

        # 如果 patch_tokens 是 [B*P, D], 恢复成 [B,P,D]
        if patch_tokens is not None and patch_tokens.ndim == 2:
            B = cls_token.size(0)
            D = cls_token.size(-1)
            patch_tokens = patch_tokens.view(B, -1, D)

        # 2) 先把 pooled cls_token 投到 proj_dim（和 text_proj 一致）
        #    这样后面 cross_fuser 和 answer/text_proj 输出能对上
        cls_proj = self.image_proj(cls_token)    # [B, proj_dim]
        patches  = self.image_proj(patch_tokens) if patch_tokens is not None else None  # [B,P,proj_dim]

        # —— 如果没给 prompt，就回退到原来的 WSAM/TCDAM 分支 ——
        if prompt_ids is None:
            # ---------- 无 prompt：分类 / 检索 ----------
            if self.use_wsam and patches is not None:
                task_feat = self.wsam(patches)  # [B,256]
                print("after WSAM", task_feat.shape)
                task_feat = self.meta_condition(task_feat, self.task_id)
                task_feat = task_feat.mean(1) if task_feat.ndim == 3 else task_feat
            else:
                # 拿不到 patch_tokens → 用 fallback_proj 把 CLS(768) 压到 256
                task_feat = self.fallback_proj(cls_token)  # [B,256]
            if self.use_tcdam:
                proj = self.tcdam(task_feat)  # [B,512,256]
                image_feats = torch.bmm(cls_proj.unsqueeze(1), proj).squeeze(1)
            else:
                image_feats = cls_proj

            return self.l2(image_feats, dim=-1)

        # —— 3) 有 prompt 时，做 cross-modal ——
        # 3.1) text encode + project
        out_q = self.clip.text_encoder(
            prompt_ids,
            attention_mask=prompt_mask
        )
        q_feat = (out_q.pooler_output
                  if hasattr(out_q, "pooler_output") else
                  out_q.last_hidden_state[:,0])       # [B, D_vis]
        q_feat = self.text_proj(q_feat)            # [B, proj_dim]

        # 3.2) cross‐modal fuse
        q_seq = q_feat.unsqueeze(1)                # [B,1,proj_dim]
        # patches: [B,P,proj_dim]
        fused = self.cross_fuser(q_seq, patches)   # [B,1,proj_dim]
        q_feat = fused.squeeze(1)                  # [B, proj_dim]

        # 3.3) 直接把 q_feat 当 image_feats 返回
        #      （也可以做 element‐wise 或 concat，看你想要怎样利用）
        return self.l2(q_feat, dim=-1)

    @torch.no_grad()
    def encode_text(self, token_ids):
        if getattr(self, "_skip_proj", False):
            txt = self.clip.encode_text(token_ids)
            return self.l2(txt.float(), dim=-1)
        txt = self.clip.encode_text(token_ids)  # 原始 512 维
        txt = self.text_proj(txt)  # -> 256 维
        return self.l2(txt, dim=-1)
