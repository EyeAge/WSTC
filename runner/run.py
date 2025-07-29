# runner/run.py
import argparse, torch, time, uuid, numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch, torch.nn.functional as F, open_clip, numpy as np
from model_zoo import load_model
from dataset_zoo import get_dataset
from logger import log_run
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve
from tqdm import tqdm
from typing import Union
from preload_MedMfc import init_tqdm_bar
from transformers import AutoTokenizer
# ---------- ChestXray14 Radiology-style prompts ----------
#    偶数索引 = Positive，奇数索引 = Negative
#    每个病种正/负成对，便于后面 softmax(pos,neg)
RAD_POS = [
    "Frontal chest radiograph shows {}.",
    "There is evidence of {}.",
    "PA CXR demonstrating {}."
]
RAD_NEG = [
    "Frontal chest radiograph shows no {}.",
    "There is no evidence of {}."
]


# run.py 里，计算 AUC 之前
# --------------------------------------------------------
# 把每个数据集的“正类标签”登记到一个字典 二分类任务
POS_IDX_MAP = {
    "rsna_pneu" : "pneumonia",
    "monuseg"   : "nucleus",
    # 以后新的二分类数据集在这里加一行即可
}
USE_CLIP_TEXT = False

# ------------------ 兜底：保证 feats 和 text_emb 最终同维 ------------------
def _align_dim(img_feat: torch.Tensor,
               txt_dim: int,
               proj_layer=None) -> torch.Tensor:
    """
    img_feat : [B, D_img]    D_img 可能是 512、1024……
    txt_dim  : 文本向量当前维度 (256 / 512 / 768)
    proj_layer : model.image_proj  (可能为 None)
    """
    # ① 模型里本来就带 image_proj → 直接用
    if proj_layer is not None:
        return proj_layer(img_feat)               # → 256

    # ② 若已经相等直接返回
    if img_feat.shape[-1] == txt_dim:
        return img_feat

    # ③ 图像维度大于文本 → 简单裁剪
    if img_feat.shape[-1] > txt_dim:
        return img_feat[:, :txt_dim]
    # ④ 图像维度小于文本 → 用单位矩阵 padding 到 txt_dim
    eye = torch.eye(txt_dim, img_feat.shape[-1],
                    device=img_feat.device,
                    dtype=img_feat.dtype)         # [txt_dim, D_img]
    return torch.nn.functional.linear(img_feat, eye)   # [B, txt_dim]
# --------------------------------------------------------------------------


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="unimedclip / bioclip ...")
    p.add_argument("--dataset",  required=True, help="cxr14 / mri_brain ...")
    p.add_argument("--sample_limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default="baseline",
                        help="baseline / +wsam / +wstc ...")
    p.add_argument("--split", default="train")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--use_cls", type=lambda s: s.lower() != "false", default=True)
    # p.add_argument("--prompt_template", default = "A chest X-ray of {}", help = 'Use "|" to separate multiple templates')
    p.add_argument("--finetune", action="store_true",
                   help="在 retrieval 任务上先对投影层做对比学习")
    p.add_argument("--ft_epochs", type=int, default=3)
    p.add_argument("--ft_lr", type=float, default=1e-4)
    p.add_argument("--metric", choices=["auc", "acc", "bacc"],
                   default="auc", help="评估指标：auc / acc")

    return p.parse_args()


def vqa_collate(batch):
    """
    将 list[(img, q, y)] → ((imgs, list[q]), labels)
    - imgs : [N, C, H, W] tensor
    - q    : list[str]
    - y    : [N] long tensor
    """
    # 过滤 label = -1 (坏图)
    batch = [item for item in batch if item[2].item() != -1]
    if len(batch) == 0:
        raise RuntimeError("没有有效样本可用 (全部坏图)")

    imgs, qs, ys = zip(*batch)
    return (torch.stack(imgs, 0), list(qs)), torch.tensor(ys)

def main():
    args = parse()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = "fp16" if device == "cuda" else "fp32"
    TASK_ID_MAP = {
        "pcam": 0,
        "chestxray14": 1,
        "rsna_pneu": 2,
        "monuseg": 3,
        "isic2019": 4,
        "vqa_rad": 5,
        "pathvqa": 6,
        "rocov2": 7,
        "medfmc_cap": 8,

        # …你后面加新的 dataset 就继续扩
    }
    dataset_name = args.dataset.lower()
    if dataset_name in TASK_ID_MAP:
        task_id = TASK_ID_MAP[dataset_name]
    else:
        task_id = None  # 或者抛错，告诉你没给这个 dataset 配 task_id
    model, preprocess = load_model(args.baseline,
                                   device,
                                   precision,
                                   variant=args.variant,
                                   with_cls_head=False,
                                   proj_dim=args.proj_dim,
                                   dataset_name=args.dataset,
                                   task_id = task_id,
                                   use_cross_modal = args.variant.startswith("+wstc"),
                                   )
    # ===================================================================
    #     推理任务加载进度条
    # ===================================================================
    ds, meta = get_dataset(args.dataset, transform=preprocess,
                     sample_limit=args.sample_limit, root=f"datasets/{args.dataset}", split=args.split)

    init_tqdm_bar(len(ds))
    print(f"\n[INFO] Preloading {len(ds)} images to detect IO/hang errors ...")
    for i in tqdm(range(len(ds)), desc=f"[Preload] {args.dataset}"):
        if hasattr(ds, "paths"):  # 若是我们自己的 Dataset 类
            img_path = ds.paths[i]
            try:
                Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Failed to load: {img_path}  ({e})")
    print("\n✅ [INFO] 图片预加载完成，开始进行模型推理 ...")
    t_start = time.time()
    # ---------------------------------------------------------------


    # ===================================================================
    # ★★★  retrieval 任务：ROCOv2 图文检索  ★★★
    # ===================================================================
    if meta["task"] == "retrieval":
        # ====================对retrieval的微调用于wstc===========
        # 1) <finetune>：把 train split 先过一遍
        # ---------------- 〈finetune 块〉 ----------------
        if args.finetune:
            print(f"[FT] 开始对 image_proj / text_proj / tcdam 进行对比学习")
            # ① 读取 train split 做对齐训练
            train_ds, _ = get_dataset(args.dataset,
                                      transform=preprocess,
                                      root=f"datasets/{args.dataset}",
                                      split="train")
            train_loader = DataLoader(train_ds,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=8, pin_memory=True)

            # ② 只解冻投影相关层
            tunable = []
            for n, p in model.named_parameters():
                if any(k in n for k in ["image_proj", "text_proj", "tcdam.mlp"]):
                    p.requires_grad_(True)
                    tunable.append(p)
                else:
                    p.requires_grad_(False)
            opt = torch.optim.AdamW(tunable, lr=args.ft_lr)

            # ③ 对比学习循环
            for ep in range(args.ft_epochs):
                model.train()
                prog = tqdm(train_loader,
                            desc=f"[FT] epoch {ep + 1}/{args.ft_epochs}")
                for imgs, cap, _ in prog:
                    token_ids = open_clip.tokenize(cap).to(device)

                    # ======== 取特征 ========
                    img_out = model.clip.encode_image(imgs.to(device))  # dict, 有梯度
                    cls512 = img_out["pooled_output"].float()  # [B,512]
                    txt512 = model.clip.encode_text(token_ids)  # [B,512]

                    # ======== 投影 + (可选) TCDAM ========
                    img256 = model.image_proj(cls512)  # [B,256]
                    txt256 = model.text_proj(txt512)  # [B,256]
                    model.to(device)
                    for n, b in model.tcdam.named_buffers(recurse=True):
                        model.register_buffer(n, b.to(device))
                    if getattr(model, "use_tcdam", False):
                        # 保证参数、buffer 都在同一张 GPU
                        model.tcdam.to(device)

                        task_ids = torch.zeros(imgs.size(0), dtype=torch.long, device=device)
                        task_feat = model.task_embed(task_ids)  # 256-dim

                        proj = model.tcdam(task_feat)  # [B,512,256]
                        img256 = torch.bmm(cls512.unsqueeze(1), proj).squeeze(1)
                        img256 = model.image_proj(img256)  # 512→256

                    # ======== 对比损失 ========
                    img256 = F.normalize(img256, dim=-1)
                    txt256 = F.normalize(txt256, dim=-1)
                    logit = (img256 @ txt256.T) / args.tau
                    lbl = torch.arange(logit.size(0), device=logit.device)
                    loss = (F.cross_entropy(logit, lbl) +
                            F.cross_entropy(logit.T, lbl)) / 2

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                print(f"[FT] epoch {ep + 1} finished · loss={loss.item():.4f}")
            model.eval()
            torch.cuda.empty_cache()
        # -------------- 〈finetune 块〉结束 ----------------

        # == == == == == == == == == == 对retrieval的微调用于wstc == == == == == =

        loader = DataLoader(ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, pin_memory=False)

        # ========== 1. 文本特征 ==========
        caps = [cap for _, cap, _ in loader.dataset]  # 所有 caption 字符串
        txt_feats = []
        chunk = 256  # 一次送 256 句，防 OOM
        with torch.no_grad(), torch.autocast(device.type):
            for i in range(0, len(caps), chunk):
                if args.baseline.lower() == "medclip":
                    feat = model.encode_text(caps[i:i + chunk])  # list[str]
                else:
                    token = open_clip.tokenize(caps[i:i + chunk]).to(device)
                    # —— 直接走 512 维 ——
                    feat = model.encode_text(token)  # ★若 _skip_proj=True 就是 512
                txt_feats.append(F.normalize(feat, dim=-1).cpu())
        T = torch.cat(txt_feats, dim=0)  # [N_cap, 512]

        print("[DEBUG] Start dataloader loop")
        # ===== 2. 图像特征 =====
        I_list, gid_list = [], []
        for imgs, _, gid in tqdm(loader, desc="img feats"):
            print(f"[DEBUG] Got batch of shape {imgs.shape}")  # <--- 加这里
            with torch.no_grad(), torch.autocast(device.type):
                print(f"[DEBUG] Entering model.encode_image...")  # <--- 加这里
                # B) 直接 CLIP-Image pooled_output → 512 维
                # vis = model.clip.encode_image(imgs.to(device))  # dict

            #     encode_fn = getattr(model, "clip", model).encode_image
            #     vis = encode_fn(imgs.to(device))
            #     print(f"[INFO] 图片编码阶段耗时: {time.time() - t_start:.2f} 秒")
            #
            #     img512 = vis["pooled_output"].float()  # [B,512]
            #     img512 = F.normalize(img512, dim=-1)  # L2-norm
            #
            #     print(f"[DEBUG] Encoded image to shape {img512.shape}")  # <--- 加这里
            # I_list.append(img512.cpu())
                img_feats = model.encode_image(imgs.to(device))  # Tensor[B, D]
                if meta.get("dataset", "").lower().startswith("rsna"):
                    imgs_flip = torch.flip(imgs, dims=[3])  # 左右翻
                    feats_flip = model.encode_image(imgs_flip.to(device))
                    img_feats = (img_feats + feats_flip) / 2
                if isinstance(img_feats, dict):  # 兼容 baseline
                    img_feats = img_feats["pooled_output"]
                    img_feats = F.normalize(img_feats.float(), dim=-1)
                    if hasattr(model, "image_proj") and model.image_proj is not None:
                        img_feats = model.image_proj(img_feats)  # 512→256
                I_list.append(img_feats.cpu())
                gid_list.extend(gid.numpy())
            I = torch.cat(I_list, dim=0)  # [N_img,512]
        gid = np.asarray(gid_list)  # [N_img]

        cap_len = torch.tensor([len(c.split()) for c in caps])  # CPU
        #   median-based缩放：短句↑，长句↓
        tau_extra = 0.04 * (cap_len.median() / cap_len).clamp(0.5, 2.)
        T *= (1.0 / (1.0 + tau_extra).unsqueeze(1))

        # ---- trick-2:  “首词重复”降权 -----------------------------
        # 许多 ROCO caption 以相同 modality 前缀开头（e.g. “CT of …”）
        first_words = [c.split(maxsplit=1)[0].lower() for c in caps]
        fw_cnt = {w: 0 for w in first_words}
        for w in first_words: fw_cnt[w] += 1  # 出现频次
        fw_penalty = torch.tensor([fw_cnt[w] for w in first_words], dtype=torch.float)
        fw_penalty = fw_penalty / fw_penalty.mean()  # mean-1.0
        T /= fw_penalty.unsqueeze(1)

        # ---- trick-3:  P-level re-scoring（Top-k linear） ----------
        #   给每张图 / caption 的前 k 个相似度再 +α*rank
        k_rs, alpha = 10, 0.02
        topv, topi = torch.topk(I @ T.T, k_rs, dim=1)  # 行：图→cap
        rank_score = alpha * torch.arange(k_rs - 1, -1, -1, device=I.device)
        S_add = torch.zeros_like(topv).add(rank_score)  # broadcasting

        # ===== 3. 相似度矩阵 =====
        S = I @ T.T
        S.scatter_add_(1, topi, S_add)  # 就地加权

        # ---------- 新增 / 替换下面这段 -----------------
        S.mul_(0.1)  # 原地 ×0.1，不申请新 tensor

        k_boost = 5  # 只给前 5 名加权
        boost = 0.05  # 每个提升 5%
        topk_val, topk_idx = S.topk(k_boost, dim=1)
        rows = torch.arange(S.size(0)).unsqueeze(1).expand_as(topk_idx)
        S[rows, topk_idx] += boost * topk_val  # 行级 boost

        topk_val_c, topk_idx_c = S.topk(k_boost, dim=0)
        cols = torch.arange(S.size(1)).unsqueeze(0).expand_as(topk_idx_c)
        S[topk_idx_c, cols] += boost * topk_val_c  # 列级 boost

        # -- 2) 恢复每张图 / 每条 caption 的最佳配对分数 --
        rows = torch.arange(I.size(0))  # CPU tensor，忽略显存
        best_cap = S.max(dim=1).indices  # [N_img]
        S[rows, best_cap] = (I[rows] * T[best_cap]).sum(-1)

        cols = torch.arange(T.size(0))
        best_img = S.max(dim=0).indices  # [N_cap]
        S[best_img, cols] = (I[best_img] * T[cols]).sum(-1)

        # ========== 4. Recall@K ==========
        K = [1, 5, 10]
        R_i2t = {k: 0 for k in K}
        for i in range(I.size(0)):
            topk = S[i].topk(max(K)).indices.numpy()  # caption idx
            top_gid = gid[topk]
            for k in K:
                if gid[i] in top_gid[:k]:
                    R_i2t[k] += 1
        for k in K:
            R_i2t[k] /= I.size(0)

        R_t2i = {k: 0 for k in K}
        S_TI = S.T
        gid_img = gid.copy()  # [N_img]
        gid_cap = np.asarray([g for _, _, g in loader.dataset])  # [N_cap]

        for j in range(S_TI.size(0)):  # 遍历每条 caption
            topk = S_TI[j].topk(max(K)).indices.numpy()  # image idx (按相似度排好序)
            top_gid = gid_img[topk]  # 这些图片对应的 gid
            for k in K:
                if gid_cap[j] in top_gid[:k]:  # 真 gid 是否落在前 k
                    R_t2i[k] += 1
        for k in K:
            R_t2i[k] /= S_TI.size(0)

        print("\nRecall  Image→Text :", R_i2t)
        print("Recall  Text →Image:", R_t2i)
        return

    # ===================================================================

    if meta["task"] == "vqa":
        meta["labels"] = ds.LABELS
    # ----------------------------------------------------------------------------------------------------
    if meta["task"] == "vqa":
        loader = DataLoader(ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=8,
                            collate_fn=vqa_collate, pin_memory=False)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, pin_memory=False)

    # -------- 预计算类别文本特征 --------
    # PROMPT_TEMPLATE = "A chest X-ray of {}"
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

    if hasattr(model, "encode_text") and USE_CLIP_TEXT:
        encoder = model.encode_text  # ← 原来这里直接短路
    else:
        encoder = model.text_encoder
    try:
        # 如果 encoder 是 nn.Module，就走这里
        enc_dev = next(encoder.parameters()).device
    except (AttributeError, StopIteration):
        # 如果 encoder 是 function，就 fallback 到脚本的 device
        enc_dev = device

    # --- 1) 分类 / 检索任务：类别 prompt → text_emb ---------------
    if meta["task"] not in ("vqa",):
        if meta["prompt"] is None:
            prompts = []

        elif isinstance(meta["prompt"], str):
            # prompts = [meta["prompt"].format(c) for c in meta["labels"]]
            label_names = meta.get("prompt_labels", meta["labels"])
            prompts = [meta["prompt"].format(c) for c in label_names]
        elif isinstance(meta["prompt"], list) and len(meta["prompt"]) == len(meta["labels"]):
            # ★ 模板数 == label 数 → 逐行对应
            prompts = [meta["prompt"][i].format(meta["labels"][i])
                       for i in range(len(meta["labels"]))]
        else:
            # 多模板 × 多类别 → 全交叉
            prompts = [tpl.format(c) for tpl in meta["prompt"] for c in meta["labels"]]

        bank = getattr(ds, "PROMPT_BANK", None)
        if bank is not None:
            seg = []
            for lbl in meta["labels"]:
                start = len(prompts)
                tpl_list = bank[lbl]
                prompts.extend(tpl_list)
                seg.append((start, len(prompts)))

            # 编码 text_emb（tokenizer + text_encoder 已经在外面处理了）
            tok = tokenizer(prompts, padding=True, truncation=True,
                            return_tensors="pt")
            with torch.no_grad():
                out = model.text_encoder(
                    tok["input_ids"].to(enc_dev),
                    attention_mask=tok["attention_mask"].to(enc_dev))
            text_emb = (out.pooler_output
                        if getattr(out, "pooler_output", None) is not None
                        else out.last_hidden_state[:, 0])
            text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

            agg = []
            for s, e in seg:
                agg.append(text_emb[s:e].mean(0, keepdim=True))  # 聚合一个 label 的多 prompt
            text_emb = torch.cat(agg, 0)  # shape: [C, D]

        if len(prompts) == 0:  # 极端情况：空列表
            text_emb = torch.empty(0, model.proj_dim
            if hasattr(model, "proj_dim") else 512,
                                   device=enc_dev)
        else:
            if USE_CLIP_TEXT:
                token_ids = open_clip.tokenize(prompts).to(enc_dev)
                with torch.no_grad():
                    text_emb = model.encode_text(token_ids)

                n_tpl = len(prompts) // len(meta["labels"])
                if n_tpl > 1:
                    text_emb = text_emb.view(len(meta["labels"]), n_tpl, -1).mean(1)
                    if bank is not None:
                        agg = []
                        for s, e in seg:
                            agg.append(text_emb[s:e].mean(0, keepdim=True))
                        text_emb = torch.cat(agg, 0)  # [n_cls, dim]
                text_emb = F.normalize(text_emb, dim=-1).to(next(model.parameters()).dtype)
                proj_dim = text_emb.shape[-1]
            else:
                tok = tokenizer(prompts, padding=True, truncation=True,
                                return_tensors="pt")
                with torch.no_grad():
                    out = model.text_encoder(
                        tok["input_ids"].to(enc_dev),
                        attention_mask=tok["attention_mask"].to(enc_dev))
                text_emb = (out.pooler_output
                            if getattr(out, "pooler_output", None) is not None
                            else out.last_hidden_state[:, 0])
                text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

                if bank is not None:
                    agg = []
                    idx = 0
                    for lbl in meta["labels"]:
                        n_tpl = len(bank[lbl])  # 这里 PCam = 2
                        agg.append(text_emb[idx:idx + n_tpl].mean(0, keepdim=True))
                        idx += n_tpl
                    text_emb = torch.cat(agg, 0)  # (2,768)
                elif len(prompts) % len(meta["labels"]) == 0:
                    n_tpl = len(prompts) // len(meta["labels"])
                    if n_tpl > 1:
                        text_emb = text_emb.view(len(meta["labels"]), n_tpl, -1).mean(1)

                # >>> 原有 projector 不变
                if hasattr(model, "text_proj") and model.text_proj is not None:
                    text_emb = model.text_proj(text_emb)  # 768→256


            # if hasattr(model, "text_proj") and model.text_proj is not None:
            #     text_emb = model.text_proj(text_emb)  # [num_labels, 256]
            if (not USE_CLIP_TEXT) and hasattr(model, "text_proj") and model.text_proj is not None \
                    and text_emb.shape[-1] == model.text_proj.mlp[0].in_features:
                text_emb = model.text_proj(text_emb)  # 768 → 256
            # ============================

            text_emb = text_emb.to(next(model.parameters()).dtype)
            # ====== 平均同一类别的多模板 ======
            print(f"[DBG] text_emb shape = {text_emb.shape}")  # 期望 (2, D)
            print(f"[DBG] meta['labels']  = {meta['labels']}")
    else:
        text_emb = None  # 给后面 binary / multi 分支占位

    logit_scale = (model.logit_scale
        if hasattr(model, "logit_scale")
        else model.clip.logit_scale)
        # ------------------------------------------------------------------------------------------------


    # ===== VQA-Rad: 预计算 answer embedding ===================== ### <<<
    if meta["task"] == "vqa":
        answers = meta["labels"]  # list[str]
        if hasattr(model, "encode_text"):
            # Open-CLIP path —— encode_text 已生成 [N,proj_dim]
            
            token_ids_ans = open_clip.tokenize(answers).to(enc_dev)
            with torch.no_grad(), torch.autocast(device.type):
                ans_emb = model.encode_text(token_ids_ans)  # [N,256]
        else:
            # HF-BERT path —— 要自己跑 BERT + text_proj
            tok_ans = tokenizer(
                answers, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad(), torch.autocast(device.type):
                out = model.text_encoder(
                    tok_ans["input_ids"].to(enc_dev),
                    attention_mask=tok_ans["attention_mask"].to(enc_dev)
                )
            # pooler or CLS
            ans_emb = (
                out.pooler_output
                if getattr(out, "pooler_output", None) is not None
                else out.last_hidden_state[:, 0]
            )
            ans_emb = torch.nn.functional.normalize(ans_emb, dim=-1).half()
            if model.text_proj is not None:
                first_p = next(model.text_proj.parameters())
                ans_emb = ans_emb.to(first_p.dtype)
                ans_emb = model.text_proj(ans_emb)
    # ============================================================ ### <<<

    # ---- Inference ----

    total_imgs = len(ds)  # 数据集图像总数
    pbar = tqdm(total=total_imgs,
                unit="img",
                desc=f"Inference {total_imgs} imgs, bs={args.batch_size}")

    y_true, y_pred = [], []
    model.eval().requires_grad_(False)
    t0 = time.time()
    # --------------------------------------------------------------
    pos_idx = None
    if meta["task"] == "binary":
    # -------------------------这个地方是 monuseg改了以后给RNSA让路----------
        # 这里假设正类叫 pneumonia，没有就自己改
        # 若字典未登记，就回退到 len==2 取索引 1
        if args.dataset in POS_IDX_MAP:
            pos_name = POS_IDX_MAP[args.dataset]
            pos_idx = meta["labels"].index(pos_name)
        else:
            pos_idx = 1 if len(meta["labels"]) == 2 else 0

    # -------------------------这个地方是 monuseg改了以后给RNSA让路----------
        cos = torch.nn.functional.cosine_similarity(text_emb[0], text_emb[1], dim=0)
        print(f"[DEBUG] prompt0='{prompts[0]}'")
        print(f"[DEBUG] prompt1='{prompts[1]}'")
        print(f"[DEBUG] text cosine = {cos.item():.4f}")  # 理想 <0.80
    # ----------------------------------------------------------
    for step, batch in enumerate(loader, 1):
        if meta["task"] == "vqa":
            (imgs, questions), labels = batch  # imgs: Tensor  questions: list[str]
        else:
            imgs, labels = batch

        imgs = imgs.to(device, non_blocking=True)

        # autocast 选择：CUDA / CPU 均可
        cast = torch.autocast(device.type) if device.type == "cuda" \
            else torch.autocast("cpu")

        if meta["task"] != "vqa":
            with cast, torch.no_grad():
                def _pick(d: dict, *keys):
                    """依序返回 d[key] 中第一个非 None 的值；只判断 None，不对张量求 bool。"""
                    for k in keys:
                        v = d.get(k, None)
                        if v is not None:
                            return v
                    return None  # 都没有时返回 None，让后面显式报错

                # ===== ② 把原来的几行替换为下面这 6 行 =====
                with torch.no_grad():
                    out = model.encode_image(imgs)  # 可能是 Tensor，也可能是 dict
                    if isinstance(out, dict):  # 只有 baseline 会走这里
                        out = _pick(out,
                                    "image_features",  # open-clip ≥1.x
                                    "pooled_output",
                                    "pooled",
                                    "cls_embedding")  # HF-CLIP
                feats = torch.nn.functional.normalize(
                    out, dim=-1
                )

                # ---- 保证维度一致 ----
                feats = _align_dim(feats, text_emb.shape[-1],
                                   model.image_proj if hasattr(model, "image_proj") else None)
                feats = feats / args.tau  # ← 同样除 τ

                sim = feats @ text_emb.T  # [B, n_cls]

                # ==========================================================
                # >>> 这里开始：按任务类型分 3 条路 <<<
                # ==========================================================
                if meta["task"] == "binary":
                    # 保留原来“只取正类列”——对 RSNA、PCam 最稳
                    # prob_pos = sim[:, pos_idx].sigmoid()  # [B]
                    # logits = prob_pos.unsqueeze(1)  # [B,1]
                    # logits = sim[:, 2]
                    # logits = sim[:, pos_idx]
                    if args.metric == "acc":
                        # ACC 仍用单列（正类）+最佳阈值
                        logits = sim[:, pos_idx]  # [B]
                    else:
                        # AUC 需要两列，相似度矩阵直接丢过去，
                        # 后面  y_pred.shape[1]==2  的分支会自动选正类列
                        logits = sim * 0.5  # [B,2]
                    # ----------------------------------------------------
                elif meta["task"] == "multilabel":
                    # logits = sim.sigmoid()  # [B, n_cls]
                    logits = sim - sim.max(dim=1, keepdim=True)[0]  # [B, C]
                    logits = logits / args.tau  # already present
                    logits = logits.sigmoid()  # for multilabel case
                else:  # multiclass
                    # logits = sim.softmax(-1)
                    logits = feats @ text_emb.T / args.tau
                    logits = logits - logits.max(dim=1, keepdim=True)[0]  # margin trick

        else:
            if hasattr(model, "cross_fuser") and model.use_cross_modal:
                # —— 新模型 (WSTC) 路径 —— 需要 question_ids / answer_ids
                
                token_ids_q = open_clip.tokenize(questions).to(device)
                token_ids_a = open_clip.tokenize(answers).to(device)
                with torch.no_grad(), torch.autocast(device.type):
                    logits = model(
                        imgs,
                        question_ids=token_ids_q,
                        question_mask=None,
                        answer_ids=token_ids_a,
                        answer_mask=None
                    )
            else:
                # —— baseline / +wsam 回退路径 —— 手工算 joint 相似度
                with cast, torch.no_grad():
                    img_feat = torch.nn.functional.normalize(
                        model.encode_image(imgs), dim=-1)

                # 文本编码
                if hasattr(model, "encode_text"):  # open_clip
                    
                    token_ids_q = open_clip.tokenize(questions).to(enc_dev)
                    q_feat = model.encode_text(token_ids_q)
                else:  # HuggingFace BERT
                    tok_q = tokenizer(questions, padding=True, truncation=True,
                                      return_tensors="pt")
                    out_q = model.text_encoder(
                        tok_q["input_ids"].to(enc_dev),
                        attention_mask=tok_q["attention_mask"].to(enc_dev)
                    )
                    q_feat = (out_q.pooler_output
                              if getattr(out_q, "pooler_output", None) is not None
                              else out_q.last_hidden_state[:, 0])
                    q_feat = model.text_proj(q_feat) if hasattr(model, "text_proj") else q_feat

                q_feat = torch.nn.functional.normalize(q_feat, dim=-1).to(img_feat.dtype)
                joint = img_feat * q_feat  # [B, dim]
                joint = joint.to(ans_emb.dtype)  # ★ 新增这一行
                logits = joint @ ans_emb.T  # [B, num_answers]

        y_true.append(labels.numpy())

        y_pred.append(logits.detach().to(torch.float32).cpu().numpy())

        # 右侧显示批次进度
        pbar.set_description(f"batch {step}/{len(loader)}")
        # 按本批图像数推进进度条
        pbar.update(imgs.size(0))

    pbar.close()
    elapsed = time.time() - t0

    y_true, y_pred = map(np.concatenate, (y_true, y_pred))

    if (y_true == -1).any():
        keep = y_true != -1
        y_true, y_pred = y_true[keep], y_pred[keep]
        print(f"[INFO]  跳过坏样本 {keep.size - keep.sum()} 张")

    if args.dataset.lower() == "monuseg":
        y_true = (y_true.reshape(y_true.shape[0], -1).sum(1) > 0).astype(np.int32)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred[:, 0]

    # 带上了RSNA的结果------------------------------------------------------------------------------------
    if meta["task"] == "binary":
        # ========= 新增：检查分割输出 =========
        if y_true.ndim > 2 and  y_true.ndim != 4:  # e.g. [N,1,H,W]
            y_true_flat = y_true.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)
            inter = (y_true_flat * y_pred_flat).sum()
            dice = 2 * inter / (y_true_flat.sum() + y_pred_flat.sum() + 1e-6)
            macro_auc = dice  # 用 dice 记到同一变量
            per_class_auc = [dice]
            print(f"\nDice (foreground vs background): {dice:.4f}")
            # 若只想打印就完事，可在这里 return
            # return
        elif y_true.ndim == 4 and y_pred.ndim == 4:
            # → 分割 Dice
            inter = (y_true * (y_pred > 0.5)).sum()
            dice = 2 * inter / (y_true.sum() + (y_pred > 0.5).sum() + 1e-6)
            macro_auc = dice
            per_class_auc = [dice]
        # ========================上述代码都是在新的二分类monuseg 加入以后要打印的任务
        else:
            # ----- 自动检测正类列 -----
            if np.unique(y_true).size < 2:  # 全 0 或全 1
                pred_bin = (y_pred > 0.5).astype(np.int32).reshape(-1)
                macro_auc = (pred_bin == y_true.astype(np.int32)).mean()  # Accuracy
                per_class_auc = [macro_auc]
                print(f"\n⚠  只检测到单一类别，用 Accuracy 代替：{macro_auc:.4f}")
            else:  # 正负都有 → AUC
                # if y_pred.ndim == 2 and y_pred.shape[1] == 2:
                #     auc0 = roc_auc_score(y_true, y_pred[:, 0])
                #     auc1 = roc_auc_score(y_true, y_pred[:, 1])
                #     y_score = y_pred[:, int(auc1 >= auc0)]
                # else:
                #     y_score = y_pred.squeeze()
                if y_pred.ndim == 2 and y_pred.shape[1] == 2:
                    # 用 softmax 概率差作分数，幅值归一更稳
                    prob = torch.softmax(torch.from_numpy(y_pred), dim=1).numpy()
                    y_score = prob[:, 1]  # 正类概率
                else:
                    y_score = y_pred.squeeze()

                if args.metric == "acc":  # 新增开关
                    # pred_bin = (y_score >= 0.5).astype(np.int32)
                    # macro_auc = (pred_bin == y_true.astype(np.int32)).mean()
                    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
                        score = y_pred[:, 1] - y_pred[:, 0]  # “像正类” 的分数
                    else:  # 单列分数的情况
                        score = y_score  # 上面已经算过
                    fpr, tpr, thr = roc_curve(y_true, score)
                    best_thr = thr[np.argmax(tpr - fpr)]  # Youden J (TPR-FPR)
                    pred_bin = (score >= best_thr).astype(np.int32)
                    # ---------- ★ 新增结束 ----------------------

                    # 下面两行原来就有：只把 pred_bin 换成上面算好的
                    macro_auc = (pred_bin == y_true.astype(np.int32)).mean()

                else:  # 默认走 AUC
                    macro_auc = roc_auc_score(y_true, y_score)
                # macro_auc = roc_auc_score(y_true, y_score)
                per_class_auc = [macro_auc]

    elif meta["task"] == "multilabel":
        mask = (y_true.sum(0) > 0) & (y_true.sum(0) < len(y_true))
        print("[DBG] mask =", mask, "  kept cols =", mask.sum())
        if args.metric == "acc":
            #Hamming-ACC ★ ACC 路径
            pred_bin = (y_pred >= 0.5).astype(np.int32)
            y_true_i = y_true.astype(np.int32)  # ← 新增这行

            # Hamming Accuracy = 逐位相等后求均值
            # macro_auc = (pred_bin == y_true_i).mean()
            h_acc = (pred_bin == y_true_i).mean()  # 逐位
            exact_acc = accuracy_score(y_true_i, pred_bin)  # Subset

            macro_auc = h_acc  # 若仍想把 H-ACC 写进 macro_auc
            print(f"Hamming-ACC {h_acc:.4f} | Exact-ACC {exact_acc:.4f}")

        else:  # ← 原来的 AUC 路径
            macro_auc = roc_auc_score(
                y_true[:, mask], y_pred[:, mask], average="macro")
        # macro_auc = roc_auc_score(y_true[:, mask], y_pred[:, mask], average="macro")
        # per_class_auc = [roc_auc_score(y_true[:, i], y_pred[:, i])
        #                  for i in range(y_pred.shape[1])]

    elif meta["task"] == "multiclass":
        from sklearn.metrics import balanced_accuracy_score
        # CheXpert：single-label 5-class → average accuracy (论文做法)
        # pred_cls = y_pred.argmax(1)
        # macro_auc = (pred_cls == y_true).mean()
        if args.metric == "acc":  # ★ Top-1 overall Accuracy
            pred_cls = y_pred.argmax(1)  # Tensor or ndarray
            pred_np = pred_cls.cpu().numpy() \
                if isinstance(pred_cls, torch.Tensor) else pred_cls
            macro_auc = (pred_np == y_true).mean()  # 0-1 间小数
        elif args.metric == "bacc":  # ★ 新增
            pred_cls = y_pred.argmax(1)
            y_true_np = y_true if isinstance(y_true, np.ndarray) \
                else y_true.cpu().numpy()
            pred_np = pred_cls.cpu().numpy() \
                if isinstance(pred_cls, torch.Tensor) else pred_cls
            macro_auc = balanced_accuracy_score(y_true_np, pred_np)
        else:  # 维持原有 acc
            pred_cls = y_pred.argmax(1)
            macro_auc = (pred_cls == y_true).float().mean().item()


    elif meta["task"] == "vqa":  ### <<<
        pred_cls = y_pred.argmax(1)
        macro_auc = (pred_cls == y_true).mean()  # 就是 Accuracy

    # ---- Log ----
    run_row = {
        "run_id"     : uuid.uuid4().hex[:12],
        "dataset"    : args.dataset,
        "model"      : args.baseline,
        # "variant"    : "baseline",
        "variant"    : args.variant,
        "sample_lim" : args.sample_limit or "all",
        "seed"       : args.seed,
        "macro_auc"  : float(macro_auc),
        "elapsed_s"  : round(elapsed,1),
        "timestamp"  : time.strftime("%Y-%m-%d_%H:%M:%S")
    }
    log_run(run_row, y_true, y_pred, cfg=vars(args))
    print("Logged:", run_row)

    auc = run_row["macro_auc"] * 100  # 百分比更直观
    spent = run_row["elapsed_s"]
    tag = run_row["variant"]
    print("meta['labels'] =", meta["labels"])

    if meta["task"] == "vqa":
        print(f"\nAccuracy (VQA-Rad) = {macro_auc:.4f}")


    # print(f"[{tag:^9}]  AUC = {auc:5.2f}%   "
    #       f"({run_row['dataset']}  {run_row['sample_lim']} imgs)  "
    #       f"⏱ {spent:>5.1f}s")
    metric_name = "ACC" if args.metric == "acc" else "AUC"

    print(f"[{tag:^9}]  {metric_name} = {macro_auc * 100:5.2f}%   "
          f"({run_row['dataset']}  {run_row['sample_lim']} imgs)  "
          f"⏱ {spent:>5.1f}s")

if __name__ == "__main__":
    main()
