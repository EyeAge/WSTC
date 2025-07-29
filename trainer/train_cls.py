# trainer/train_cls.py
import argparse, torch, numpy as np, time, pathlib, json
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score,accuracy_score, precision_recall_curve, roc_curve, balanced_accuracy_score
from tqdm import tqdm
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # 把项目根加入 PYTHONPATH

from runner.model_zoo import load_model
from runner.dataset_zoo import get_dataset

import warnings, timm
warnings.filterwarnings(
    "ignore",
    ".*timm.models.layers is deprecated.*",
    category=FutureWarning,
    module="timm",
)

from sklearn.metrics import precision_recall_curve, roc_auc_score

# def calculate_optimal_threshold(y_true, y_scores):
#     precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
#     f1_scores = 2 * (precision * recall) / (precision + recall)
#     optimal_threshold = thresholds[f1_scores.argmax()]
#     return optimal_threshold

def calculate_optimal_threshold(y_true, y_scores, eps=1e-8):
    """F1-max threshold（修好长度 + 避免 0/0）"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # thresholds.shape = (len(precision)-1,)
    f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + eps)
    return thresholds[f1.argmax()]

# ==== Temperature Scaling util ====
import torch.nn.functional as F
def temperature_scaling(logits: torch.Tensor, T: float = 1.0):
    """logits / T  →  calibrated log-prob (不改变量名)"""
    return logits / T

def weighted_vote(models, X_test, y_test, weights):
    """加权投票法"""
    all_predictions = np.array([model.predict(X_test) for model in models])
    weighted_predictions = np.average(all_predictions, axis=0, weights=weights)

    final_predictions = (weighted_predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_test, final_predictions)
    return final_predictions


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline',  default='unimedclip')
    p.add_argument('--variant',   default='baseline')  # baseline / +wsam / +wstc
    p.add_argument('--dataset',   default='chestxray14')
    p.add_argument('--kshot',     type=int, default=10)  # 每类正样本数
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs',    type=int, default=20)
    p.add_argument('--val_freq', type=int, default=1, help = 'Run validation every N epochs')
    p.add_argument('--val_lim', type=int, default=None, help='Validate on at most N random samples ' '(None = use the full validation set)')
    p.add_argument('--lr',        type=float, default=2e-4)
    p.add_argument('--seed',      type=int, default=0)
    p.add_argument('--temp', type=float, default=1.0, help='temperature for logit calibration')
    p.add_argument('--metric', choices=['acc', 'auc'], default=None)

    return p.parse_args()


def kshot_subset(ds, k=5, seed=0):
    """
    轻量级 k-shot 采样：
      - 绝不读取图片；只用内存里的 label 向量
      - 自动兼容 rows / _rows / df / labels 等多种写法
    """
    rng = np.random.default_rng(seed)

    # ---------- ① 取出标签矩阵 [N, C] ----------
    if hasattr(ds, "rows"):
        label_mat = np.asarray([r["labels"] for r in ds.rows])
    elif hasattr(ds, "_rows"):
        label_mat = np.asarray([r["labels"] for r in ds._rows])
    elif hasattr(ds, "df"):
        # pandas DataFrame：假设 0/1 列在 df[label_cols]
        label_cols = [c for c in ds.df.columns if c not in ("path", "image", "img")]
        label_mat  = ds.df[label_cols].values
    elif hasattr(ds, "labels"):
        label_mat = np.asarray(ds.labels)
    elif hasattr(ds, "Y"):
        label_mat = np.asarray(ds.Y)
    else:
        raise RuntimeError("找不到标签矩阵，无法构造 k-shot 子集")

    print("🔍 label_mat =", label_mat)
    print("🔍 label_mat.shape =", label_mat.shape)
    print("🔍 label_mat.ndim =", label_mat.ndim)
    # N, C = label_mat.shape
    # pos_pool = [np.where(label_mat[:, c] == 1)[0] for c in range(C)]
    if (label_mat.ndim == 1) or (label_mat.ndim == 2 and label_mat.shape[1] == 1):
        flat = label_mat if label_mat.ndim == 1 else label_mat.squeeze(1)

        pos_idx = np.where(flat == 1)[0]
        neg_idx = np.where(flat == 0)[0]

        sel_pos = rng.choice(pos_idx, min(k, len(pos_idx)), replace=False)
        sel_neg = rng.choice(neg_idx, min(k, len(neg_idx)), replace=False)

        sel_idx = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(sel_idx)
        return Subset(ds, sel_idx)
    else:
        # 多标签情况，例如 ChestXray14 (N, C)
        N, C = label_mat.shape
        pos_pool = [np.where(label_mat[:, c] == 1)[0] for c in range(C)]

    # ---------- ② 每类随机抽 k 张 ----------
    sel_idx = []
    for pool in pos_pool:
        n_take = min(k, len(pool))
        sel_idx.extend(rng.choice(pool, n_take, replace=False))

    rng.shuffle(sel_idx)          # 打乱一下
    return Subset(ds, sel_idx)

def main():
    args = parse()
    torch.manual_seed(args.seed)
    task_map = {"pcam": 0, "rsna": 1, "chestxray14": 2, "monuseg": 3,}
    task_id = task_map.get(args.dataset, 0)

    # ==== 模型 & 预处理 ====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(">>> 0  开始调 load_model")

    model, preprocess = load_model(args.baseline, device, 'fp16', variant=args.variant,num_classes=None, dataset_name=args.dataset, task_id=task_id)
    # 冻结 backbone
    # for p in model.clip.parameters(): p.requires_grad_(False)
    backbone = model.clip if hasattr(model, "clip") else model
    # 冻结它

    print(">>> 1  load_model 返回，准备做 requires_grad_")

    for p in backbone.parameters():
        p.requires_grad_(False)

    model.train(); model.to(device)
    print(">>> 2  freeze OK，开始构造 k-shot 数据集")

    model = model.float()
    # ==== 数据 ====
    root = f'datasets/{args.dataset}'
    full_ds, meta = get_dataset(args.dataset, transform=preprocess, root=root, split='train')
    train_ds = kshot_subset(full_ds, args.kshot, args.seed)
    task_type = meta.get("task", "binary")
    print(f">>> 3  k-shot subset 选出 {len(train_ds)} 张图")

    val_ds, _ = get_dataset(args.dataset, transform=preprocess, root=root, split='valid')

    if args.val_lim:
        val_idx = np.random.default_rng(args.seed).choice(
            len(val_ds), args.val_lim, replace=False)
        val_ds = Subset(val_ds, val_idx)
        # tune_idx = rng.choice(len(val_ds), int(0.1 * len(val_ds)), replace=False)
        # tune_ds = Subset(val_ds, tune_idx)
        # val_ds = Subset(val_ds, [i for i in range(len(val_ds)) if i not in tune_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4 ,persistent_workers=False )
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)


    if args.baseline.lower() in {"tip_adapter", "tip_adapter_f"}:
        # ---------- Tip-Adapter 系列 ----------
        cache_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=8)
        model.build_cache(cache_loader)

        if args.baseline.lower() == "tip_adapter_f":
            model.finetune_cache(cache_loader, epochs=args.epochs)

            # === 仅 Tip-Adapter-F 额外插入线性头并开启外层训练 ===
            D = model.txt.size(1)  # 文本特征维度
            model.cls = torch.nn.Linear(D, 1, bias=False).to(device)
            for p in model.cls.parameters():
                p.requires_grad_(True)
            model.with_cls = True  # 通过 if 判断
            requires_train = True  # 覆盖之前的 False
        else:
            requires_train = False
        model.eval()

    elif args.baseline.lower() in {"meta_adapter", "meta_adapter_f"}:
    # ---------- Meta-Adapter 系列 ----------
        support_loader = DataLoader(train_ds,
                                    batch_size=args.batch_size,
                                    shuffle=True, num_workers=8)
        model.adapt(support_loader, epochs=args.epochs, lr=args.lr)

        if args.baseline.lower() == "meta_adapter":
            # 纯 Meta-Adapter（论文原版）：适配完直接推理
            model.eval()
            requires_train = False

        else:  # ------ meta_adapter_f ------
            # 2) 给 backbone 接上线性头：① 创建 & ② 让参数需 grad
            if args.baseline.lower() == "meta_adapter_f":
                C = model.head.out_features  # safe & 一次到位

            # 2. 如果将来还想用 meta_adapter  + 自己加 cls-head，可以这样取
            else:
                C = len(model.txt)  # txt = [C,D]，行数就是类别数
            D = model.txt.size(1)
            model.head = torch.nn.Linear(D, C, bias=False).to(device)
            # 只训练 head
            for p in model.head.parameters():
                p.requires_grad_(True)

            model.eval()  # backbone & s 固定，只有 head 参与梯度
            requires_train = True
    else:
        # 走原始监督训练 (+wstc_train / baseline fine-tune 等)
        requires_train = True

    # ==== 优化器 ====
    head_params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    if requires_train:
        head_params = [p for p in model.parameters() if p.requires_grad]
        optim = AdamW(head_params, lr=args.lr, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()
    print(">>> 4  start training ...")


    best_auc, best_state = 0, None
    for ep in range(1, args.epochs + 1):

        # ------------- 训练 -----------------
        model.train()
        running_loss = 0.0

        prog = tqdm(enumerate(train_loader, 1),
                    total=len(train_loader),
                    desc=f"[ep {ep:02d}/{args.epochs}]",
                    leave=True)

        if requires_train:
            for step, (imgs, labels) in prog:
                imgs = imgs.to(device)
                labels = labels.to(device)

                if labels.ndim == 1:
                    labels = labels.unsqueeze(1)

                logits = model(imgs)  # [B, 15]
                # 判断是否是分类训练模式
                if hasattr(model, 'cls') and model.with_cls:
                    # loss = criterion(logits, labels)
                    if logits.ndim == 2 and logits.shape[1] == 2:  # Tip-Adapter 两列
                        logits = logits[:, 1].unsqueeze(1)  # 只取「正类」列 → [B,1]
                    loss = criterion(logits, labels)
                else:
                    raise RuntimeError("当前模型未启用分类头，不能进行监督训练")
                # loss = criterion(logits, labels)

                optim.zero_grad()
                loss.backward()
                optim.step()

                running_loss += loss.item()
                prog.set_postfix(loss=f"{running_loss / step:.4f}")
            print(f"[ep {ep:02d}] loss = {running_loss / len(train_loader):.4f}")
        else:
            print(f"[ep {ep:02d}] Tip-Adapter baseline —— 跳过训练阶段")

        # ---------- 验证（每 val_freq 轮一次） ----------
        if (ep % args.val_freq == 0) or (ep == args.epochs):
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                prog_val = tqdm(val_loader, desc=f"[val @ ep {ep:02d}]",
                                total=len(val_loader), leave=False)
                for imgs, labels in prog_val:
                    imgs = imgs.to(device, non_blocking=True)
                    if labels.ndim == 1:
                        labels = labels.unsqueeze(1)

                    # =================================================================================
                    # logits = model(imgs).sigmoid().cpu()  # ← 同步搬回 CPU
                    out = model(imgs).cpu()  # (B,2) or (B,1)
                    out = temperature_scaling(out, T=args.temp)  # ← 新增
                    if out.ndim == 2:
                        C = out.shape[1]
                        if C == 2:  # Tip-Adapter / Meta-Adapter
                            # logits = out[:, 1]  # 直接用正类列（不做 sigmoid）
                            probs = out.softmax(dim=1)  # ← NEW
                            logits = probs[:, 0]  # 正类概率
                        elif C == 1:  # 线性 cls-head 二分类
                            logits = out[:, 0].sigmoid()  # 做 sigmoid 得正类概率
                            print('Meta-Adapter 走掉了走掉了走掉了走掉了！！！！')
                        else:  # 多标签 / 多类
                            logits = out.sigmoid()  # 保留 (B,C) 做宏 AUC
                    else:  # (B,) 罕见形状
                        logits = torch.sigmoid(out)
                    # ==========================================================================

                    # ====== 在这里插入一次性打印 ======
                    if len(ys) == 0:  # 只在第一批次打印一次
                        print("label sample:", labels[:8].squeeze().tolist())

                        if logits.ndim == 2 and logits.shape[1] == 2 and task_type == "binary":
                            logits = logits[:, 0].unsqueeze(1)
                        if logits.ndim == 2 and logits.shape[1] >= 2:  # 二分类两列
                            print("logit sample:", logits[:8, 1].tolist())
                        else:  # (B,1) 形式
                            print("logit sample:", logits[:8].squeeze().tolist())
                    # ===================================

                    ys.append(labels)
                    ps.append(logits)

            # ys = torch.cat(ys).numpy()
            # ps = torch.cat(ps).numpy()
            ys = torch.cat(ys).cpu().numpy()
            ps = torch.cat(ps).cpu().numpy()
            # print(f"[DEBUG] ys shape {ys.shape}  ps shape {ps.shape}")  # ← 可留可删
            # print("ps unique:", np.unique(ps[:20]))
            #
            # optimal_threshold = calculate_optimal_threshold(ys, ps)
            #
            # # 使用最佳阈值对预测结果进行调整
            # preds = (ps >= optimal_threshold).astype(int)
            # # auc = roc_auc_score(ys, preds, average="macro")
            # # print(f"  ✔ Val AUC @ ep {ep:02d}: {auc:.4f}")
            # auc = roc_auc_score(ys, ps, average="macro")  # 真·AUC
            # acc = (preds == ys).mean()  # 可选，又快又直观
            # print(f"[Val] AUC {auc:.4f} | ACC {acc:.4f}")
            if task_type == "binary":
                auc = roc_auc_score(ys, ps)
                thr = calculate_optimal_threshold(ys, ps)
                preds = (ps >= thr).astype(int)
                # acc = (preds == ys).mean()

                acc = accuracy_score(ys, preds).mean()  # ② 普通 Accuracy
                bacc = balanced_accuracy_score(ys, preds).mean()  # ③ 建议同时报 BACC
                print(f"[Val] AUC {auc:.4f} | ACC {acc:.4f} | BACC {bacc:.4f}")

            elif task_type == "multiclass":
                preds = ps.argmax(1)
                acc = (preds == ys).mean()
                print(f"[Val] ACC {acc:.4f}")

            else:  # multilabel
                auc = roc_auc_score(ys, ps, average="macro")
                preds = (ps >= 0.5).astype(int)
                h_acc = (preds == ys).mean()
                subset_acc = accuracy_score(ys, preds)  # exact-match / subset accuracy

                print(f"[Val] macro-AUC {auc:.4f} | Hamming-ACC {h_acc:.4f} | Exact {subset_acc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_state = model.state_dict()

    # ==== 保存最优模型 ====
    save_dir = pathlib.Path('checkpoints'); save_dir.mkdir(exist_ok=True)
    ckpt_path = save_dir / f'{args.dataset}_{args.kshot}shot_{args.variant}.pt'
    torch.save(best_state, ckpt_path)
    print(f'★ Best AUC {best_auc:.4f}  saved to {ckpt_path}')

if __name__ == "__main__":
    main()
