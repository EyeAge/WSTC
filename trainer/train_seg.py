import torch
import argparse, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # 把项目根加入 PYTHONPATH
from runner.dataset_zoo import get_dataset
from runner.model_zoo import load_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class DummySegLabelDataset(torch.utils.data.Dataset):
    """
    包装分割数据集，为 Tip-Adapter 提供 fake classification label（如1类）
    """
    def __init__(self, base_dataset, label_val=1):
        self.base_dataset = base_dataset
        self.label_val = label_val

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]  # 忽略原始 mask
        return img, self.label_val  # 返回 fake label

def visualize_batch(imgs, masks):
    for i in range(min(4, imgs.shape[0])):
        img = imgs[i].permute(1, 2, 0).cpu().numpy()
        mask = masks[i].squeeze().cpu().numpy()  # squeeze 掉 channel（如果有）

        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis('off')

        plt.subplot(2, 4, i + 5)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.show(block=False)  # 非阻塞显示
    plt.pause(3)  # 显示 3 秒
    plt.close()

# 创建 Dice 和 Lou值计算函数
def calculate_dice_and_lou(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    dice = 2 * intersection / (y_true.sum() + y_pred.sum() + 1e-6)
    lou = intersection / (y_true.sum() + y_pred.sum() - intersection + 1e-6)
    return dice, lou

def dice_loss(pred, target, smooth=1e-6):
    # pred 和 target 都是 float，target 也需要转换为 float
    pred = pred.contiguous()
    target = target.contiguous().float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target, smooth=1e-6):
    # Dice Loss
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = 2. * intersection / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    dice_loss = 1 - dice.mean()

    # BCE Loss
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='mean')

    # Combine both losses
    total_loss = dice_loss + bce_loss  # You can weight these losses if necessary
    return total_loss

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', default='unimedclip')
    p.add_argument('--variant', default='baseline')  # baseline / +wsam / +wstc
    p.add_argument('--dataset', default='monuseg')  # 对应分割任务
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse()
    torch.manual_seed(args.seed)
    task_map = {"pcam": 0, "rsna": 1, "chestxray14": 2, "monuseg": 3, "isic2018": 8 }
    task_id = task_map.get(args.dataset, 0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    requires_train = True
    model, preprocess = load_model(args.baseline, device, 'fp32', variant=args.variant, dataset_name=args.dataset, task_id=task_id)


    def freeze_partial_blocks(model, freeze_depth=6):
        blocks = model.clip.visual.transformer.resblocks
        for i, block in enumerate(blocks):
            if i < freeze_depth:
                for p in block.parameters():
                    p.requires_grad_(False)
        print(f"[INFO] 已冻结前 {freeze_depth} 层 block")

    # 应用于 UniMedCLIP 主干
    if hasattr(model.clip, 'visual') and hasattr(model.clip.visual, 'transformer') and hasattr(
            model.clip.visual.transformer, 'resblocks') and args.baseline.lower() in {"unimedclip", "+wsam", "+wstc"}:
        freeze_partial_blocks(model, freeze_depth=2)  # 传递整个模型，而不是 transformer 部分
    else:
        print("[WARN] model.clip.visual.blocks 不存在，跳过 freeze_partial")

    # 假设你想训练 wsam、tcdam 和 seg_head
    for name, param in model.named_parameters():
        if any(k in name for k in ["wsam", "tcdam", "seg_head"]):
            param.requires_grad_(True)
    model.train()
    model.to(device)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print("[DEBUG][Trainable] ", name)
    # 加载数据集
    root = f'datasets/{args.dataset}'
    train_ds, _ = get_dataset(args.dataset, transform=preprocess, root=root, split='train')
    val_ds, _ = get_dataset(args.dataset, transform=preprocess, root=root, split='test')

    # 创建数据加载器
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    # ========================DEBUG==========================================
    # imgs, masks = next(iter(train_loader))
    # visualize_batch(imgs, masks)
    # exit()
    # ====================================================

    # 定义优化器
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=3, verbose=True)
    if args.baseline.lower() in {"tip_adapter", "tip_adapter_f"}:
        print(">>> 使用 Tip-Adapter baseline")
        # 用 DummySegLabelDataset 包装原始分割数据集
        wrapped_ds = DummySegLabelDataset(train_ds, label_val=1)  # 或其他你想设的类别
        cache_loader = DataLoader(wrapped_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        model.build_cache(cache_loader)
        if args.baseline.lower() == "tip_adapter_f":
            model.finetune_cache(cache_loader, epochs=args.epochs)
        model.eval()
        requires_train = False

    elif args.baseline.lower() in {"meta_adapter", "meta_adapter_f"}:
        print(">>> 使用 Meta-Adapter baseline")
        # ⚠️ 包装一下，伪装成带“分类标签”的数据集
        # dummy_ds = DummySegLabelDataset(train_ds, num_classes=2)  # 或其他类数
        dummy_ds = DummySegLabelDataset(train_ds, label_val=0)  # ← 只要 < num_classes
        support_loader = DataLoader(dummy_ds, batch_size=args.batch_size,
                                    shuffle=True, num_workers=4)
        model.adapt(support_loader, epochs=args.epochs, lr=args.lr)

        if args.baseline.lower() == "meta_adapter":
            model.eval()  # 不训练，仅适配
            requires_train = False
        else:
            requires_train = True  # 可训练 seg_head（forward_seg）

    # for group in optim.param_groups:
    #     print("[debug] learning rate:", group["lr"])
    #     for param in group["params"]:
    #         print("[debug] grad exists:", param.requires_grad, param.shape)

    # 开始训练
    for epoch in range(args.epochs):
        # model.train()
        if requires_train:
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        prog = tqdm(train_loader, total=len(train_loader), desc=f"[Epoch {epoch + 1}/{args.epochs}]")

        for imgs, masks in prog:
            imgs = imgs.to(device)
            masks = masks.to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            # print("[debug] mask unique values:", masks.unique())
            optim.zero_grad()
            # outputs = model(imgs) # 这里假设模型输出的是分割的预测结果
            if hasattr(model, "forward_seg"):
                outputs = model.forward_seg(imgs)
            else:
                outputs = model(imgs)
            # ⬇️ 打印 tensor 形状（关键！）
            with torch.no_grad():

                dice, lou = calculate_dice_and_lou(masks, (outputs > 0.5).float())

            # loss = -dice  # 你可以根据需要调整损失函数
            # loss = -dice.mean()  # 确保是 Tensor，并加入 batch 维度均值
            loss = dice_loss(outputs, masks.float())
            # loss = combined_loss(outputs, masks.float())  # 使用联合损失函数
            # loss = 0.5 * dice_loss(outputs, masks.float()) + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
            #     outputs, masks.float())

            if requires_train:
                loss.backward()
                optim.step()
            # loss.backward()
            # optim.step()

            running_loss += loss.item()

            prog.set_postfix(loss=f"{running_loss / len(prog):.4f}", dice=dice, lou=lou.item())
        scheduler.step(running_loss)  # 基于训练损失或验证损失来调整学习率
        print(
            f"[Epoch {epoch + 1}] Loss: {running_loss / len(train_loader):.4f}, Dice: {dice.item():.4f}, Lou: {lou.item():.4f}")

        # 每个epoch后验证
        model.eval()
        with torch.no_grad():
            val_dice, val_lou = 0, 0
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                # outputs = model(imgs)
                # val_dice_batch, val_lou_batch = calculate_dice_and_lou(masks, outputs)
                outputs = model(imgs)
                if hasattr(model, "forward_seg"):
                    outputs = model.forward_seg(imgs)

                # 确保维度一致：squeeze 掉 channel=1，或 expand 一致
                if outputs.ndim == 4 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)  # [B, H, W]
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)  # [B, H, W]

                # 注意 threshold，输出为二值 mask
                val_dice_batch, val_lou_batch = calculate_dice_and_lou(masks, (outputs > 0.5).float())

                val_dice += val_dice_batch
                val_lou += val_lou_batch

            print(f"[Validation] Dice: {val_dice / len(val_loader):.4f}, Lou: {val_lou / len(val_loader):.4f}")


if __name__ == "__main__":
    main()
