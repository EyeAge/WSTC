# runner/dataset_zoo.py
# ---------- 公共函数：XML → mask ----------
import inspect
import sys
from PIL import Image, ImageDraw
from pathlib import Path
from tifffile import imread
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision import transforms as T
# from preload_MedMfc import update_bar


def rasterize_xml(xml_path, H=1000, W=1000):
    from PIL import Image, ImageDraw
    import numpy as np
    import xml.etree.ElementTree as ET
    import torch

    # 创建 PIL Image 来绘图
    mask_img = Image.new("L", (W, H), 0)  # 灰度图，背景为 0
    draw = ImageDraw.Draw(mask_img)

    # 解析 XML 结构
    root = ET.parse(xml_path).getroot()
    for reg in root.findall(".//Region"):
        pts = [(float(pt.get("X")), float(pt.get("Y")))
               for pt in reg.findall(".//Vertex")]
        draw.polygon(pts, outline=1, fill=1)

    mask = np.array(mask_img, dtype=np.uint8)
    # print(f"[debug]rasterize_xml 函数里面的[xml->mask_big] mask shape  = {mask.shape}, unique = {np.unique(mask)}")
    return torch.from_numpy(mask)


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, torch, pandas as pd
import numpy as np
from PIL import UnidentifiedImageError

class _BaseCXR(Dataset):
    LABELS  : list[str]  # 必填
    PROMPTS : str | list[str] = "A CXR of {}."
    TASK    : str        = "binary"  # multilabel / multiclass

def register_dataset(name: str):
    """装饰器：自动写进全局表，get_dataset 里不再 if-else"""
    def wrap(cls):
        _DATASETS[name.lower()] = cls
        return cls
    return wrap

_DATASETS = {}


# --- ChestXray14 dataset class ---
@register_dataset("chestxray14")
class ChestXray14(Dataset):
    LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
        "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
        "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
    ]
    CLASSES = LABELS          # ← 加这一行别名
    # PROMPT = "A chest X-ray showing {}."  # str 或 list[str]
    PROMPT = "A lateral chest X-ray showing {}"
    PROMPT_BANK = {
        "Cardiomegaly": [
            "a chest X-ray with enlarged heart shadow",
            "a chest radiograph showing cardiomegaly"
        ],
        "Effusion": [
            "a chest X-ray indicating pleural effusion",
            "a radiograph with fluid in the pleural space"
        ],
        "Pneumonia": [
            "a chest radiograph with signs of pneumonia",
            "frontal lung image showing pneumonia infiltration"
        ],
        "Infiltration": [
            "a chest X-ray with diffuse infiltration in the lung",
            "a chest radiograph showing patchy infiltration"
        ],
        "Atelectasis": [
            "a radiograph showing collapsed lung lobe",
            "a chest X-ray indicating atelectasis"
        ],
        "Mass": [
            "a chest X-ray with a pulmonary mass",
            "a chest radiograph showing abnormal lung mass"
        ],
        "Nodule": [
            "a radiograph with a small lung nodule",
            "a chest X-ray showing solitary pulmonary nodule"
        ],
        "Edema": [
            "a chest radiograph showing pulmonary edema",
            "a chest X-ray indicating fluid overload in lungs"
        ],
        "Consolidation": [
            "a chest X-ray showing lung consolidation",
            "a radiograph with homogeneous opacity in lung"
        ],
        "Emphysema": [
            "a chest X-ray with hyperinflated lungs",
            "a radiograph showing signs of emphysema"
        ],
        "Fibrosis": [
            "a chest X-ray with fibrotic lung changes",
            "a radiograph indicating pulmonary fibrosis"
        ],
        "Pleural_Thickening": [
            "a chest X-ray showing pleural thickening",
            "a radiograph with thickened pleural lining"
        ],
        "Pneumothorax": [
            "a chest radiograph showing collapsed lung",
            "a chest X-ray indicating pneumothorax"
        ],
        "Hernia": [
            "a chest X-ray showing diaphragmatic hernia",
            "a radiograph with herniated abdominal content"
        ],
        "No Finding": [
            "a normal chest radiograph without abnormalities",
            "a chest X-ray showing clear lungs and heart"
        ],
    }
    TASK = "multilabel"  # multilabel / binary / multiclass
    def __init__(self, root, transform, csv="Data_Entry_2017_v2020.csv",
                 sample_limit=None):
        df = pd.read_csv(os.path.join(root, csv))
        df = df[df["Finding Labels"].notna()].reset_index(drop=True)
        # if sample_limit: df = df.sample(sample_limit, random_state=42)
        if sample_limit:
            df = (df.sample(sample_limit, random_state=42).reset_index(drop=True))
        self.df = df; self.root = os.path.join(root, "images_flat")
        self.t = transform
        self.cache = {i:set(r["Finding Labels"].split("|"))
                      for i,r in df.iterrows()}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.t(Image.open(os.path.join(self.root,row["Image Index"]))
                     .convert("RGB"))
        label = torch.tensor([c in self.cache[idx] for c in self.CLASSES],
                             dtype=torch.float32)
        return img, label

# =================  RNSA  ====================
@register_dataset("rsna_pneu")
class RSNAPneuDataset(Dataset):
    # LABELS = ["normal lungs", "pneumonia"]
    LABELS = ["normal", "pneumonia"]  # 2-类
    # PROMPT = [
    #     "Frontal chest radiograph showing **clear lungs**.",
    #     "Frontal chest radiograph showing **pneumonia**."
    # ]
    PROMPT = "A chest X-ray showing {}."
    PROMPT_BANK = {
        "pneumonia": [
            "a chest X-ray showing consolidation in the lungs",
            "a chest radiograph indicative of pneumonia-related infiltration",
            "frontal chest image with signs of pneumonia"
        ],
        "normal": [
            "a normal chest X-ray with no signs of disease",
            "lungs appear clear and normal in the radiograph",
            "no abnormalities in the chest X-ray"
        ],
    }
    TASK = "binary"
    LABEL_MAP = {"Normal": 0.0, "Pneumonia": 1.0}   # 新增
    def __init__(self, root, transform, sample_limit=None):   ### ← 接收 sample_limit
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        if sample_limit:
            df = df.sample(sample_limit, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(frac=1).reset_index(drop=True)     # 打乱
        df["label_num"] = df.iloc[:, 1].map(self.LABEL_MAP).astype("int8")
        self.labels = df["label_num"].values  # (N,) ndarray
        self.id_series = df.iloc[:, 0]  # id/文件名
        # 把 df 自己留作索引用，不再让 kshot_subset() 看到两列字符串
        self.df = df[["label_num"]]

        self.root = os.path.join(root, "images")
        self.tfm = transform

        # self.labels = df.iloc[:, 1].map(self.LABEL_MAP).astype("int8").values

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        pid   = self.id_series.iloc[idx]
        label = self.labels[idx]

        # img = Image.open(f"{self.root}/{pid}.png").convert("L")
        try:
            img = Image.open(f"{self.root}/{pid}.png").convert("L")
        except (UnidentifiedImageError, OSError):
            # —— 遇到坏图：随便递归拿下一个样本（最简单做法）
            return self.__getitem__((idx + 1) % len(self))
        img = self.tfm(img)

        # label = np.array([self.LABEL_MAP[label_str]], dtype=np.float32)
        # return img, label
        return img, torch.tensor(float(label))

# =================  MoNuSeg  (instance-seg tile dataset)  ================
@register_dataset("monuseg")
class MoNuSegTile(Dataset):
    """
    将 1000×1000 病理图切成 256×256 tile；若 annots/ 下存在 xml，
    则返回二值 mask，否则返回 0-mask（可用于 test 集零样本推理）。
    """
    LABELS  = ["nucleus"]              # 单前景类
    PROMPT  = ["A microscopy image of a cell nucleus",
               "A microscopy image of background tissue", "An H&E stained histology image {}."]
    TASK    = "binary"

    def __init__(self, root, transform, split="train",
                 tile_size=256, overlap=64, sample_limit=None):
        from xml.etree.ElementTree import parse          # 仅此处用到
        from PIL import Image, ImageDraw

        self.img_dir = os.path.join(root, split, "images")
        # self.ann_dir = os.path.join(root, split, "annots")
        if split == "test":
            self.ann_dir = self.img_dir  # test 的 xml 混在 images 里
        else:
            self.ann_dir = os.path.join(root, split, "annots")
        self.tfm = transform
        self.size, self.stride = tile_size, tile_size - overlap

        self.files = [f for f in os.listdir(self.img_dir) if f.endswith(".tif")]
        if sample_limit:
            self.files = self.files[:sample_limit]

        # —— 预索引所有 tile 坐标 (img_idx, y, x)
        self.tiles = []
        for i, fname in enumerate(self.files):
            for y in range(0, 1000 - tile_size + 1, self.stride):
                for x in range(0, 1000 - tile_size + 1, self.stride):
                    self.tiles.append((i, y, x))

        # —— 简易 XML→mask 缓存（训练或离线评估时才用）
        self._xml_cache = {}
        # from functools import lru_cache
        # self._xml_cache = lru_cache(maxsize=None)(rasterize_xml)

    def __len__(self): return len(self.tiles)

    def __getitem__(self, idx):
        img_idx, y, x = self.tiles[idx]
        img_path = os.path.join(self.img_dir, self.files[img_idx])
        # img = Image.open(img_path).convert("RGB")
        img = load_image(img_path, 'monuseg')
        img = self.tfm(img.crop((x, y, x + self.size, y + self.size)))

        # -------- mask（如果 annots 有 GT，test 集则全 0） ----------
        xml_path = os.path.join(
            self.ann_dir, self.files[img_idx].replace(".tif", ".xml"))
        if os.path.exists(xml_path):
            # print(f"[debug][xml found] {xml_path}")
            if xml_path not in self._xml_cache:
                # print(f"[debug][xml->mask_big] Parsing {xml_path}")
                self._xml_cache[xml_path] = rasterize_xml(xml_path)
            mask_big = self._xml_cache[xml_path]
            # print(f"[debug][xml->mask_big] mask shape = {mask_big.shape}, unique = {torch.unique(mask_big)}")
            mask = mask_big[y:y + self.size, x:x + self.size].unsqueeze(0)
        else:
            # print(f"[debug][xml missing] {xml_path}")
            mask = torch.zeros(1, self.size, self.size, dtype=torch.uint8)

        # print("[debug] mask info", mask.shape, mask.dtype, np.unique(mask))
        # print("image path =", img_path)
        # print("mask path  =", xml_path)
        # print("mask unique =", mask.unique())  # 加这个
        # print("self._xml_cache =", self._xml_cache)
        if xml_path not in self._xml_cache:
            mask_big = rasterize_xml(xml_path)
            # print("[debug][xml->mask_big]", mask_big.shape, mask_big.dtype, np.unique(mask_big))
            self._xml_cache[xml_path] = mask_big

        return img, mask.float()

# ===============  ISIC-2019 dermoscopy (官方 split)  ===============
@register_dataset("isic2019")
class ISIC2019(Dataset):
    """
    官方 8-类单标签分类（multiclass）。目录结构示例：

    datasets/
      isic2019/
        train/   *.jpg  (25 331)
        test/    *.jpg   (8 238)
        ISIC_2019_Training_GroundTruth.csv
        ISIC_2019_Test_GroundTruth.csv
    """
    LABELS = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
    PROMPT_LABELS = [
        "Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions",
        "Dermatofibroma", "Melanoma", "Melanocytic nevi",
        "Squamous cell carcinoma", "Vascular lesions"
    ]
    # PROMPT = "A dermoscopic image of {}"
    PROMPT = "A dermoscopic image showing {}"
    PROMPT_BANK = {
        "AK": [
            "a dermoscopic image of actinic keratosis",
            "scaly erythematous lesion consistent with AK",
            "sun-damaged skin patch diagnosed as actinic keratosis",
            "keratotic plaque under dermoscopy"
        ],
        "BCC": [
            "basal cell carcinoma under dermoscopy",
            "pearly tumor with arborizing vessels",
            "nodular lesion showing basal cell carcinoma patterns",
            "pigmented BCC in a close-up dermoscopic image"
        ],
        "BKL": [
            "benign keratosis-like lesion in dermoscopy",
            "seborrheic keratosis with milia-like cysts",
            "dermatoscopy of benign keratosis",
            "waxy brown plaque resembling keratosis"
        ],
        "DF": [
            "dermatofibroma in a dermoscopic view",
            "central white scar-like area of dermatofibroma",
            "firm papule diagnosed as dermatofibroma",
            "dot-like and streak-like pigmentation of DF"
        ],
        "MEL": [
            "melanoma under dermoscopy",
            "asymmetric pigmented lesion suspicious for melanoma",
            "irregular network and blue-white veil of melanoma",
            "malignant melanoma pattern in a dermoscopic image"
        ],
        "NV": [
            "benign melanocytic nevus dermoscopic image",
            "common mole with regular pigment network",
            "symmetrical nevus under dermoscopy",
            "pigmented nevus showing homogenous pattern"
        ],
        "SCC": [
            "squamous cell carcinoma dermoscopy",
            "keratinous lesion indicative of SCC",
            "hyperkeratotic tumor with white circles sign",
            "dermoscopic appearance of invasive SCC"
        ],
        "VASC": [
            "vascular lesion on dermoscopy",
            "hemangioma presenting red lacunae",
            "angioma with multiple red lagoons",
            "vascular skin lesion showing lacunar pattern"
        ],
    }

    TASK   = "multiclass"              # 8-类单标签

    _LABEL_MAP = {lab: i for i, lab in enumerate(LABELS)}

    def __init__(self, root, transform,
                 split="train", kshot=None, seed=0,
                 sample_limit=None):
        csv_file = ("ISIC_2019_Training_GroundTruth.csv"
                    if split in ["train", "val"] else
                    "ISIC_2019_Test_GroundTruth.csv")
        df = pd.read_csv(os.path.join(root, csv_file))

        # --------  将 8 个 0/1 列 → 单整型 label_idx  --------
        df["label_idx"] = df[self.LABELS].values.argmax(axis=1)

        # --------  few-shot 采样（可选）  --------
        if kshot:
            rng = np.random.RandomState(seed)
            df = (df.groupby("label_idx", group_keys=False)
                    .apply(lambda x: x.sample(n=min(kshot, len(x)),
                                              random_state=rng)))
        if sample_limit:
            df = df.sample(sample_limit, random_state=seed)

        self.df  = df.reset_index(drop=True)
        self.dir = os.path.join(root, split)
        self.tfm = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.dir, f"{row.image}.jpg")
        img = self.tfm(Image.open(img_path).convert("RGB"))
        y   = torch.tensor(row.label_idx, dtype=torch.long)
        return img, y

# ==============  PatchCamelyon (PCam) 96×96 patch dataset  ==============
@register_dataset("pcam")  # 传参 --dataset pcam
class PCamDataset(Dataset):
    LABELS = ["normal", "tumor"]
    PROMPT = "A histopathology patch showing {} tissue"
    TASK   = "binary"
    PROMPT_BANK = {
        "tumor": [
            "an H&E stained histopathology patch showing malignant tumor cells",
            "a high-resolution micrograph with carcinoma tissue"
        ],
        "normal": [
            "an H&E stained histology patch of benign tissue",
            "a micrograph of healthy tissue without malignancy"
        ],
    }

    def __init__(self, root, transform, split="train",
                 kshot=None, seed=0, sample_limit=None):
        import numpy as np, os
        self.x_path = os.path.join(root,
            f"camelyonpatch_level_2_split_{split}_x.h5")
        self.y_path = os.path.join(root,
            f"camelyonpatch_level_2_split_{split}_y.h5")

        import h5py
        with h5py.File(self.y_path, "r") as f:
            full_y = f["y"][:]

        ids = np.arange(len(full_y))
        if kshot:
            rng = np.random.RandomState(seed)
            pos = ids[full_y == 1]
            neg = ids[full_y == 0]
            ids = np.concatenate([rng.choice(pos, kshot, False),
                                  rng.choice(neg, kshot, False)])
            rng.shuffle(ids)
        if sample_limit:
            ids = ids[:sample_limit]

        self.ids     = ids.astype(np.int64)
        # self.labels  = full_y[self.ids].astype(np.float32)
        self.labels = full_y[self.ids].astype(np.float32)
        if self.labels.ndim > 1:
            self.labels = self.labels.squeeze()
        self.tfm     = transform

        # 句柄占位：真正到 worker 里再打开
        self._h5x = None

    def _ensure_open(self):
        if self._h5x is None:           # 每个 worker 只会进来一次
            import h5py
            self._h5x = h5py.File(self.x_path, "r")["x"]

    def __len__(self): return len(self.ids)

    # def __getitem__(self, idx):
    #     self._ensure_open()
    #     i = self.ids[idx]
    #     img = self._h5x[i]              # uint8 HWC
    #     img = self.tfm(Image.fromarray(img))
    #     label = torch.tensor(self.labels[idx], dtype=torch.float32)
    #     return img, label
    def __getitem__(self, idx):
        self._ensure_open()
        i = int(self.ids[idx])
        img = self._h5x[i]  # uint8 HWC
        img = self.tfm(Image.fromarray(img))

        # ① 取标量 .item()；② 不再额外包装 list
        scalar_label = float(self.labels[idx])  # 0.0 / 1.0
        label = torch.tensor(scalar_label, dtype=torch.float32)  # 0-D tensor

        return img, label

# --------------------------  VQA-RAD  --------------------------- #
@register_dataset("vqa_rad")
class VQARad(Dataset):
    """
    VQA-RAD 数据集       (https://doi.org/10.48550/arXiv.1904.08920)
    ├── images/       synpic1234.jpg / png ...
    ├── train.json
    ├── val.json
    └── test.json

    每条 JSON:
    {
      "image_name": "synpic12345.jpg",
      "question":   "Is there a fracture?",
      "answer":     "yes"
    }
    """

    TASK = "vqa"          # 供 run.py 分流
    PROMPT = []           # VQA 本身不用 prompt

    # 初始化参数:
    #   root : 数据根目录
    #   transform : 视觉预处理
    #   split : train / val / test
    #   build_vocab : 是否自动统计答案表
    #   top_k : 仅保留出现频次最高的前 k 个答案 (None=全量)
    #   sample_limit : 随机抽样 n 条 (调试用)
    def __init__(self, root, transform,
                 split="test",
                 build_vocab=True,
                 top_k=None,
                 sample_limit=None):

        from collections import Counter
        import json, random
        from pathlib import Path

        self.tfm   = transform
        self.split = split.lower()
        self.img_dir = Path(root) / "images"

        # ---------- 读 JSON ----------
        json_path = Path(root) / f"{self.split}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"[VQARad] 找不到 {json_path}")
        with open(json_path, "r", encoding="utf-8") as fp:
            samples = json.load(fp)

        # ---------- 样本抽样 (可选) ----------
        if sample_limit:
            random.seed(42)
            random.shuffle(samples)
            samples = samples[:sample_limit]

        # ---------- 构建答案表 ----------
        if build_vocab:
            # 统计频次
            ans_freq = Counter(s["answer"].lower() for s in samples)
            if top_k:
                ans_list = [a for a, _ in ans_freq.most_common(top_k)]
            else:
                ans_list = sorted(ans_freq)          # 全量 (≈2k)
        else:
            ans_list = ["no", "yes"]                 # 只做二分类

        self.ans2id = {a: i for i, a in enumerate(ans_list)}
        self.LABELS = ans_list  # 实例属性
        self.__class__.LABELS = ans_list  # 供 run.py 读取

        # ---------- 过滤缺图 & 不在 vocab 的答案 ----------
        self.samples: list[dict] = []
        miss_img, out_vocab = 0, 0
        for s in samples:
            stem = Path(s["image_name"]).stem.lower()
            jpg  = self.img_dir / f"{stem}.jpg"
            png  = self.img_dir / f"{stem}.png"
            img_path = jpg if jpg.is_file() else png

            if not img_path.is_file():
                miss_img += 1
                continue
            ans = s["answer"].lower()
            if ans not in self.ans2id:
                out_vocab += 1
                continue
            s["img_path"] = img_path
            self.samples.append(s)

        kept = len(self.samples)
        print(f"[VQARad] {split}  读取 {kept} 条 | 缺图 {miss_img} | "
              f"答案不在 vocab {out_vocab}")

        if kept == 0:
            raise RuntimeError("[VQARad] 无可用样本！检查数据路径或 vocab 设置")

    # ---------- 必备接口 ----------
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 图像
        try:
            img = self.tfm(Image.open(s["img_path"]).convert("RGB"))
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            # 坏图兜底：给个全 0 tensor，并返回 label = -1 方便上层过滤
            C, H, W = 3, self.tfm.crop_size, self.tfm.crop_size
            return torch.zeros(C, H, W), "", torch.tensor(-1)

        # 文本 & 标签
        q = s["question"]
        y = torch.tensor(self.ans2id[s["answer"].lower()], dtype=torch.long)
        return img, q, y

# -------------------------------- VQA-path -------------------------------- #
@register_dataset("pathvqa")
class PathVQA(Dataset):
    TASK   = "vqa"
    PROMPT = []

    def __init__(self, root, transform, split="test",
                 build_vocab=True, top_k=None, sample_limit=None):
        import csv, sys, random
        from collections import Counter
        from pathlib import Path
        from PIL import Image
        root = Path(root)
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))   # 放宽字段长度

        split   = split.lower()
        # ---------- CSV 路径：先 images/ 下找，再退回根目录 ----------
        csv_path = Path(root) / "images" / f"{split}.csv"
        if not csv_path.is_file():
            csv_path = Path(root) / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)


        # ---------- 读 CSV ----------
        with open(csv_path, newline='', encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            header = reader.fieldnames
            rows   = list(reader)

        if sample_limit:
            random.seed(42); random.shuffle(rows)
            rows = rows[:sample_limit]

        # ---------- 自动探测列名 ----------
        # 可能的列：image / img_id, question / ques, answer / answers / label
        def _find(col_opts):
            for c in col_opts:
                if c in header:
                    return c
            raise KeyError(f"在 {csv_path.name} 里找不到列 {col_opts}")
        COL_IMG = _find(["image", "img_id", "img"])
        COL_Q   = _find(["question", "ques", "q"])
        COL_A   = _find(["answer", "answers", "label", "gt_answer"])

        # ---------- 构建答案表 ----------
        if build_vocab:
            cnt = Counter(r[COL_A].lower() for r in rows)
            ans_list = [a for a,_ in (cnt.most_common(top_k) if top_k else cnt.items())]
        else:
            ans_list = ["no", "yes"]

        self.ans2id = {a:i for i,a in enumerate(ans_list)}
        self.LABELS = ans_list
        self.__class__.LABELS = ans_list     # 兼容 get_dataset()

        # ---------- 定位图片 ----------
        img_dir   = Path(root) / "images" / split
        exts      = [".png", ".jpg", ".jpeg"]
        keep, miss, oov = [], 0, 0
        for r in rows:
            stem = str(r[COL_IMG]).strip()
            img_path = next((img_dir / f"{stem}{e}" for e in exts
                             if (img_dir / f"{stem}{e}").is_file()), None)
            if img_path is None:
                miss += 1; continue
            ans = r[COL_A].lower()
            if ans not in self.ans2id:
                oov += 1; continue
            keep.append(dict(img_path=img_path,
                             question=r[COL_Q],
                             answer=ans))

        self.samples = keep
        print(f"[PathVQA] {split}: kept {len(keep)} | missing_img {miss} | oov_ans {oov}")

        self.tfm = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self.tfm(Image.open(s["img_path"]).convert("RGB"))
        y   = torch.tensor(self.ans2id[s["answer"]], dtype=torch.long)
        return img, s["question"], y

# ========  ROCOv2 15-类多标签 ======== #
@register_dataset("rocov2")
class RocoV2(Dataset):
    LABELS = [
        "abdomen","liver","kidney","lung","heart",
        "brain","bone","knee","spine","tumor",
        "cyst","lesion","fracture","infection","hemorrhage"
    ]
    PROMPT = "A radiology image showing {}."
    TASK   = "multilabel"                    # ← 让 run.py 走多标签分支

    def __init__(self, root, transform, split="train", sample_limit=None):
        import json, pandas as pd, os
        csv_path = os.path.join(root, f"{split}.csv")   # 前一步生成的 csv
        df = pd.read_csv(csv_path)
        if sample_limit:
            df = df.sample(sample_limit, random_state=42).reset_index(drop=True)

        self.paths  = df["path"].tolist()               # 相对路径
        self.labels = np.stack(df["labels"].apply(json.loads).values)
        self.root   = root
        self.tfm    = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.paths[idx])).convert("RGB")
        img = self.tfm(img)
        y   = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, y

# =========  ROCO-v2 Caption Retrieval  ========= #
@register_dataset("rocov2_cap")
class RocoCaption(Dataset):
    """
    返回 (img, caption, gid)：
        img      : Tensor 3×224×224
        caption  : str
        gid      : int   —— 原图索引，可用于 I→T / T→I 评测
    """
    TASK = "retrieval"          # 🠚 run.py 里走新分支
    PROMPT = []                 # 不用 prompt
    LABELS = []                 # 不用

    def __init__(self, root, transform, split="test", sample_limit=None):
        import os, pandas as pd

        # ---------- 目录 fallback ----------
        if not os.path.isdir(root):
            root = root.replace("_cap", "")          # datasets/rocov2

        # ---------- 读 CSV ----------
        csv_path = os.path.join(root, f"{split}_captions.csv")
        df = pd.read_csv(csv_path)

        # ① 统一列名为小写，去掉首尾空格
        df.columns = [c.lower().strip() for c in df.columns]

        # ② optional sample
        if sample_limit:
            df = df.sample(sample_limit, random_state=42).reset_index(drop=True)

        # ---------- 自动找列 ----------
        def pick(cand):
            for c in cand:
                if c in df.columns:
                    return c
            raise KeyError(f"{csv_path} 列名缺少 {cand}，实际列: {list(df.columns)}")

        col_img = pick(["image_name", "image_id", "image", "id"])     # ← 加 "id"
        col_cap = pick(["caption", "description", "caption_0"])       # ← 小写后能匹配

        # ---------- 路径 ----------
        fnames = df[col_img].astype(str)
        # 若 ID 列里没扩展名，补 ".jpg"
        fnames = fnames.apply(lambda s: s if os.path.splitext(s)[1] else f"{s}.jpg")

        self.paths = [
            os.path.join(root, f"{split}_images/{split}/{n}") for n in fnames
        ]

        # ---------- gid / caption ----------
        name2gid = {n: i for i, n in enumerate(fnames.unique())}
        self.gids = fnames.map(name2gid).values.astype("int64")
        # self.caps = df[col_cap].tolist()
        self.tfm  = transform
        self.df = df
        cap_cols = [c for c in
                    ["caption", "caption_0", "caption_1",
                     "caption_2", "description"]  # 想要哪几列就加进去
                    if c in df.columns]
        df["full_cap"] = (
            df[cap_cols].fillna("")  # NaN → ""
            .agg(" ".join, axis=1)  # 拼成一句
            .str.replace(r"\s+", " ", regex=True)  # 多空格压一次
            .str.strip()
        )
        PROMPT = "A medical image: "  # 或 "This figure shows "
        self.caps = (PROMPT + df["full_cap"]).tolist()

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        img = self.tfm(Image.open(self.paths[idx]).convert("RGB"))
        return img, self.caps[idx], int(self.gids[idx])


@register_dataset("medfmc_cap")
class MedFMCCaption(Dataset):
    TASK = "retrieval"
    PROMPT = []
    LABELS = []

    def __init__(self, root, transform, split="test", sample_limit=None):
        csv_path = os.path.join(root, f"{split}_captions.csv")
        df = pd.read_csv(csv_path)
        self.paths = [os.path.join(root, f"{split}_images/{split}/images", fname) for fname in df["image"]]
        self.caps = df["caption"].tolist()
        self.gids = list(range(len(self.paths)))
        self.tfm = transform

    def __len__(self): return len(self.caps)

    def __getitem__(self, idx):
        # img = self.tfm(Image.open(self.paths[idx]).convert("RGB"))
        path = self.paths[idx]
        update_bar()
        # print(f"[DEBUG] Loading image: {path}")
        # print(f"[DEBUG] Loading image {self.paths[idx]}")
        try:
            img = self.tfm(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"[ERROR] Failed to load image {path}: {e}")
            raise e
        return img, self.caps[idx], self.gids[idx]

@register_dataset("isic2018")
class ISIC2018Dataset(Dataset):
    TASK = "segmentation"
    LABELS = ["lesion"]  # 至少有 1 个条目
    PROMPT = "A dermoscopy image of {}"  # 若还没有就补上
    # PROMPT = []
    # LABELS = []

    def __init__(self, root, transform=None, split="train"):
        self.root = Path(root) / split
        valid_exts = {".jpg", ".jpeg", ".png"}
        self.img_paths = sorted(
            p for p in (self.root / "images").glob("*")
            if p.suffix.lower() in valid_exts
        )
        self.mask_paths = sorted(
            p for p in (self.root / "masks").glob("*")
            if p.suffix.lower() in valid_exts
        )
        self.tfm = transform

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # 单通道灰度

        if self.tfm:
            img = self.tfm(img)
        else:
            img = T.ToTensor()(img)

        # ➤ 添加 resize，和你训练网络输入保持一致，例：256x256
        mask = mask.resize((256, 256), resample=Image.NEAREST)
        mask = TF.to_tensor(mask).squeeze(0)  # H x W
        mask = (mask > 0.5).float()  # 二值化

        return img, mask

def _wrap(ds_cls, **kw):
    ds = ds_cls(**kw)
    meta = {
        "labels": ds_cls.LABELS,
        "prompt": ds_cls.PROMPT,
        "task": ds_cls.TASK,
        "prompt_labels": getattr(ds_cls, "PROMPT_LABELS", ds_cls.LABELS),  # ← 新增这一行

        # ——————— 新增这一行 ———————
        "task_id": {
            "pcam": 0,
            "chestxray14": 1,
            "rsna_pneu": 2,
            "monuseg": 3,
            "isic2019": 4,
            "vqa_rad": 5,
            "pathvqa": 6,
            "isic2018": 8,
        }.get(name.lower(), None)
    }
    return ds, meta

def load_image(img_path, dataset_name):
    """
    根据数据集类型加载图片
    """
    if dataset_name.lower() == 'monuseg':
        try:
            img_np = imread(img_path)  # 使用 tifffile 读取 LZW .tif
            if img_np.ndim == 2:  # 灰度图 → 转 RGB
                img_np = np.stack([img_np] * 3, axis=-1)
            img = Image.fromarray(img_np).convert("RGB")
        except Exception as e:
            print(f"[load_image] Failed to load {img_path}: {e}")
            raise
        return img
    else:
        # 默认 PIL 加载（用于 jpg/png）
        return Image.open(img_path).convert("RGB")

def get_dataset(name, **kw):
    ds_cls = _DATASETS[name.lower()]
    # name_lower = name.lower()
    if "split" in kw:
        sig = inspect.signature(ds_cls.__init__)
    if "split" not in sig.parameters:
        kw.pop("split")

    ds      = ds_cls(**kw)
    meta    = {"labels": ds_cls.LABELS,
               "prompt": ds_cls.PROMPT,
               "task"  : ds_cls.TASK,
               # "task_id": list(_DATASETS.keys()).index(name_lower),
               }
    if hasattr(ds, "labels"):
        lbl = np.asarray(ds.labels)
        if np.issubdtype(lbl.dtype, np.floating):
            # 对二分类 float 标签 (0. / 1.) → 转 int
            if ((lbl == 0.) | (lbl == 1.)).all():
                lbl = lbl.astype(np.int64)
            else:
                lbl = None  # 多标签 / 连续值就别 bin 了
        elif lbl.dtype.kind not in {"i", "u"}:  # 其它非整数类型
            lbl = None
        if lbl is not None:
            if lbl.ndim == 1:  # 单维 OK
                cnt = np.bincount(lbl)
            else:  # 多标签 → 按列求和
                cnt = lbl.sum(axis=0)
            print(f"[DEBUG] {name.upper()} size={len(ds)},  label count:", cnt)

    return ds, meta