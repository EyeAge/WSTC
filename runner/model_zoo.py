# runner/model_zoo.py
import sys, os, pathlib, torch
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer, AutoModel
from runner.dataset_zoo import _DATASETS   # 直接复用注册表
from types import SimpleNamespace

# -------- 把 baselines/UniMed-CLIP/src 加到 import 路径 --------
# UNI_SRC = (pathlib.Path(__file__).resolve()
#            .parents[1] / "baselines" / "UniMed-CLIP" / "src")
# sys.path.insert(0, str(UNI_SRC))
# sys.path.append(str(UNI_SRC))
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "baselines", "UniMed-CLIP", "src"))
# sys.path.insert(0, root_dir)
# from open_clip_old.factory import create_model_and_transforms


def _load_unimedclip(device: torch.device, precision: str = "fp32"):
    # from open_clip.factory import create_model_and_transforms   # ← 已被 sys.path 覆盖

    # ---------- 这里插入 ----------
    if isinstance(device, str):
        device = torch.device(device)
    ckpt_path = pathlib.Path("baselines/UniMed-CLIP/checkpoints/b16_400m.pt")
    import open_clip
    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        # model_name='ViT-B-16-quickgelu',
        # pretrained="baselines/UniMed-CLIP/checkpoints/b16_400m.pt",
        pretrained=str(ckpt_path),
        # pretrained="openai",
        # text_encoder_name="microsoft/BiomedVLP-CXR-BERT-general",
        # text_encoder_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        # device="cpu",                # ① 先全部加载到 CPU
        device=device,
        # precision=precision,
        # output_dict=True,
        precision    = precision,
        # force_quick_gelu=True,
        # weights_only=False
    )
    ckpt = torch.load("baselines/UniMed-CLIP/checkpoints/b16_400m.pt", map_location="cpu")
    vis_sd = {k[len("visual."):]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("visual.")}
    missing, unexpected = model.visual.load_state_dict(vis_sd, strict=False)
    assert not missing, f"缺失视觉权重: {missing}"
    # unexpected 里通常是 text_encoder 的参数——忽略即可

    # 3) 保留原始医学 BERT 文本编码器
    bert_id = "microsoft/BiomedVLP-CXR-BERT-general"
    # bert_id ="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    tokenizer = AutoTokenizer.from_pretrained(bert_id)
    text_encoder = AutoModel.from_pretrained(bert_id)
    model.tokenizer = tokenizer  # 替换 tokenizer
    model.text_encoder = text_encoder.to(device)  # 替换 encoder


    # ② 视觉分支挪到 GPU，文本分支留在 CPU
    # if device.type == "cuda":
    #     # model.visual = model.visual.to(device).half()
    #     model = model.to(device)  # ① 全模型搬显卡
    #     if precision == "fp16":
    #         model.half()  # ② 整体转 fp16（就地操作）
    #         # ③ 让 text_encoder 回 CPU 省显存
    #         model.text_encoder = model.text_encoder.to("cpu")
    #     # text_encoder 依旧在 CPU，计算时用 .to("cpu")
    model = model.to(device)  # GPU / CPU 都 OK
    if precision == "fp16":
        model.half()  # 直接整模型 .half() 就行
    if device.type == "cuda" and precision == "fp16":  # 省显存：文本分支搬回 CPU
        model.text_encoder = model.text_encoder.to("cpu")
    from types import MethodType

    def _install_patch_hook(mdl):

        grabbed = {"tokens": None}

        def _try_hook(mod, inp, out):
            # out 可能是 (tokens, None)
            if isinstance(out, tuple):
                out = out[0]  # [B, 197, 768]
            # print("[HOOK] out.shape =", out.shape)  # ①
            # ---- 1. 拿到纯 patch ----
            out = out.transpose(0, 1)
            cls = out[:, 0, :]  # [B, 768]
            patch = out[:, 1:, :]
            # patch = out[:, 1:, :]  # [B, 196, 768]
            # print("[HOOK] patch.shape =", patch.shape)  # ②

            # ---- 2. 与 CLS 同样的 LN / proj ----
            ln_post, proj = mdl.visual.ln_post, mdl.visual.proj
            cls = ln_post(cls)

            patch = ln_post(patch)
            if proj is not None:  # ViT-B/16 有 proj
                cls = cls @ proj
                patch = patch @ proj  # 768 → 512
            # 现在 patch 是 [B, 196, 512]

            # ---- 3. *不要* 再 view/reshape！直接保存 ----
            grabbed["cls"] = cls.detach()
            grabbed["tokens"] = patch.detach()

        mdl.visual.transformer.register_forward_hook(_try_hook)
        # for blk in getattr(mdl.visual.transformer, "blocks",
        #                    getattr(mdl.visual.transformer, "resblocks", [])):
        #     blk.register_forward_hook(_try_hook)

        # ---- monkey-patch encode_image -- --
        from types import MethodType
        def _encode_image_with_patch(self, img):
            _ = self.visual(img)  # 触发 hook
            # cls = self.visual(img)  # [B, 512]
            # patch = grabbed["tokens"]  # [B, 196, 512]
            return {"pooled_output": grabbed["cls"],
                    "patch_tokens": grabbed["tokens"]}

        mdl.encode_image = MethodType(_encode_image_with_patch, mdl)

    _install_patch_hook(model)

    return model, preprocess

# ------------ 统一入口 ------------

"""
name     : 'unimedclip' / 'bioclip' ...
variant  : baseline / +wsam / +wstc
"""
# ----- UniMed-CLIP -----
def load_model(name: str,
               device: torch.device,
               precision: str = "fp32",
               variant: str = "baseline",
               num_classes: int = None,
               dataset_name: str = None,
               task_id = None,
               proj_dim=256,
               use_cross_modal=False,
               with_cls_head: bool = False
               ):
    """
    name     : 'unimedclip' / 'bioclip' ...
    variant  :
        baseline      → 纯 UniMed-CLIP
        +wsam         → 只挂 WSAM
        +tcdam        → 只挂 TCDAM
        +wstc         → WSAM + TCDAM
        +wstc_train   → WSAM + TCDAM + Linear head（需训练）
    """
    need_prompt = name.lower() in {"tip_adapter", "tip_adapter_f", "meta_adapter"}
    if need_prompt and dataset_name is None:
        raise ValueError("Tip-Adapter 需要 --dataset 参数，用来生成 prompt 与文本特征")

        # ===== UniMed-CLIP =====
    if name.lower() == "unimedclip":
        model, preprocess = _load_unimedclip(device, precision)

        if variant != "baseline":
            # ① 解析 variant -> 三个开关
            if variant == "+wsam":
                use_wsam, use_tcdam, with_cls_head = True, False, False
            elif variant == "+tcdam":
                use_wsam, use_tcdam, with_cls_head = False, True, False
            elif variant == "+wstc":
                use_wsam, use_tcdam, with_cls_head = True, True, False
            elif variant == "+wstc_train":
                use_wsam, use_tcdam, with_cls_head = True, True, True
            else:
                raise ValueError(f"Unknown variant: {variant}")

            # ② 动态 import 包装器再实例化
            MOD_DIR = pathlib.Path(__file__).resolve().parents[1] / \
                      "baselines" / "UniMed-CLIP" / "modules"
            sys.path.insert(0, str(MOD_DIR))
            from wstc import UniMedCLIP_WSTC

            if num_classes is None and with_cls_head:
                if dataset_name is None:
                    raise ValueError("with_cls_head=True 但未指定 dataset_name 无法自动设置 num_classes")
                dataset_lower = dataset_name.lower()
                if dataset_lower == "pcam":
                    num_classes = 1
                elif dataset_lower == "chestxray14":
                    num_classes = 15
                elif dataset_lower == "rsna_pneu":
                    num_classes = 1
                else:
                    raise ValueError("未知数据集，无法自动设置 num_classes")

            model = UniMedCLIP_WSTC(
                model,
                use_wsam=use_wsam,
                use_tcdam=use_tcdam,
                with_cls_head=with_cls_head,
                num_classes=num_classes,
                # proj_dim=vars(device)["proj_dim"] if hasattr(device, "proj_dim") else 256,\
                proj_dim=proj_dim,
                task_id=task_id,
                use_cross_modal=use_cross_modal,
                # new_interface = True
            )  # ChestXray14
            print('dataset_namedataset_namedataset_name',dataset_name.lower())
            if dataset_name and dataset_name.startswith("rocov2") and variant == "+wstc":
                # model._skip_proj = True
                print("[DBG] dataset_name =", dataset_name, " variant =", variant)
                print("[DBG] passes test? ",
                      dataset_name and dataset_name.startswith("rocov2"),
                      variant == "+wstc")
                print("[INFO]  ROCOv2 +WSTC : reset text projector → 512→256")
                root_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "baselines", "UniMed-CLIP", "modules"))
                sys.path.insert(0, root_dir)
                from text_projector import SmallMLPProjector
                model.text_proj = SmallMLPProjector(
                    input_dim=512,  # ← CLIP-Text 输出维度
                    hidden_dim=512,
                    output_dim=256
                ).to(next(model.parameters()).device)
                # import torch.nn as nn
                # model.image_proj = nn.Linear(512, 256, bias=False).to(next(model.parameters()).device)
            if dataset_name and dataset_name.startswith("medfmc_cap") and variant == "+wstc":
                print("[INFO]  MedFMC +WSTC : reset text projector → 512→256")
                from text_projector import SmallMLPProjector
                model.text_proj = SmallMLPProjector(
                    input_dim=512, hidden_dim=512, output_dim=256
                ).to(next(model.parameters()).device)
        model.new_interface = True

        return model, preprocess

    # ---------- OpenAI CLIP baseline ----------
    if name.lower() == "clip":
        import math, torch, torch.nn.functional as F
        import open_clip

        # 1) 原生 CLIP
        _clip, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai", device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        ctx_len = _clip.context_length  # = 77
        pad_id = 0  # open_clip pad token

        # 2) ---------- 核心接口 ----------
        @torch.no_grad()
        def encode_image(img):  # -> [B,512]
            return F.normalize(_clip.encode_image(img), dim=-1)

        @torch.no_grad()
        def encode_text(inp):
            """
            支持两种输入:
              • list[str]           —— 检索路径
              • Tensor[int64] (B,L) —— 分类/分割路径 (open_clip.tokenize 结果)
            """
            if isinstance(inp, torch.Tensor):  # Tensor path
                ids = inp.to(device)
                # 如果长度 < 77, 右侧 padding
                if ids.shape[1] != ctx_len:
                    pad_len = ctx_len - ids.shape[1]
                    ids = F.pad(ids, (0, pad_len), value=pad_id)
                feat = _clip.encode_text(ids)  # [B,512]
                return F.normalize(feat, dim=-1)

            # list[str] path
            if isinstance(inp, str):
                inp = [inp]
            tok = tokenizer(inp).to(device)  # (B,77)
            feat = _clip.encode_text(tok)
            return F.normalize(feat, dim=-1)

        # 3) ---------- text_encoder 供 run.py 调 CLS ----------
        def text_encoder(input_ids, attention_mask=None):
            feat = encode_text(input_ids)  # [B,512]
            return SimpleNamespace(last_hidden_state=feat.unsqueeze(1))

        # 4) ---------- 汇总 & 补 run.py 依赖 ----------
        clip_ns = SimpleNamespace()
        clip_ns.encode_image = encode_image
        clip_ns.encode_text = encode_text  # 检索直接用
        clip_ns.text_encoder = text_encoder  # 分类/分割用
        clip_ns.clip = SimpleNamespace(  # logit_scale
            logit_scale=_clip.logit_scale)
        clip_ns.parameters = _clip.parameters  # dtype / device
        clip_ns.to = lambda *a, **k: clip_ns
        clip_ns.eval = lambda *a, **k: clip_ns
        clip_ns.requires_grad_ = lambda *a, **k: clip_ns

        return clip_ns, preprocess

        # ---------- 2. MedCLIP ----------
    if name.lower() == "medclip":
        baselines_root = pathlib.Path(__file__).resolve().parents[1] / "baselines"
        sys.path.insert(0, str(baselines_root))  # 把 baselines 根目录塞进去
        from transformers import AutoTokenizer  # ← 新增
        from MedCLIP.medclip.modeling_medclip import MedCLIPModel
        import open_clip, torch, torch.nn.functional as F
        model = MedCLIPModel()
        model.from_pretrained("RUCAIBox/MedCLIP-ViT-B-16-256px")
        model.eval().to(device)

        # 2) 统一图像预处理：直接复用 open-clip 的 Transform
        _, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )

        # 3) 文本 tokenizer（MedCLIP 论文使用 Bio_ClinicalBERT）
        tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )

        # 4) 封装统一接口
        @torch.no_grad()
        def encode_image(img):  # -> [B,512]
            return F.normalize(model.vision_model(img.float()), dim=-1)

        @torch.no_grad()
        def encode_text(texts, attention_mask=None):
            # ---------- open_clip.tokenize path ----------
            if isinstance(texts, torch.Tensor):
                input_ids = texts.to(device)
                if attention_mask is None:
                    attention_mask = (input_ids != tokenizer.pad_token_id).long()
                outputs = model.text_model(input_ids=input_ids,
                                           attention_mask=attention_mask.to(device))
                # ★ 关键兼容：既可能是 Tensor，也可能是带 pooler_output 的对象
                if isinstance(outputs, torch.Tensor):
                    feat = outputs  # [B,512]
                elif hasattr(outputs, "pooler_output"):
                    feat = outputs.pooler_output  # [B,512]
                else:  # 兜底：取 CLS
                    feat = outputs.last_hidden_state[:, 0]
                return F.normalize(feat, dim=-1)

            # ---------- raw str / List[str] path ----------
            if isinstance(texts, str):
                texts = [texts]
            tok = tokenizer(texts, padding=True, truncation=True,
                            max_length=64, return_tensors="pt").to(device)
            outputs = model.text_model(**tok)
            feat = outputs.pooler_output if hasattr(outputs, "pooler_output") \
                else outputs.last_hidden_state[:, 0]
            return F.normalize(feat, dim=-1)

        import math
        medclip_ns = SimpleNamespace(
            encode_image=encode_image,
            encode_text=encode_text
        )

        # -------- 2) 额外提供 text_encoder 给分类/分割 ----------
        def text_encoder(input_ids, attention_mask=None):
            outputs = model.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask)
            if isinstance(outputs, torch.Tensor):  # 旧权重直接给 CLS
                return SimpleNamespace(last_hidden_state=outputs.unsqueeze(1))
            return outputs

        medclip_ns.text_encoder = text_encoder

        #  logit_scale ≈ ln(1/τ)；OpenAI-CLIP 默认 τ≈0.07
        logit_scale = torch.nn.Parameter(
            torch.ones([], device=device) * math.log(1 / 0.07),
            requires_grad=False  # 推理用，不需要训练
        )
        medclip_ns.clip = SimpleNamespace(logit_scale=logit_scale)

        medclip_ns.parameters = model.vision_model.parameters
        medclip_ns.to = lambda *a, **k: medclip_ns
        medclip_ns.eval = lambda *a, **k: medclip_ns  # returns self
        medclip_ns.requires_grad_ = lambda *a, **k: medclip_ns
        return medclip_ns, preprocess
        # return SimpleNamespace(encode_image=encode_image,
        #                        encode_text=encode_text), preprocess

    # ---------- BioViL (ECCV22) ----------
    if name.lower() == "biovil":
        # ---------- model_zoo.py · "biovil" 分支 --------------------
        from health_multimodal.image import get_image_inference
        from health_multimodal.image.utils import ImageModelType
        from health_multimodal.text.utils import get_bert_inference, BertEncoderType
        import torch, torch.nn.functional as F
        from torchvision import transforms as T  # ★ 新增

        # 假设 load_model(device=...) 已经把 device 传进来
        # -----------------------------------------------------------------
        img_engine = get_image_inference(ImageModelType.BIOVIL_T)
        txt_engine = get_bert_inference(BertEncoderType.CXR_BERT)

        # **关键：把底层网络搬到同一块 device**
        img_enc = img_engine.model.to(device)  # ViT-B/16
        txt_enc = txt_engine.model.to(device)  # CXR-BERT
        tokenizer = txt_engine.tokenizer
        base_tfm = img_engine.transform  # <class 'torchvision.transforms.Compose'>
        resize_tfm, *rest_tfms = base_tfm.transforms

        preprocess = T.Compose([
            resize_tfm,  # Resize(448)
            T.Grayscale(num_output_channels=1),  # ☆ 新增：RGB → 1-channel
            *rest_tfms  # ToTensor(), Normalize(...)
        ])
        # preprocess = img_engine.transform  # torchvision.transforms
        dtype = torch.float16 if (device.type == "cuda" and precision == "fp16") else torch.float32
        img_enc = img_enc.to(dtype)
        txt_enc = txt_enc.to(dtype)

        # ---------- 统一接口 ----------------------------------------------
        @torch.no_grad()
        def encode_image(img):  # img:[B,C,H,W]
            img = img.to(device, dtype=dtype)

            # 如果是 RGB，在线转灰度：Y' = 0.2989 R + 0.5870 G + 0.1140 B
            # if img.shape[1] == 3:
            #     r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            #     img = 0.2989 * r + 0.5870 * g + 0.1140 * b  # → [B,1,H,W]

            out = img_enc(img)  # ImageModelOutput
            feat = F.normalize(out.projected_global_embedding, dim=-1)  # [B,512]
            return feat

        @torch.no_grad()
        def encode_text(x, attention_mask=None):
            # a) open_clip.tokenize Tensor path
            if isinstance(x, torch.Tensor):
                x = x.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                feat = txt_enc(input_ids=x, attention_mask=attention_mask
                               ).last_hidden_state[:, 0]  # CLS
                return F.normalize(feat, dim=-1)

            # b) list[str] / str path
            if isinstance(x, str): x = [x]
            tok = tokenizer(x, padding=True, truncation=True,
                            max_length=64, return_tensors="pt").to(device)
            feat = txt_enc(**tok).last_hidden_state[:, 0]
            return F.normalize(feat, dim=-1)

        def text_encoder(input_ids, attention_mask=None):
            out = txt_enc(input_ids.to(device),
                          attention_mask=attention_mask.to(device)
                          if attention_mask is not None else None)
            return SimpleNamespace(last_hidden_state=out.last_hidden_state)

        # ---------- 打包给 run.py 使用 --------------------------------------
        biovil = SimpleNamespace(
            encode_image=encode_image,
            encode_text=encode_text,
            text_encoder=text_encoder,
            clip=SimpleNamespace(
                logit_scale=torch.nn.Parameter(
                    torch.log(torch.tensor(1 / 0.07)), requires_grad=False)
            ),
            parameters=img_enc.parameters,
            to=lambda *a, **k: biovil,
            eval=lambda *a, **k: biovil,
            requires_grad_=lambda *a, **k: biovil,
        )
        return biovil, preprocess




    elif name.lower() in {"tip_adapter", "tip_adapter_f"}:
        clip, preprocess = _load_unimedclip(device, precision)
        # === prompt embedding ===
        import open_clip  # NEW
        ds_cls = _DATASETS[dataset_name.lower()]
        tmpl = ds_cls.PROMPT
        if isinstance(tmpl, str):  # "A CX-ray of {}."
            prompts = [tmpl.format(c) for c in ds_cls.LABELS]
        elif isinstance(tmpl, (list, tuple)):  # 已给定完整 prompt 列表
            prompts = list(tmpl)
        else:
            raise ValueError("PROMPT 必须是 str 或 list[str]")

        token_ids = open_clip.tokenize(prompts).to(device)  # [C, 77]

        import open_clip, torch, torch.nn.functional as F  # ← 多引入 torch
        orig_dtype = next(clip.parameters()).dtype
        with torch.no_grad():
            if orig_dtype == torch.float16:
                clip.float()  # 全模型先转 fp32
            txt_feats = clip.encode_text(token_ids)  # [C, 512] float32
            txt_feats = F.normalize(txt_feats, dim=-1)  # L2-norm
            if orig_dtype == torch.float16:  # 还原回 fp16
                clip.half()

        MOD_DIR = pathlib.Path(__file__).resolve().parents[1] / \
                  "baselines" / "Tip-Adapter"
        sys.path.insert(0, str(MOD_DIR))
        from wrapper import TipAdapter
        model = TipAdapter(clip, txt_feats, alpha=10.0, beta=1.0, temp=0.05)

        if name.lower() == "tip_adapter_f":  # TA-F
            model.enable_finetune(lr=5e-2)

        model.new_interface = True
        return model, preprocess


    elif name.lower() in {"meta_adapter", "meta_adapter_f"}:
        # 1) 先拿 UniMed-CLIP backbone
        clip, preprocess = _load_unimedclip(device, precision="fp32")
        # 2) 取医学 prompt → text feature
        import open_clip, torch ,torch.nn.functional as F
        tmpl = _DATASETS[dataset_name.lower()].PROMPT
        lbls = _DATASETS[dataset_name.lower()].LABELS
        prompts = [tmpl.format(c) for c in lbls] if isinstance(tmpl, str) else list(tmpl)
        txt_ids = open_clip.tokenize(prompts).to(device)
        with torch.no_grad():
            txt = clip.encode_text(txt_ids).float()
            txt = F.normalize(txt, dim=-1)  # [C,512]
        # 3) Meta-Adapter 基类
        MOD_DIR = pathlib.Path(__file__).resolve().parents[1] / \
                  "baselines" / "Meta-Adapter"
        sys.path.insert(0, str(MOD_DIR))
        from wrapper import MetaAdapter
        # model = MetaAdapter(clip, txt, temp=0.01)
        # C = len(lbls)  # 类别数
        want_head = name.lower().endswith("_f")  # True ↔ meta_adapter_f
        num_classes = 1 if len(lbls) == 2 else len(lbls)  # 2 → 单列；其余保持 C 列
        # ★ 统一只在这里建 head；num_classes=0 表示“别建头”
        model = MetaAdapter(
            clip_model=clip,
            text_feats=txt,
            num_classes=num_classes if want_head else 0,  # 0 = 不建线性头
            temp=0.01
        )
        model.cls = model.head
        model.with_cls = True
        for p in clip.parameters():
            p.requires_grad_(False)


        model.new_interface = True
        return model, preprocess

    # ===== 其它 baseline（占位）=====
    # elif name.lower() == "bioclip":
    #     import baselines.bioclip as bc
    #     return bc.load(device)

    raise ValueError(f"Unknown baseline: {name}")
