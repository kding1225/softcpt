import os
import os.path as osp
import itertools
from collections import OrderedDict

import tqdm
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy_map
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.evaluation import build_evaluator

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import Transformer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class TaskPromptLearner(nn.Module):
    def __init__(self, cfg, tasknames, clip_model):
        super().__init__()
        n_task = len(tasknames)
        n_ctx = cfg.TRAINER.MTCOOP.TASK_N_CTX
        ctx_init = cfg.TRAINER.MTCOOP.TASK_CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.MTCOOP.TASK_CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_task, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        if n_ctx == 0:
            self.register_buffer("ctx", ctx_vectors)  # not optimized
        else:
            self.register_parameter("ctx", nn.Parameter(ctx_vectors))  # to be optimized

        tasknames = [name.replace("_", " ") for name in tasknames]
        name_lens = [len(_tokenizer.encode(name)) for name in tasknames]
        prompts = [prompt_prefix + " " + name + "." for name in tasknames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_task = n_task
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.MTCOOP.TASK_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_task, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_task, 1, dim)
                    ctx,  # (n_task, n_ctx, dim)
                    suffix,  # (n_task, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_task):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_task):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClassPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MTCOOP.CLASS_N_CTX
        ctx_init = cfg.TRAINER.MTCOOP.CLASS_CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.MTCOOP.CLASS_CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        if n_ctx == 0:
            self.register_buffer("ctx", ctx_vectors)  # not optimized
        else:
            self.register_parameter("ctx", nn.Parameter(ctx_vectors))  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.MTCOOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MTCOOP.N_CTX
        ctx_init = cfg.TRAINER.MTCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_prefix = ctx_init
        else:
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.token_position = cfg.TRAINER.MTCOOP.TOKEN_POSITION
        
    def forward(self, ctx, idx=None):
        """
        ctx: context generated by prompt_gen
        """
        name_lens = self.name_lens
        n_cls = self.n_cls

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if idx is not None:
            prefix = prefix[idx]
            suffix = suffix[idx]
            name_lens = [name_lens[i.item()] for i in idx]
            n_cls = len(idx)

        if self.token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(n_cls):
                name_len = name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.token_position == "front":
            prompts = []
            for i in range(n_cls):
                name_len = name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class PromptGen(nn.Module):
    """
    prompt generator, two use cases:
    1. input task feats only, output prompt for each task
    2. input task+cls feats, output prompt for each cls
    """

    def __init__(self, cfg, n_ctx, num_tasks, feat_dim, is_csc, out_dim, dtype):
        super().__init__()

        self.dtype = dtype
        self.n_ctx = n_ctx
        self.out_dim = out_dim
        self.prompt_gen_type = cfg.TRAINER.MTCOOP.PG_TYPE

        if self.prompt_gen_type == "lin":
            # task_feat or cat(task_feat, cls_feat) then linear
            width = feat_dim * 2 if is_csc else feat_dim
            self.register_parameter("proj", nn.Parameter(torch.zeros(width, n_ctx * out_dim)))
            nn.init.normal_(self.proj, std=0.02)
        elif self.prompt_gen_type == "linmean":
            # task_feat or mean(task_feat, cls_feat) then linear
            self.register_parameter("proj", nn.Parameter(torch.zeros(feat_dim, n_ctx * out_dim)))
            nn.init.normal_(self.proj, std=0.02)
        elif self.prompt_gen_type == "lincls":
            # only use cls feats, then apply linear
            assert is_csc
            self.register_parameter("proj", nn.Parameter(torch.zeros(feat_dim, n_ctx * out_dim)))
            nn.init.normal_(self.proj, std=0.02)
        else:
            raise NotImplementedError

    def forward(self, x):
        ctx = None
        if self.prompt_gen_type == "lin":
            ctx = x @ self.proj.type(x.dtype)
            ctx = ctx.reshape(x.shape[0], self.n_ctx, self.out_dim)
        elif self.prompt_gen_type == "linmean":
            d = x.shape[-1]
            ctx = ((x[..., :d // 2] + x[..., d // 2:]) / 2.) @ self.proj.type(x.dtype)
            ctx = ctx.reshape(x.shape[0], self.n_ctx, self.out_dim)
        elif self.prompt_gen_type == "lincls":
            d = x.shape[-1]
            ctx = x[..., :d // 2] @ self.proj.type(x.dtype)
            ctx = ctx.reshape(x.shape[0], self.n_ctx, self.out_dim)

        return ctx


class CustomCLIP(nn.Module):
    def __init__(self, cfg, tasknames, classnames, clip_model):
        super().__init__()
        self.prompt_learner = nn.ModuleList(
            [PromptLearner(cfg, c, clip_model) for c in classnames]
        )
        self.task_prompt_learner = TaskPromptLearner(cfg, tasknames, clip_model)
        self.task_tokenized_prompts = self.task_prompt_learner.tokenized_prompts

        # when prompt_learner is cls-specific, ClassPromptLearner is needed
        self.is_csc = cfg.TRAINER.MTCOOP.CSC
        if self.is_csc:
            self.class_prompt_learner = ClassPromptLearner(cfg, list(itertools.chain.from_iterable(classnames)),
                                                           clip_model)
            self.class_tokenized_prompts = self.class_prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.n_ctx = cfg.TRAINER.MTCOOP.N_CTX
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.task_feat_dim = clip_model.text_projection.shape[1]

        self.prompt_gen = PromptGen(
            cfg,
            self.n_ctx,
            len(tasknames),
            self.task_feat_dim,
            self.is_csc,
            self.ctx_dim,
            self.dtype
        )

        self.num_classes = [len(x) for x in classnames]
        self.output_dir = cfg.OUTPUT_DIR

        self.cls_sample_rate = cfg.TRAINER.MTCOOP.CLS_SAMPLE_RATE
        if not hasattr(self, "class_prompt_learner"):
            assert self.cls_sample_rate == 0.0

    def compute_cls_feats(self, classnames, clip_model):

        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        cls_features = self.text_encoder(prompts, embedding)

        return cls_features

    def forward(self, image, label=None):

        prompts = self.task_prompt_learner()
        task_tokenized_prompts = self.task_tokenized_prompts
        task_features = self.text_encoder(prompts, task_tokenized_prompts)

        if self.is_csc:
            prompts = self.class_prompt_learner()
            class_tokenized_prompts = self.class_tokenized_prompts
            class_features = self.text_encoder(prompts, class_tokenized_prompts)

            cc = 0
            features = []
            for i, num in enumerate(self.num_classes):
                features.append(torch.cat([class_features[cc:cc + num], task_features[[i], :].repeat(num, 1)], dim=1))
                cc += num
            features = torch.cat(features, dim=0)

            ctx = self.prompt_gen(features.type(self.dtype))
            ctx = torch.split(ctx, self.num_classes, dim=0)
        else:
            ctx = self.prompt_gen(task_features.type(self.dtype))

        image_features = self.image_encoder(image.type(self.dtype))

        if self.cls_sample_rate > 0 and label is not None:
            text_features, selected = self.compute_text_features_subsample(
                label, ctx, self.cls_sample_rate
            )
        else:
            text_features = self.compute_text_features_full(ctx)
            selected = None

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, selected

    def compute_text_features_full(self, ctx):
        text_features = []
        for i, p in enumerate(self.prompt_learner):
            prompts = p.forward(ctx[i])
            f = self.text_encoder(prompts, p.tokenized_prompts)
            text_features.append(f)
        text_features = torch.cat(text_features, dim=0)
        return text_features

    def compute_text_features_subsample(self, label, ctx, cls_sample_rate):
        """
        subsample labels for each task to reduce GPU mem cost, but may loss acc,
        need better sampling method
        """

        def _choose(x, rate):
            k = math.ceil(len(x) * rate)
            perm = torch.randperm(len(x))[:k]
            return x[perm]

        cc = 0
        text_features = []
        selected = []
        for i, p in enumerate(self.prompt_learner):
            c = self.num_classes[i]
            cur_label = label[:, cc:cc + c]
            mask = (cur_label == 1).any(dim=0)
            idx_pos = torch.where(mask)[0]
            idx_neg = torch.where(~mask)[0]
            idx_neg = _choose(idx_neg, cls_sample_rate)
            idx = torch.cat([idx_pos, idx_neg])
            selected.append(idx + cc)

            prompts = p.forward(ctx[i][idx], idx=idx)
            f = self.text_encoder(prompts, p.tokenized_prompts[idx])
            f0 = 1e-6 * torch.ones(c, f.shape[1], device=f.device, dtype=f.dtype)
            f0[idx, :] = f0[idx, :] + f
            text_features.append(f0)
            cc += c
        text_features = torch.cat(text_features, dim=0)
        selected = torch.cat(selected)

        return text_features, selected


@TRAINER_REGISTRY.register()
class MTCoOp(TrainerX):
    """Multitask Context Optimization (MTCOOP).
    """

    def __init__(self, cfg):

        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(
            cfg,
            lab2cname=self.taskid2cname,
            task2tname=self.taskid2tname,
            task2type=self.taskid2type,
            task2metric=self.taskid2metric,
        )
        self.best_result = -np.inf

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        super().build_data_loader()

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MTCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg

        classnames = self.dm.dataset.taskid2cname
        self.taskid2tname = self.dm.dataset.taskid2tname
        self.taskid2cname = self.dm.dataset.taskid2cname
        self.taskid2type = self.dm.dataset.taskid2type
        self.taskid2metric = self.dm.dataset.taskid2metric
        self.num_tasks = len(self.taskid2tname)
        self.cls_range = self.compute_cls_range([len(x) for x in self.taskid2cname])

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MTCOOP.PREC == "fp32" or cfg.TRAINER.MTCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.taskid2tname, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

            if "prompt_learner" in name:
                param.requires_grad_(True)
            if "class_prompt_learner" in name:
                param.requires_grad_(True)
            if "task_prompt_learner" in name:
                param.requires_grad_(True)
            if "prompt_gen" in name:
                param.requires_grad_(True)
        
        freeze_cls_prompt = cfg.TRAINER.MTCOOP.FREEZE_CLS_PROMPT
        if freeze_cls_prompt:
            for name, param in self.model.named_parameters():
                if "class_prompt_learner" in name:
                    param.requires_grad_(False)
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        if hasattr(self.model, "prompt_learner"):
            self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        if hasattr(self.model, "class_prompt_learner") and (not freeze_cls_prompt):
            self.register_model("class_prompt_learner", self.model.class_prompt_learner, self.optim, self.sched)
        if hasattr(self.model, "task_prompt_learner"):
            self.register_model("task_prompt_learner", self.model.task_prompt_learner, self.optim, self.sched)
        if hasattr(self.model, "prompt_gen"):
            self.register_model("prompt_gen", self.model.prompt_gen, self.optim, self.sched)

        cnt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                cnt += 1
        print("number learnable params", cnt)

        cnt2 = 0
        for param_group in self.optim.param_groups:
            for g in param_group["params"]:
                if g.requires_grad:
                    cnt2 += 1
        print("number learnable params", cnt2)
        assert cnt == cnt2

        self.scaler = GradScaler() if cfg.TRAINER.MTCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        output2 = None
        prec = self.cfg.TRAINER.MTCOOP.PREC
        if prec == "amp":
            with autocast():
                output, selected = self.model(image, label)
                if selected is None:
                    loss, output2 = self.mutitask_loss(output, label)
                else:
                    loss, output2 = self.mutitask_loss2(output, label, selected)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, selected = self.model(image, label)
            if selected is None:
                loss, output2 = self.mutitask_loss(output, label)
            else:
                loss, output2 = self.mutitask_loss2(output, label, selected)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "map": compute_accuracy_map(output2.data, label.data),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def mutitask_loss(self, output, label):
        batch_size = output.shape[0]
        output2 = output.data.clone()

        loss = 0.
        for i in range(self.num_tasks):
            st, ed = self.cls_range[i]
            cur_output = output[:, st:ed]
            cur_label = label[:, st:ed]
            mask = torch.sum(cur_label != -1, dim=-1) > 0  # samples of the current task

            if mask.sum() == 0:
                continue

            typ = self.taskid2type[i]
            assert typ == "sl"
            loss += F.cross_entropy(cur_output[mask, :], cur_label[mask, :].argmax(dim=-1), reduction='sum')
            output2[:, st:ed] = output2[:, st:ed].softmax(dim=-1)
        loss = loss / batch_size

        return loss, output2

    def mutitask_loss2(self, output, label, selected):
        """
        loss computation with cls sampling
        """
        batch_size = output.shape[0]
        output2 = output.data.clone()
        if selected is not None:
            select_mask = torch.zeros_like(label[0, :]).bool()
            select_mask[selected] = 1

        loss = 0.
        for i in range(self.num_tasks):
            st, ed = self.cls_range[i]
            cur_output = output[:, st:ed]
            cur_label = label[:, st:ed]
            mask_row = torch.sum(cur_label != -1, dim=-1) > 0

            if mask_row.sum() == 0:
                continue

            typ = self.taskid2type[i]
            assert typ == "sl"
            if selected is not None:
                mask_col = select_mask[st:ed]
                loss += F.cross_entropy(
                    cur_output[mask_row][:, mask_col],
                    cur_label[mask_row][:, mask_col].argmax(dim=-1),
                    reduction='sum'
                )
            else:
                loss += F.cross_entropy(
                    cur_output[mask_row, :],
                    cur_label[mask_row, :].argmax(dim=-1),
                    reduction='sum'
                )
            output2[:, st:ed] = output2[:, st:ed].softmax(dim=-1)
        loss = loss / batch_size

        return loss, output2

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def compute_cls_range(self, num_classes):
        cls_range = []
        tt = 0
        for i, c in enumerate(num_classes):
            cls_range.append((tt, tt + c))
            tt += c
        return cls_range

    def model_inference(self, input):
        return self.model(input)[0]