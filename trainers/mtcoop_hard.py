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

    def _forward(self, prompts, tokenized_prompts):
        """
        
        todo: dropout prompts
        
        prompts: embeded txt
        tokenized_prompts: tokens of prompts
        """
        
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
    def forward(self, prompts, tokenized_prompts):
        if isinstance(prompts, list):
            return torch.stack([self._forward(p, t) for p, t in zip(prompts, tokenized_prompts)], dim=0).mean(dim=0)
        else:
            return self._forward(prompts, tokenized_prompts)


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, task_share):
        super().__init__()
        
        self.task_share = task_share
        if self.task_share:  # lean prompts for each task separately
            self.learners = nn.ModuleList(
                [PromptLearner(cfg, x, clip_model, False) for x in classnames]
            )
            self.tokenized_prompts = torch.cat([m.tokenized_prompts for m in self.learners], dim=0)
        else:
            n_cls = len(classnames)
            n_ctx = cfg.TRAINER.MTCOOPH.N_CTX
            ctx_init = cfg.TRAINER.MTCOOPH.CTX_INIT
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
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                prompt_prefix = ctx_init

            else:
                # random initialization
                if cfg.TRAINER.MTCOOPH.CSC:
                    print("Initializing class-specific contexts")
                    ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                else:
                    print("Initializing a generic context")
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)

            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")

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
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

            self.n_cls = n_cls
            self.n_ctx = n_ctx
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
            self.name_lens = name_lens
            self.token_position = cfg.TRAINER.MTCOOPH.TOKEN_POSITION

    def forward(self):
        
        if self.task_share:
            results = [m() for m in self.learners]
            return torch.cat(results, dim=0)
        else:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix

            if self.token_position == "end":
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )

            elif self.token_position == "middle":
                half_n_ctx = self.n_ctx // 2
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.name_lens[i]
                    prefix_i = prefix[i : i + 1, :, :]
                    class_i = suffix[i : i + 1, :name_len, :]
                    suffix_i = suffix[i : i + 1, name_len:, :]
                    ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                    ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,     # (1, 1, dim)
                            ctx_i_half1,  # (1, n_ctx//2, dim)
                            class_i,      # (1, name_len, dim)
                            ctx_i_half2,  # (1, n_ctx//2, dim)
                            suffix_i,     # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)

            elif self.token_position == "front":
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.name_lens[i]
                    prefix_i = prefix[i : i + 1, :, :]
                    class_i = suffix[i : i + 1, :name_len, :]
                    suffix_i = suffix[i : i + 1, name_len:, :]
                    ctx_i = ctx[i : i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            class_i,   # (1, name_len, dim)
                            ctx_i,     # (1, n_ctx, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)

            else:
                raise ValueError

            return prompts

        
class CustomCLIP(nn.Module):
    def __init__(self, cfg, tasknames, classnames, clip_model):
        super().__init__()
        
        is_tsc = cfg.TRAINER.MTCOOPH.TSC  # task-specific context
        
        if is_tsc:  # learn one prompt for each task
            prompt_learner = PromptLearner(cfg, classnames, clip_model, True)
        else:  # all cls share one prompt or learn one prompt for each cls
            classnames_ = list(itertools.chain.from_iterable(classnames))
            prompt_learner = PromptLearner(cfg, classnames_, clip_model, False)
        
        self.prompt_learner = prompt_learner
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
    
    def forward(self, image):
        
        image_features = self.image_encoder(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)  # text_features: m*d
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits
    

@TRAINER_REGISTRY.register()
class MTCoOpHard(TrainerX):
    """
    Multi-task Context Optimization (CoOp) baselines.
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
        assert cfg.TRAINER.MTCOOPH.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        
        classnames = self.dm.dataset.taskid2cname
        self.taskid2tname = self.dm.dataset.taskid2tname
        self.taskid2cname = self.dm.dataset.taskid2cname
        self.taskid2type = self.dm.dataset.taskid2type
        self.taskid2metric = self.dm.dataset.taskid2metric
        self.num_tasks = len(self.taskid2tname)
        self.cls_range = self.compute_cls_range([len(x) for x in self.taskid2cname])

        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MTCOOPH.PREC == "fp32" or cfg.TRAINER.MTCOOPH.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.taskid2tname, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        cnt = 0
        for param_group in self.optim.param_groups:
            for g in param_group["params"]:
                if g.requires_grad:
                    print(g.shape, g)
                    cnt += 1
        print("number learnable params", cnt)
        
        self.scaler = GradScaler() if cfg.TRAINER.MTCOOPH.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        output2 = None
        prec = self.cfg.TRAINER.MTCOOPH.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss, output2 = self.mutitask_loss(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss, output2 = self.mutitask_loss(output, label)
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
            mask = torch.sum(cur_label != -1, dim=-1) > 0
            typ = self.taskid2type[i]
            assert typ == "sl"
            loss += F.cross_entropy(cur_output[mask, :], cur_label[mask, :].argmax(dim=-1), reduction='sum')
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
    
    def model_inference(self, input):
        return self.model(input)
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        for batch_idx, batch in enumerate(tqdm.tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)
        
        results = self.evaluator.evaluate()
        
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)
        
        return list(results.values())[0]
    
    def compute_cls_range(self, num_classes):
        cls_range = []
        tt = 0
        for i, c in enumerate(num_classes):
            cls_range.append((tt, tt+c))
            tt += c
        return cls_range