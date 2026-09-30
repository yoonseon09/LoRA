"""
조정 가능한 축:
  - rank, layer/모듈별 rank (rank_pattern)
  - lora_alpha, 모듈별 alpha (alpha_pattern)
  - 어디에 붙일지 (target_modules)
  - A/B 초기화 방식 (init_a, init_b)
  - A/B 개별 freeze (train_a, train_b)
  - A/B 개별 learning rate (lr_a, lr_b)
  - A/B 행렬 직접 주입 (set_lora_matrices)
  - autograd 대신 수동 gradient 계산 (manual_backward)
  - 실효 배치 (batch_size / micro_batch_size)
  - LR 스케줄, optimizer 하이퍼파라미터
"""

import argparse
import json
import math
import os
import random
import time
import warnings

import torch
import torch.nn as nn


# ---------------------------------------------------------------- 1. Config

class Config:
    def __init__(self, **kwargs):
        self.vocab_size = 256
        self.n_layer = 6
        self.n_head = 8
        self.d_model = 512
        self.d_ff = 2048
        self.block_size = 128
        self.dropout = 0.0
        self.tie_embeddings = True

        self.batch_size = 16
        self.micro_batch_size = 4
        self.max_steps = 2000
        self.eval_interval = 200
        self.eval_batches = 20

        self.lr = 3e-4
        self.lr_a = None
        self.lr_b = None
        self.betas = (0.9, 0.95)
        self.eps = 1e-8
        self.weight_decay = 0.0
        self.warmup_steps = 100
        self.min_lr_ratio = 0.1
        self.grad_clip = 1.0

        self.rank = 8
        self.lora_alpha = 16.0
        self.lora_dropout = 0.0
        self.target_modules = ["q_proj", "v_proj"]
        self.init_a = "kaiming"
        self.init_b = "zeros"
        self.train_a = True
        self.train_b = True
        self.rank_pattern = {}
        self.alpha_pattern = {}
        self.manual_backward = False

        self.data_path = None
        self.val_ratio = 0.05
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = "float32"
        self.out_dir = "./runs"
        self.run_name = "lora"
        self.base_ckpt = None      # 사전학습된 베이스 가중치 경로 (None 이면 랜덤 초기화)

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(f"unknown config key: {key}")
            setattr(self, key, value)

        self.validate()

    def validate(self):
        """조용히 잘못 도는 조합을 시작 시점에 잡는다."""
        if self.rank > 0 and not self.train_b and self.init_b == "zeros":
            raise ValueError(
                "train_b=False 와 init_b='zeros' 조합은 학습이 전혀 일어나지 않습니다. "
                "dL/dA = scaling * B^T G 인데 B=0 이면 A 의 gradient 도 0 이 됩니다. "
                "init_b 를 'normal' 또는 'orthogonal' 로 바꾸거나 train_b=True 로 두세요."
            )
        if self.rank > 0 and not self.train_a and not self.train_b:
            raise ValueError(
                "train_a=False 와 train_b=False 를 동시에 지정하면 학습 파라미터가 "
                "0 개가 되어 backward 가 실패합니다."
            )
        if self.init_a == "zeros" and self.init_b == "zeros":
            raise ValueError("init_a 와 init_b 를 모두 zeros 로 두면 영원히 학습되지 않습니다.")
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model({self.d_model}) 이 n_head({self.n_head}) 로 나누어떨어져야 합니다.")
        if self.batch_size % self.micro_batch_size != 0:
            warnings.warn(
                f"batch_size({self.batch_size}) 가 micro_batch_size({self.micro_batch_size}) 의 "
                f"배수가 아닙니다. 실효 배치는 {self.accumulation_steps * self.micro_batch_size} 가 됩니다."
            )

    @property
    def accumulation_steps(self):
        return max(1, self.batch_size // self.micro_batch_size)

    @property
    def resolved_lr_a(self):
        return self.lr if self.lr_a is None else self.lr_a

    @property
    def resolved_lr_b(self):
        return self.lr if self.lr_b is None else self.lr_b

    def to_dict(self):
        return {k: v for k, v in vars(self).items()}

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args_into_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overrides", type=str, default="{}")
    args = parser.parse_args()
    return Config(**json.loads(args.overrides))


# ------------------------------------------------------------- 2. Tokenizer

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def encode_file(self, path):
        with open(path, "rb") as f:
            return list(f.read())


# ------------------------------------------------------------------ 3. Data

class LMDataset:
    def __init__(self, token_ids, block_size):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        if len(self.data) < block_size + 2:
            raise ValueError(
                f"데이터가 너무 짧습니다: 토큰 {len(self.data)} 개, block_size {block_size}. "
                f"최소 {block_size + 2} 개가 필요합니다."
            )

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def get(self, index):
        chunk = self.data[index:index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class BatchIterator:
    def __init__(self, dataset, batch_size, shuffle=True, seed=0, device="cpu"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def sample_batch(self):
        n = len(self.dataset)
        if self.shuffle:
            indices = torch.randint(0, n, (self.batch_size,), generator=self.generator)
        else:
            start = torch.randint(0, max(1, n - self.batch_size), (1,), generator=self.generator)
            indices = torch.arange(start.item(), start.item() + self.batch_size) % n
        xs, ys = [], []
        for index in indices.tolist():
            x, y = self.dataset.get(index)
            xs.append(x)
            ys.append(y)
        x_batch = torch.stack(xs).to(self.device, non_blocking=True)
        y_batch = torch.stack(ys).to(self.device, non_blocking=True)
        return x_batch, y_batch


def build_splits(token_ids, block_size, val_ratio=0.05):
    split_point = int(len(token_ids) * (1.0 - val_ratio))
    return (LMDataset(token_ids[:split_point], block_size),
            LMDataset(token_ids[split_point:], block_size))


# ------------------------------------------------------------ 4. Base model

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_head
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2) #n_head로 쪼개서 'multi'head로..
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head) #유사도계산
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        h = self.up_proj(x)
        h = h * torch.sigmoid(1.702 * h)
        return self.dropout(self.down_proj(h))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"입력 길이 T={T} 가 block_size={self.config.block_size} 를 초과합니다. "
                f"pos_emb 와 causal_mask 가 block_size 까지만 정의되어 있습니다."
            )
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad_(False)
        return self


# ----------------------------------------------------------- 5. LoRA module wq wk wv가 ab로부터 어떻게 변경하는지 

class LoRALinear(nn.Module):
    """y = x W0^T + scaling * (x A^T) B^T,  W0 frozen."""

    def __init__(self, in_features, out_features, bias=True, rank=8, lora_alpha=16.0,
                 lora_dropout=0.0, init_a="kaiming", init_b="zeros",
                 train_a=True, train_b=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha          # scaling 은 property 로 파생
        self.lora_dropout = lora_dropout
        self.merged = False
        self.enabled = True
        self.manual_mode = False              # True 면 ΔW 를 명시적으로 만들어 G 를 캡처
        self._delta_ref = None                # 직전 forward 의 ΔW 텐서 (grad 캡처용)

        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory),
                                   requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features, **factory),
                                       requires_grad=train_a)
            self.lora_B = nn.Parameter(torch.empty(out_features, rank, **factory),
                                       requires_grad=train_b)
            self.reset_lora_parameters(init_a, init_b)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def scaling(self):
        """lora_alpha 를 나중에 바꿔도 즉시 반영되도록 파생값으로 둔다."""
        return (self.lora_alpha / self.rank) if self.rank > 0 else 0.0

    @property
    def has_lora(self):
        return self.rank > 0 and self.lora_A is not None

    @classmethod
    def from_linear(cls, linear, rank, lora_alpha, lora_dropout,
                    init_a, init_b, train_a, train_b):
        module = cls(linear.in_features, linear.out_features,
                     bias=linear.bias is not None, rank=rank, lora_alpha=lora_alpha,
                     lora_dropout=lora_dropout, init_a=init_a, init_b=init_b,
                     train_a=train_a, train_b=train_b,
                     device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def reset_lora_parameters(self, init_a="kaiming", init_b="zeros"):
        if not self.has_lora:
            return self
        with torch.no_grad():
            if init_a == "kaiming":
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            elif init_a == "normal":
                nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            elif init_a == "orthogonal":
                nn.init.orthogonal_(self.lora_A)
            elif init_a == "zeros":
                nn.init.zeros_(self.lora_A)
            else:
                raise ValueError(f"unknown init_a: {init_a}")

            if init_b == "zeros":
                nn.init.zeros_(self.lora_B)
            elif init_b == "normal":
                nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
            elif init_b == "orthogonal":
                nn.init.orthogonal_(self.lora_B)
            else:
                raise ValueError(f"unknown init_b: {init_b}")
        return self

    def set_lora_matrices(self, A=None, B=None):
        if not self.has_lora:
            raise RuntimeError("rank=0 모듈에는 A/B 가 없습니다.")
        with torch.no_grad():
            if A is not None:
                if tuple(A.shape) != tuple(self.lora_A.shape):
                    raise ValueError(f"A shape {tuple(A.shape)} != {tuple(self.lora_A.shape)}")
                self.lora_A.copy_(A)
            if B is not None:
                if tuple(B.shape) != tuple(self.lora_B.shape):
                    raise ValueError(f"B shape {tuple(B.shape)} != {tuple(self.lora_B.shape)}")
                self.lora_B.copy_(B)
        return self

    def get_lora_matrices(self):
        if not self.has_lora:
            return None, None
        return self.lora_A.detach().clone(), self.lora_B.detach().clone()

    def delta_weight(self):
        if not self.has_lora:
            return torch.zeros_like(self.weight)
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x):
        if not self.enabled or self.merged or not self.has_lora:
            return nn.functional.linear(x, self.weight, self.bias)

        if self.manual_mode:
            # ΔW 를 명시적으로 만들어 leaf 로 두면 backward 후 ΔW.grad 가 곧 G 다.
            # A/B 는 detach 하므로 autograd 는 A/B 로 흐르지 않고,
            # manual_lora_backward 가 G 로부터 직접 grad_A / grad_B 를 만든다.
            delta = (self.lora_B.detach() @ self.lora_A.detach()) * self.scaling
            delta.requires_grad_(True)
            delta.retain_grad() #여기서는 gradient 보존만 진행함. 

            self._delta_ref = delta
            h = x
            if self.lora_dropout > 0.0 and self.training:
                h = nn.functional.dropout(h, p=self.lora_dropout)
            return nn.functional.linear(h, self.weight + delta, self.bias)

        base = nn.functional.linear(x, self.weight, self.bias)
        h = x
        if self.lora_dropout > 0.0 and self.training:
            h = nn.functional.dropout(h, p=self.lora_dropout)
        h = nn.functional.linear(h, self.lora_A)
        h = nn.functional.linear(h, self.lora_B)
        return base + h * self.scaling

    def merge(self):
        if self.merged or not self.has_lora:
            return self
        with torch.no_grad():
            self.weight.add_(self.delta_weight())
        self.merged = True
        return self

    def unmerge(self):
        if not self.merged or not self.has_lora:
            return self
        with torch.no_grad():
            self.weight.sub_(self.delta_weight())
        self.merged = False
        return self

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
                f"alpha={self.lora_alpha}, scaling={self.scaling:.4f}, merged={self.merged}")


# -------------------------------------------------------------- 6. Injection

def resolve_rank_and_alpha(module_name, config):
    rank, alpha = config.rank, config.lora_alpha
    for pattern, value in config.rank_pattern.items():
        if pattern in module_name:
            rank = value
    for pattern, value in config.alpha_pattern.items():
        if pattern in module_name:
            alpha = value
    return rank, alpha


def find_target_linears(model, target_modules):
    found = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module_name.split(".")[-1] in target_modules:
            found.append((module_name, module))
    return found


def replace_module(model, module_name, new_module):
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora(model, config):
    model.freeze_all()
    injected = []
    targets = find_target_linears(model, config.target_modules)
    if not targets:
        raise ValueError(
            f"target_modules={config.target_modules} 에 해당하는 nn.Linear 를 찾지 못했습니다. "
            f"이름을 확인하세요."
        )
    for module_name, linear in targets:
        rank, alpha = resolve_rank_and_alpha(module_name, config)
        lora_module = LoRALinear.from_linear(
            linear, rank=rank, lora_alpha=alpha, lora_dropout=config.lora_dropout,
            init_a=config.init_a, init_b=config.init_b,
            train_a=config.train_a, train_b=config.train_b)
        replace_module(model, module_name, lora_module)
        injected.append((module_name, rank, alpha))
    return injected


def iter_lora_modules(model):
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield module_name, module


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def set_lora_enabled(model, enabled):
    for _, module in iter_lora_modules(model):
        module.enabled = enabled


def set_manual_mode(model, enabled):
    for _, module in iter_lora_modules(model):
        module.manual_mode = enabled


# --------------------------------------------------------------- 7. Forward

def forward_pass(model, x, y, loss_fn):
    logits = model(x)
    return logits, loss_fn(logits, y)


@torch.no_grad()
def forward_no_lora(model, x):
    set_lora_enabled(model, False)
    logits = model(x)
    set_lora_enabled(model, True)
    return logits


# ------------------------------------------------------------------ 8. Loss

def cross_entropy_manual(logits, targets, ignore_index=-100):
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    picked = shifted.gather(1, targets.clamp_min(0).unsqueeze(1)).squeeze(1)
    nll = log_sum_exp - picked
    mask = targets != ignore_index
    if mask.all():
        return nll.mean()
    return (nll * mask).sum() / mask.sum().clamp_min(1)


def perplexity_from_loss(loss_value):
    return math.exp(min(loss_value, 20.0))


# -------------------------------------------------------------- 9. Backward autograd + algebra grad 방법 

def autograd_backward(loss):
    loss.backward()


def manual_lora_backward(model):
    """manual_mode forward 가 남긴 ΔW.grad (= G) 로부터 A/B 의 grad 를 직접 만든다.

    autograd 가 자동으로 해주는 계산을 LoRA 의 gradient 공식 그대로 써놓은 것이다.

        G     = dL/d(ΔW)              [out, in]
        dL/dB = scaling * G  A^T      [out,in] @ [in,rank] -> [out,rank]
        dL/dA = scaling * B^T G       [rank,out] @ [out,in] -> [rank,in]

    autograd 경로와 수치적으로 일치하는 것을 test 에서 검증한다.
    """
    touched = []
    for module_name, module in iter_lora_modules(model):
        if not module.has_lora or module._delta_ref is None:
            continue
        G = module._delta_ref.grad #backward에서 고치기
        if G is None:
            continue
        s = module.scaling
        with torch.no_grad():
            if module.lora_B.requires_grad:
                grad_B = (G @ module.lora_A.T) * s
                module.lora_B.grad = grad_B if module.lora_B.grad is None else module.lora_B.grad + grad_B
            if module.lora_A.requires_grad:
                grad_A = (module.lora_B.T @ G) * s
                module.lora_A.grad = grad_A if module.lora_A.grad is None else module.lora_A.grad + grad_A
        module._delta_ref = None
        touched.append(module_name)
    return touched


@torch.no_grad()
def collect_grad_norms(model):
    norms = {}
    for module_name, module in iter_lora_modules(model):
        if not module.has_lora:
            continue
        entry = {}
        if module.lora_A.grad is not None:
            entry["A"] = module.lora_A.grad.norm().item()
        if module.lora_B.grad is not None:
            entry["B"] = module.lora_B.grad.norm().item()
        norms[module_name] = entry
    return norms


@torch.no_grad()
def collect_lora_stats(model):
    stats = {}
    for module_name, module in iter_lora_modules(model):
        base_norm = module.weight.norm()
        if not module.has_lora:
            stats[module_name] = {"rank": 0, "A_norm": 0.0, "B_norm": 0.0,
                                  "delta_norm": 0.0, "base_norm": base_norm.item(),
                                  "ratio": 0.0}
            continue
        A, B = module.get_lora_matrices()
        delta = module.delta_weight()
        stats[module_name] = {
            "rank": module.rank,
            "A_norm": A.norm().item(),
            "B_norm": B.norm().item(),
            "delta_norm": delta.norm().item(),
            "base_norm": base_norm.item(),
            "ratio": (delta.norm() / base_norm.clamp_min(1e-12)).item(),
        }
    return stats


# ------------------------------------------------- 10. Optimizer & scheduler

class AdamW:
    def __init__(self, param_groups, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.param_groups = []
        for group in param_groups:
            self.param_groups.append({
                "params": list(group["params"]),
                "lr": group.get("lr", lr),
                "betas": group.get("betas", betas),
                "eps": group.get("eps", eps),
                "weight_decay": group.get("weight_decay", weight_decay),
                "name": group.get("name", "default"),
            })
        self.state = {}
        self.step_count = 0

    def zero_grad(self):
        for group in self.param_groups:
            for param in group["params"]:
                param.grad = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state.setdefault(param, {
                    "m": torch.zeros_like(param),
                    "v": torch.zeros_like(param),
                })
                if group["weight_decay"] != 0.0:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])
                state["m"].mul_(beta1).add_(param.grad, alpha=1.0 - beta1)
                state["v"].mul_(beta2).addcmul_(param.grad, param.grad, value=1.0 - beta2)
                m_hat = state["m"] / (1.0 - beta1 ** self.step_count)
                v_hat = state["v"] / (1.0 - beta2 ** self.step_count)
                param.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-group["lr"])

    def set_lr(self, lr_map):
        for group in self.param_groups:
            if group["name"] in lr_map:
                group["lr"] = lr_map[group["name"]]

    def state_dict(self):
        return {"step_count": self.step_count,
                "lrs": {g["name"]: g["lr"] for g in self.param_groups}}


class WarmupCosine:
    def __init__(self, base_lrs, warmup_steps, max_steps, min_lr_ratio=0.1):
        self.base_lrs = dict(base_lrs)
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

    def lr_at(self, step):
        result = {}
        for name, base_lr in self.base_lrs.items():
            if step < self.warmup_steps:
                scale = (step + 1) / max(1, self.warmup_steps)
            else:
                progress = min(1.0, (step - self.warmup_steps) /
                               max(1, self.max_steps - self.warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
            result[name] = base_lr * scale
        return result


def build_param_groups(model, config):
    a_params, b_params = [], []
    for _, module in iter_lora_modules(model):
        if module.has_lora and module.lora_A.requires_grad:
            a_params.append(module.lora_A)
        if module.has_lora and module.lora_B.requires_grad:
            b_params.append(module.lora_B)
    groups = []
    if a_params:
        groups.append({"params": a_params, "lr": config.resolved_lr_a, "name": "lora_A"})
    if b_params:
        groups.append({"params": b_params, "lr": config.resolved_lr_b, "name": "lora_B"})
    return groups


def clip_grad_norm(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    if max_norm > 0 and total > max_norm:
        coefficient = max_norm / (total + 1e-6)
        for g in grads:
            g.mul_(coefficient)
    return total.item()


# ------------------------------------------------------------ 11. Train loop

def train(model, train_iterator, val_iterator, config):
    merged = [n for n, m in iter_lora_modules(model) if m.merged]
    if merged:
        warnings.warn(f"merged 상태인 모듈 {len(merged)} 개를 자동으로 unmerge 합니다.")
        unmerge_all(model)

    param_groups = build_param_groups(model, config)
    if not param_groups:
        raise RuntimeError("학습 가능한 LoRA 파라미터가 없습니다. train_a / train_b / rank 를 확인하세요.")

    optimizer = AdamW(param_groups, lr=config.lr, betas=config.betas,
                      eps=config.eps, weight_decay=config.weight_decay)
    scheduler = WarmupCosine(
        base_lrs={g["name"]: g["lr"] for g in optimizer.param_groups},
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        min_lr_ratio=config.min_lr_ratio)

    set_manual_mode(model, config.manual_backward)
    trainable_params = [p for g in optimizer.param_groups for p in g["params"]]
    history = []
    model.train()
    start_time = time.time()

    for step in range(config.max_steps):
        optimizer.set_lr(scheduler.lr_at(step))
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for _ in range(config.accumulation_steps):
            x, y = train_iterator.sample_batch()
            logits = model(x)
            loss = cross_entropy_manual(logits, y) / config.accumulation_steps
            loss.backward()
            if config.manual_backward:
                manual_lora_backward(model)
            accumulated_loss += loss.item() * config.accumulation_steps

        grad_norm = clip_grad_norm(trainable_params, config.grad_clip)
        optimizer.step()

        mean_loss = accumulated_loss / config.accumulation_steps
        if (step + 1) % config.eval_interval == 0 or step == 0:
            val_loss = evaluate(model, val_iterator, config.eval_batches)
            record = {
                "step": step + 1,
                "train_loss": mean_loss,
                "val_loss": val_loss,
                "val_ppl": perplexity_from_loss(val_loss),
                "grad_norm": grad_norm,
                "lr": scheduler.lr_at(step),
                "elapsed": time.time() - start_time,
            }
            history.append(record)
            print(f"step {step+1:5d} | train {mean_loss:.4f} | val {val_loss:.4f} "
                  f"| ppl {record['val_ppl']:.2f} | gnorm {grad_norm:.3f}")

    set_manual_mode(model, False)
    return history


# ------------------------------------------------------------ 12. Evaluation

@torch.no_grad()
def evaluate(model, iterator, num_batches):
    was_training = model.training          # 호출 전 모드를 복원한다
    manual_flags = {n: m.manual_mode for n, m in iter_lora_modules(model)}
    set_manual_mode(model, False)
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        x, y = iterator.sample_batch()
        total += cross_entropy_manual(model(x), y).item()
    model.train(was_training)
    for n, m in iter_lora_modules(model):
        m.manual_mode = manual_flags[n]
    return total / max(1, num_batches)


@torch.no_grad()
def compare_base_vs_lora(model, iterator, num_batches):
    set_lora_enabled(model, False)
    base_loss = evaluate(model, iterator, num_batches)
    set_lora_enabled(model, True)
    lora_loss = evaluate(model, iterator, num_batches)
    return {"base_loss": base_loss, "lora_loss": lora_loss,
            "delta": base_loss - lora_loss}


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        context = ids[:, -block_size:]
        logits = model(context)[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    model.train(was_training)
    return tokenizer.decode(ids[0].tolist())


# -------------------------------------------------------- 13. Save/load/merge

def save_adapter(model, config, history, path):
    adapter_state = {}
    for module_name, module in iter_lora_modules(model):
        if not module.has_lora:
            continue
        if module.merged:
            raise RuntimeError(
                f"{module_name} 이 merged 상태입니다. unmerge_all(model) 후 저장하세요."
            )
        adapter_state[module_name] = {
            "lora_A": module.lora_A.detach().cpu(),
            "lora_B": module.lora_B.detach().cpu(),
            "rank": module.rank,
            "lora_alpha": module.lora_alpha,
        }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"adapter": adapter_state, "config": config.to_dict(),
                "history": history}, path)
    return path


def load_adapter(model, path, strict=True):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    adapter_state = payload["adapter"]
    loaded = []
    for module_name, module in iter_lora_modules(model):
        if not module.has_lora:
            continue
        if module_name not in adapter_state:
            if strict:
                raise KeyError(f"missing adapter entry: {module_name}")
            continue
        entry = adapter_state[module_name]
        if entry["rank"] != module.rank:
            raise ValueError(f"rank mismatch at {module_name}: "
                             f"{entry['rank']} vs {module.rank}")
        if entry["lora_alpha"] != module.lora_alpha:
            raise ValueError(f"alpha mismatch at {module_name}: "
                             f"{entry['lora_alpha']} vs {module.lora_alpha} "
                             f"(scaling 이 달라져 다른 모델이 됩니다)")
        module.set_lora_matrices(entry["lora_A"].to(module.weight.device),
                                 entry["lora_B"].to(module.weight.device))
        loaded.append(module_name)
    return loaded, payload.get("config", {}), payload.get("history", [])


def merge_all(model):
    return [n for n, m in iter_lora_modules(model) if (m.merge() or True) and m.merged]


def unmerge_all(model):
    for _, module in iter_lora_modules(model):
        module.unmerge()


def export_merged_state_dict(model):
    was_merged = {n: m.merged for n, m in iter_lora_modules(model)}
    merge_all(model)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
             if not name.endswith("lora_A") and not name.endswith("lora_B")}
    for n, m in iter_lora_modules(model):          # 원래 상태로만 되돌린다
        if not was_merged[n]:
            m.unmerge()
    return state


# ------------------------------------------------------------------- 14. main

def build_everything(config, token_ids=None):
    set_seed(config.seed)
    tokenizer = ByteTokenizer()
    config.vocab_size = tokenizer.vocab_size
    if token_ids is None:
        if config.data_path:
            token_ids = tokenizer.encode_file(config.data_path)
        else:
            token_ids = tokenizer.encode("hello world. " * 5000)
    train_set, val_set = build_splits(token_ids, config.block_size, config.val_ratio)
    train_iterator = BatchIterator(train_set, config.micro_batch_size,
                                   seed=config.seed, device=config.device)
    val_iterator = BatchIterator(val_set, config.micro_batch_size,
                                 seed=config.seed + 1, device=config.device)
    model = GPTModel(config).to(config.device)
    if config.base_ckpt:
        # LoRA 를 붙이기 전에 사전학습된 W0 를 불러온다
        state = torch.load(config.base_ckpt, map_location=config.device)
        model.load_state_dict(state)
        print("loaded base weights:", config.base_ckpt)
    injected = inject_lora(model, config)
    return model, tokenizer, train_iterator, val_iterator, injected


def main():
    config = parse_args_into_config()
    model, tokenizer, train_it, val_it, injected = build_everything(config)
    trainable, total = count_parameters(model)
    print(f"injected {len(injected)} modules | trainable {trainable:,} / {total:,} "
          f"({100*trainable/total:.3f}%)")
    history = train(model, train_it, val_it, config)
    out = os.path.join(config.out_dir, f"{config.run_name}.pt")
    save_adapter(model, config, history, out)
    config.save(os.path.join(config.out_dir, f"{config.run_name}.json"))
    print("saved:", out)


if __name__ == "__main__":
    main()
