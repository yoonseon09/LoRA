"""
scratch LoRA (lora_scratch_pure.py) vs peft LoRA 비교 실험.

Colab 사용법 (lora_scratch_pure.py 와 이 파일을 같은 폴더에 업로드):
    !pip install -q peft
    !python experiment.py --stage all

단계별로 따로 돌릴 수도 있다:
    --stage pretrain   베이스 모델 사전학습 (enwik8 앞부분) -> base.pt
    --stage equiv      1 step 동치성 테스트 (logits / gradient 비교)
    --stage compare    Tiny Shakespeare 로 두 LoRA 를 seed 별로 학습해 비교
"""

import argparse
import copy
import json
import math
import os
import time
import urllib.request
import zipfile

import torch
from peft import LoraConfig, get_peft_model

import lora_scratch_pure as L

# 두 구현의 수치 비교가 목적이므로 TF32 를 끈다 (A100 등에서 오차가 커지는 것 방지)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DATA_DIR = "./data"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"

# 사전학습과 LoRA 학습에서 반드시 같아야 하는 모델 구조
MODEL_KEYS = dict(n_layer=6, n_head=8, d_model=512, d_ff=2048, block_size=128)


# ------------------------------------------------------------------ data

def download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
    return path


def load_pretrain_tokens(n_bytes):
    """enwik8 (영어 위키피디아) 앞부분 n_bytes 를 바이트 토큰으로."""
    zpath = download(ENWIK8_URL, os.path.join(DATA_DIR, "enwik8.zip"))
    with zipfile.ZipFile(zpath) as z:
        with z.open("enwik8") as f:
            return list(f.read(n_bytes))


def load_finetune_tokens():
    path = download(SHAKESPEARE_URL, os.path.join(DATA_DIR, "shakespeare.txt"))
    return L.ByteTokenizer().encode_file(path)


# ------------------------------------------------------------- pretrain

def pretrain(args):
    """LoRA 없이 전체 파라미터를 학습해서 W0 를 만든다."""
    cfg = L.Config(**MODEL_KEYS, seed=0)
    device = cfg.device
    L.set_seed(0)

    tokens = load_pretrain_tokens(args.pretrain_bytes)
    train_set, val_set = L.build_splits(tokens, cfg.block_size, 0.02)
    train_it = L.BatchIterator(train_set, args.pretrain_batch, seed=0, device=device)
    val_it = L.BatchIterator(val_set, args.pretrain_batch, seed=1, device=device)

    model = L.GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    schedule = L.WarmupCosine({"all": args.pretrain_lr}, warmup_steps=200,
                              max_steps=args.pretrain_steps, min_lr_ratio=0.1)

    # 사전학습은 속도를 위해 mixed precision 사용 (LoRA 비교 단계는 fp32)
    use_amp = device == "cuda"
    bf16_ok = use_amp and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    print(f"pretrain: {len(tokens):,} bytes, {args.pretrain_steps} steps, "
          f"batch {args.pretrain_batch}, amp={amp_dtype if use_amp else 'off'}")
    model.train()
    start = time.time()
    for step in range(args.pretrain_steps):
        for group in optimizer.param_groups:
            group["lr"] = schedule.lr_at(step)["all"]
        x, y = train_it.sample_batch()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
        loss = L.cross_entropy_manual(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 0 or (step + 1) % 250 == 0:
            val = L.evaluate(model, val_it, 20)
            print(f"[pretrain] step {step+1:5d} | train {loss.item():.4f} | val {val:.4f} "
                  f"| {time.time()-start:.0f}s")

    torch.save(model.state_dict(), args.base_ckpt)
    print("saved base weights ->", args.base_ckpt)


# ------------------------------------------------------ build LoRA pair

def make_lora_config(args, seed):
    return L.Config(**MODEL_KEYS,
                    rank=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                    target_modules=["q_proj", "v_proj"],
                    init_a="kaiming", init_b="zeros",     # peft 기본 초기화와 동일
                    lr=args.lr, batch_size=args.batch, micro_batch_size=args.batch,
                    max_steps=args.steps, eval_interval=args.eval_interval,
                    warmup_steps=100, seed=seed, base_ckpt=args.base_ckpt,
                    manual_backward=args.manual_backward)


@torch.no_grad()
def copy_ab(scratch_model, peft_model):
    """scratch 모델의 A/B 값을 peft 모델에 그대로 복사해 시작점을 완전히 맞춘다."""
    inner = peft_model.base_model.model
    for name, module in L.iter_lora_modules(scratch_model):
        target = inner.get_submodule(name)
        target.lora_A["default"].weight.copy_(module.lora_A)
        target.lora_B["default"].weight.copy_(module.lora_B)


def build_pair(cfg):
    """같은 사전학습 가중치 위에 scratch LoRA 와 peft LoRA 를 하나씩 만든다."""
    if not os.path.exists(cfg.base_ckpt):
        raise FileNotFoundError(f"{cfg.base_ckpt} 가 없습니다. 먼저 --stage pretrain 을 돌리세요.")
    L.set_seed(cfg.seed)
    base = L.GPTModel(cfg).to(cfg.device)
    base.load_state_dict(torch.load(cfg.base_ckpt, map_location=cfg.device))
    peft_base = copy.deepcopy(base)

    L.inject_lora(base, cfg)
    scratch_model = base

    peft_cfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                          target_modules=cfg.target_modules, bias="none")
    peft_model = get_peft_model(peft_base, peft_cfg)
    copy_ab(scratch_model, peft_model)

    n_scratch = L.count_parameters(scratch_model)[0]
    n_peft = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if n_scratch != n_peft:
        raise RuntimeError(f"학습 파라미터 수 불일치: scratch {n_scratch} vs peft {n_peft}")
    return scratch_model, peft_model


def peft_param_groups(peft_model, cfg):
    a_params, b_params = [], []
    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            a_params.append(param)
        elif "lora_B" in name:
            b_params.append(param)
        else:
            raise RuntimeError(f"예상치 못한 학습 파라미터: {name}")
    return [{"params": a_params, "lr": cfg.resolved_lr_a, "name": "lora_A"},
            {"params": b_params, "lr": cfg.resolved_lr_b, "name": "lora_B"}]


def clear_grads(model):
    for param in model.parameters():
        param.grad = None


# ----------------------------------------------------- equivalence test

def equivalence_test(args):
    cfg = make_lora_config(args, seed=0)
    tokens = load_finetune_tokens()
    train_set, _ = L.build_splits(tokens, cfg.block_size, cfg.val_ratio)
    scratch_model, peft_model = build_pair(cfg)

    # B=0 이면 dL/dA = 0 이라 A 쪽 비교가 의미가 없으므로 B 를 작은 난수로 채운 뒤 복사
    with torch.no_grad():
        for _, module in L.iter_lora_modules(scratch_model):
            module.lora_B.normal_(0.0, 0.02)
    copy_ab(scratch_model, peft_model)

    x, y = L.BatchIterator(train_set, 8, seed=0, device=cfg.device).sample_batch()
    peft_inner = peft_model.base_model.model
    tol = 1e-4
    all_ok = True

    for mode in ["autograd", "manual"]:
        L.set_manual_mode(scratch_model, mode == "manual")
        clear_grads(scratch_model)
        clear_grads(peft_model)

        logits_s = scratch_model(x)
        logits_p = peft_model(x)
        loss_s = L.cross_entropy_manual(logits_s, y)
        loss_p = L.cross_entropy_manual(logits_p, y)
        loss_s.backward()
        if mode == "manual":
            L.manual_lora_backward(scratch_model)
        loss_p.backward()

        logit_diff = (logits_s - logits_p).abs().max().item()
        grad_diff = 0.0
        for name, module in L.iter_lora_modules(scratch_model):
            target = peft_inner.get_submodule(name)
            grad_diff = max(grad_diff,
                            (module.lora_A.grad - target.lora_A["default"].weight.grad).abs().max().item(),
                            (module.lora_B.grad - target.lora_B["default"].weight.grad).abs().max().item())
        ok = logit_diff < tol and grad_diff < tol
        all_ok &= ok
        print(f"[equiv/{mode:8s}] loss scratch {loss_s.item():.6f} peft {loss_p.item():.6f} | "
              f"max|logit diff| {logit_diff:.2e} | max|grad diff| {grad_diff:.2e} | "
              f"{'PASS' if ok else 'FAIL'}")

    L.set_manual_mode(scratch_model, False)
    print("동치성 테스트:", "전부 통과" if all_ok else "차이가 있습니다. 구현을 확인하세요.")
    return all_ok


# ----------------------------------------------------------- comparison

def run_training(model, param_groups, train_it, val_it, final_it, cfg, label, use_manual):
    """lora_scratch_pure.train 과 같은 로직을 두 모델에 똑같이 적용하는 루프."""
    optimizer = L.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas,
                        eps=cfg.eps, weight_decay=cfg.weight_decay)
    schedule = L.WarmupCosine({g["name"]: g["lr"] for g in optimizer.param_groups},
                              cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    if use_manual:
        L.set_manual_mode(model, True)

    cuda = cfg.device == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated()

    history = []
    model.train()
    start = time.time()
    for step in range(cfg.max_steps):
        optimizer.set_lr(schedule.lr_at(step))
        optimizer.zero_grad()
        total = 0.0
        for _ in range(cfg.accumulation_steps):
            x, y = train_it.sample_batch()
            loss = L.cross_entropy_manual(model(x), y) / cfg.accumulation_steps
            loss.backward()
            if use_manual:
                L.manual_lora_backward(model)
            total += loss.item()
        grad_norm = L.clip_grad_norm(params, cfg.grad_clip)
        optimizer.step()

        if step == 0 or (step + 1) % cfg.eval_interval == 0:
            val = L.evaluate(model, val_it, cfg.eval_batches)
            history.append({"step": step + 1, "train_loss": total, "val_loss": val})
            print(f"  [{label:7s}] step {step+1:5d} | train {total:.4f} | val {val:.4f} "
                  f"| gnorm {grad_norm:.3f}")

    if cuda:
        torch.cuda.synchronize()
    elapsed = time.time() - start
    extra_mem = (torch.cuda.max_memory_allocated() - mem_start) / 2**20 if cuda else float("nan")
    L.set_manual_mode(model, False)

    final_val = L.evaluate(model, final_it, 100)   # 고정된 100 배치로 최종 평가
    return {"history": history, "final_val_loss": final_val,
            "final_val_ppl": math.exp(final_val), "train_time_s": elapsed,
            "extra_peak_mem_mb": extra_mem}


def mean_std(values):
    m = sum(values) / len(values)
    s = math.sqrt(sum((v - m) ** 2 for v in values) / max(1, len(values) - 1))
    return m, s


def compare(args):
    tokens = load_finetune_tokens()
    results = {"scratch": [], "peft": []}
    base_losses = []

    for seed in args.seeds:
        print(f"\n===== seed {seed} =====")
        cfg = make_lora_config(args, seed)
        train_set, val_set = L.build_splits(tokens, cfg.block_size, cfg.val_ratio)
        scratch_model, peft_model = build_pair(cfg)

        base_it = L.BatchIterator(val_set, cfg.micro_batch_size, seed=999, device=cfg.device)
        base_losses.append(L.evaluate(scratch_model, base_it, 100))   # B=0 이므로 = 베이스 성능
        print(f"  LoRA 적용 전 (사전학습 베이스) val loss: {base_losses[-1]:.4f}")

        runs = [("scratch", scratch_model, L.build_param_groups(scratch_model, cfg), cfg.manual_backward),
                ("peft", peft_model, peft_param_groups(peft_model, cfg), False)]
        for label, model, groups, use_manual in runs:
            # 두 모델이 완전히 같은 배치 순서를 보도록 iterator 를 매번 새로 만든다
            train_it = L.BatchIterator(train_set, cfg.micro_batch_size, seed=seed, device=cfg.device)
            val_it = L.BatchIterator(val_set, cfg.micro_batch_size, seed=seed + 1, device=cfg.device)
            final_it = L.BatchIterator(val_set, cfg.micro_batch_size, seed=999, device=cfg.device)
            results[label].append(run_training(model, groups, train_it, val_it, final_it,
                                               cfg, label, use_manual))
        del scratch_model, peft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n===== 요약 (평균 ± 표준편차, seed %s) =====" % args.seeds)
    m, s = mean_std(base_losses)
    print(f"LoRA 적용 전 val loss: {m:.4f} ± {s:.4f}")
    summary = {}
    for label, runs in results.items():
        row = {key: mean_std([r[key] for r in runs])
               for key in ["final_val_loss", "final_val_ppl", "train_time_s", "extra_peak_mem_mb"]}
        summary[label] = row
        print(f"{label:7s} | val loss {row['final_val_loss'][0]:.4f} ± {row['final_val_loss'][1]:.4f} "
              f"| ppl {row['final_val_ppl'][0]:.3f} | time {row['train_time_s'][0]:.1f}s "
              f"| extra mem {row['extra_peak_mem_mb'][0]:.0f}MB")
    diffs = [abs(a["final_val_loss"] - b["final_val_loss"])
             for a, b in zip(results["scratch"], results["peft"])]
    print(f"seed 별 |scratch - peft| 최종 val loss 차이: {[round(d, 5) for d in diffs]}")

    with open("compare_results.json", "w") as f:
        json.dump({"args": vars(args), "base_val_loss": base_losses,
                   "summary": summary, "runs": results}, f, indent=2)
    print("saved -> compare_results.json")
    plot(results)


def plot(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(7, 4))
    styles = {"scratch": ("tab:blue", "-"), "peft": ("tab:orange", "--")}
    for label, runs in results.items():
        color, line = styles[label]
        for i, run in enumerate(runs):
            steps = [h["step"] for h in run["history"]]
            vals = [h["val_loss"] for h in run["history"]]
            plt.plot(steps, vals, color=color, linestyle=line, alpha=0.8,
                     label=label if i == 0 else None)
    plt.xlabel("step")
    plt.ylabel("val loss")
    plt.title("scratch LoRA vs peft LoRA (Tiny Shakespeare)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("compare_results.png", dpi=120)
    print("saved -> compare_results.png")


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["pretrain", "equiv", "compare", "all"])
    parser.add_argument("--base_ckpt", default="base.pt")
    # 사전학습
    parser.add_argument("--pretrain_bytes", type=int, default=5_000_000)
    parser.add_argument("--pretrain_steps", type=int, default=3000)
    parser.add_argument("--pretrain_batch", type=int, default=64)
    parser.add_argument("--pretrain_lr", type=float, default=6e-4)
    # LoRA 비교
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--manual_backward", action="store_true",
                        help="scratch 쪽을 manual gradient 경로로 학습")
    args = parser.parse_args()

    if args.stage in ("pretrain", "all"):
        if args.stage == "all" and os.path.exists(args.base_ckpt):
            print(f"{args.base_ckpt} 가 이미 있어 사전학습을 건너뜁니다.")
        else:
            pretrain(args)
    if args.stage in ("equiv", "all"):
        equivalence_test(args)
    if args.stage in ("compare", "all"):
        compare(args)


if __name__ == "__main__":
    main()
