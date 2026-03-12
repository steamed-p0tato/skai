#!/usr/bin/env python3
"""
Pre-train a foundational language model FROM SCRATCH on FineWeb-Edu.
Random weight initialization — no pretrained weights loaded.
Optimized for 24 GB VRAM + 64 GB RAM using PyTorch Native SDPA (No Flash Attention).

v3 improvements:
  1. Dataset: FineWeb-Edu (streaming) for true pre-training (common sense).
  2. Architecture: Grouped-Query Attention (GQA) & Sliding Window Attention (SWA).
  3. Context Length: 4096 (trained on 4096 packed chunks).
  4. Sequence packing with explicit BOS/EOS document boundaries.
"""

import os
import sys
import json
import math
import random
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm


# ── reproducibility ──────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── config ───────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # ── architecture (~350 M params, LLaMA-3 style) ──────────────
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4          # GQA: 4 KV heads for 16 Q heads
    intermediate_size: int = 2816         # SwiGLU sweet spot
    max_position_embeddings: int = 4096   
    sliding_window: int = 512             # SWA: Attend to 512 local tokens
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"

    # ── tokenizer ────────────────────────────────────────────────
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"

    # ── dataset ──────────────────────────────────────────────────
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"

    # ── training ─────────────────────────────────────────────────
    batch_size: int = 4                     # Reduced to 4 to accommodate SDPA overhead on 24GB
    gradient_accumulation_steps: int = 16   # Increased to maintain effective batch size
    learning_rate: float = 3e-4             
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000                
    num_epochs: int = 3                     
    max_steps: int = -1

    max_seq_length: int = 4096              

    # ── efficiency ───────────────────────────────────────────────
    use_flash_attention: bool = False       # Disabled Flash Attention
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True

    # ── output ───────────────────────────────────────────────────
    output_dir: str = "./fineweb_output"
    save_interval: int = 500
    logging_steps: int = 10
    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── packed dataset (FineWeb-Edu Streaming) ───────────────────────
class FineWebPackedDataset(Dataset):
    """
    Streams raw text from FineWeb-Edu, wraps each document in BOS/EOS tokens,
    and packs them into fixed-length sequences for zero-padding efficiency.
    """
    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        dataset_subset: str,
        max_length: int = 4096,
        split: str = "train",
        max_samples: int = 1_000_000,
        skip_samples: int = 0
    ):
        print(f"\nLoading {dataset_name} ({dataset_subset}) via streaming...")
        self.max_length = max_length

        ds = load_dataset(
            dataset_name, 
            name=dataset_subset, 
            split=split,
            streaming=True 
        )

        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id
        
        all_ids: list[int] = []
        n_docs = 0

        ds_iter = iter(ds)
        
        if skip_samples > 0:
            print(f"Skipping {skip_samples:,} documents (reserving for train/eval split)...")
            for _ in tqdm(range(skip_samples), desc="Skipping"):
                try:
                    next(ds_iter)
                except StopIteration:
                    break
        
        print(f"Fetching up to {max_samples:,} documents for packing...")
        
        for item in tqdm(ds_iter, total=max_samples, desc="Tokenizing Stream"):
            text = item.get("text", "")
            if not text or not text.strip():
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 0:
                continue

            # Explicit document boundaries
            all_ids.append(bos_id)
            all_ids.extend(ids)
            all_ids.append(eos_id)
            
            n_docs += 1
            if n_docs >= max_samples:
                break

        # Chunk into fixed-length sequences
        total_tokens = len(all_ids)
        n_chunks = total_tokens // max_length
        usable = n_chunks * max_length

        all_ids_t = torch.tensor(all_ids[:usable], dtype=torch.long)
        self.chunks = all_ids_t.view(n_chunks, max_length)

        avg_len = total_tokens / max(n_docs, 1)
        discarded = total_tokens - usable
        
        print(f"  {n_docs:,} documents fetched → {total_tokens:,} tokens")
        print(f"  Packed into {n_chunks:,} chunks of {max_length} tokens.")
        print(f"  Avg document: {avg_len:.0f} tok | Discarded tail: {discarded:,} tok | Padding: 0")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        input_ids = self.chunks[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),             
            "attention_mask": torch.ones_like(input_ids), 
        }


# ── weight initialisation ───────────────────────────────────────
def init_weights(model, std: float = 0.02):
    """Small-normal init (GPT-2 / LLaMA convention)."""
    num_layers = model.config.num_hidden_layers

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        if any(k in name for k in ("o_proj", "down_proj")):
            nn.init.normal_(param, mean=0.0, std=std / math.sqrt(2 * num_layers))
        else:
            nn.init.normal_(param, mean=0.0, std=std)

    print(f"  Initialised all weights (std={std}, residual scale=1/√{2 * num_layers})")


# ── model creation ───────────────────────────────────────────────
def create_model(config: TrainingConfig, vocab_size: int):
    print(f"\n{'─'*50}")
    print(f"  Building model FROM SCRATCH")
    print(f"{'─'*50}")
    print(f"  Architecture : LLaMA-style transformer")
    print(f"  Hidden       : {config.hidden_size}")
    print(f"  Layers       : {config.num_hidden_layers}")
    print(f"  Q heads      : {config.num_attention_heads}")
    print(f"  KV heads     : {config.num_key_value_heads} (GQA Enabled)")
    print(f"  Sliding Win  : {config.sliding_window} (SWA Enabled)")
    print(f"  MLP          : {config.intermediate_size}")
    print(f"  Vocab        : {vocab_size}")
    print(f"  Max position : {config.max_position_embeddings}")

    attn_impl = "flash_attention_2" if config.use_flash_attention else "sdpa"
    dtype = torch.bfloat16 if config.use_mixed_precision else torch.float32

    model_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        sliding_window=config.sliding_window,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=config.tie_word_embeddings,
        hidden_act=config.hidden_act,
        use_cache=True,
        attn_implementation=attn_impl,
    )

    try:
        model = LlamaForCausalLM(model_config).to(dtype)
    except Exception as e:
        print(f"  LlamaForCausalLM failed, falling back to AutoModel: {e}")
        model = AutoModelForCausalLM.from_config(model_config).to(dtype)

    init_weights(model, std=0.02)

    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ON")

    n = sum(p.numel() for p in model.parameters())
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params     : {n:>12,}  ({n/1e6:.1f} M)")
    print(f"  Trainable params : {t:>12,}  ({t/1e6:.1f} M)")

    return model


# ── optimiser / scheduler ────────────────────────────────────────
def create_optimizer_and_scheduler(model, config: TrainingConfig, num_training_steps: int):
    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight", "rmsnorm.weight"}
    groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


# ── training helpers ─────────────────────────────────────────────
def _optimizer_step(model, optimizer, scheduler, scaler, config):
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def train_epoch(model, tokenizer, train_loader, optimizer, scheduler, scaler, config: TrainingConfig, epoch: int):
    model.train()
    total_loss = 0.0
    micro_step = 0
    optim_step = 0

    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in pbar:
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=config.use_mixed_precision,
        ):
            loss = (
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss
                / config.gradient_accumulation_steps
            )

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        cur_loss = loss.item() * config.gradient_accumulation_steps
        total_loss += cur_loss
        micro_step += 1

        if micro_step % config.gradient_accumulation_steps == 0:
            _optimizer_step(model, optimizer, scheduler, scaler, config)
            optim_step += 1

            pbar.set_postfix(
                loss=f"{cur_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                ppl=f"{math.exp(min(cur_loss, 20)):.1f}",
                step=optim_step,
            )

            if 0 < config.max_steps <= optim_step:
                break

            if optim_step % config.save_interval == 0 and optim_step > 0:
                save_checkpoint(model, tokenizer, optimizer, scheduler, config, epoch, optim_step)

    # flush remaining accumulated gradients
    if micro_step % config.gradient_accumulation_steps != 0:
        _optimizer_step(model, optimizer, scheduler, scaler, config)

    return total_loss / max(micro_step, 1)


# ── evaluation ───────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, eval_loader, config: TrainingConfig):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(eval_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=config.use_mixed_precision,
        ):
            total_loss += model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss.item()

    avg = total_loss / max(len(eval_loader), 1)
    ppl = math.exp(min(avg, 100))
    return avg, ppl


# ── generation sanity check ──────────────────────────────────────
@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"  Prompt:    {prompt}")
    print(f"  Generated: {generated}")
    print(f"{'='*60}\n")


# ── checkpointing ───────────────────────────────────────────────
def save_checkpoint(model, tokenizer, optimizer, scheduler, config: TrainingConfig, epoch: int, step: int):
    ckpt = os.path.join(config.output_dir, f"checkpoint-epoch{epoch}-step{step}")
    os.makedirs(ckpt, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        os.path.join(ckpt, "training_state.pt"),
    )
    print(f"  Checkpoint saved → {ckpt}")


# ── main ─────────────────────────────────────────────────────────
def main(max_samples: int = None, config_path: str = None):
    config = TrainingConfig()

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            for k, v in json.load(f).items():
                if hasattr(config, k):
                    setattr(config, k, v)

    set_seed(config.seed)

    print("=" * 60)
    print("  PRE-TRAINING A LANGUAGE MODEL FROM SCRATCH (v3)")
    print("  — ~350 M params, GQA, Sliding Window, FineWeb-Edu —")
    print("=" * 60)

    print(f"\n  Device : {config.device}")
    if config.device == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {vram:.1f} GB")

    os.makedirs(config.output_dir, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────
    print("\n=== Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"  {config.tokenizer_name}  (vocab_size={vocab_size})")

    # ── Packed Datasets ──────────────────────────────────────────
    print("\n=== Datasets (streaming, packed, zero padding) ===")
    
    train_samples = max_samples if max_samples else 1_000_000 
    eval_samples = max(1, train_samples // 10)

    train_ds = FineWebPackedDataset(
        tokenizer, config.dataset_name, config.dataset_subset, 
        config.max_seq_length, split="train", max_samples=train_samples, skip_samples=0
    )
    
    eval_ds = FineWebPackedDataset(
        tokenizer, config.dataset_name, config.dataset_subset, 
        config.max_seq_length, split="train", max_samples=eval_samples, skip_samples=train_samples
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model (FROM SCRATCH) ─────────────────────────────────────
    print("\n=== Model ===")
    model = create_model(config, vocab_size).to(config.device)

    # ── Optimiser / Scheduler ────────────────────────────────────
    num_opt_steps = (math.ceil(len(train_loader) / config.gradient_accumulation_steps) * config.num_epochs)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_opt_steps)
    scaler = None  # bf16 doesn't need GradScaler

    eff_bs = config.batch_size * config.gradient_accumulation_steps
    tokens_per_step = eff_bs * config.max_seq_length
    total_tokens = len(train_ds) * config.max_seq_length * config.num_epochs

    print(f"\n=== Training Plan ===")
    print(f"  Optimiser steps   : {num_opt_steps:,}")
    print(f"  Epochs            : {config.num_epochs}")
    print(f"  Micro-batch       : {config.batch_size}")
    print(f"  Grad accumulation : {config.gradient_accumulation_steps}")
    print(f"  Effective batch   : {eff_bs}  ({tokens_per_step:,} tok/step)")
    print(f"  Sequence length   : {config.max_seq_length}")
    print(f"  ~Total tokens     : {total_tokens:,}  ({total_tokens/1e6:.1f} M)")
    print(f"  Learning rate     : {config.learning_rate}")
    print(f"  Warmup steps      : {config.warmup_steps}")

    # ── sanity generation (Raw Text Completion) ──────────────────
    print("\n--- Generation BEFORE training (expect random garbage) ---")
    generate_sample(model, tokenizer, "The history of the Roman Empire is")

    # ── training loop ────────────────────────────────────────────
    best_eval_loss = float("inf")

    for epoch in range(config.num_epochs):
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch + 1} / {config.num_epochs}")
        print(f"{'='*60}")

        train_loss = train_epoch(model, tokenizer, train_loader, optimizer, scheduler, scaler, config, epoch)
        train_ppl = math.exp(min(train_loss, 100))
        print(f"\n  Train loss : {train_loss:.4f}   ppl : {train_ppl:.2f}")

        eval_loss, eval_ppl = evaluate(model, eval_loader, config)
        print(f"  Eval  loss : {eval_loss:.4f}   ppl : {eval_ppl:.2f}")

        # ── sample generations (Raw Text Completion) ──
        print(f"\n--- Generation after epoch {epoch + 1} ---")
        generate_sample(model, tokenizer, "Machine learning is a field of study")

        # ── save best ──
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_dir = os.path.join(config.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "training_config.json"), "w") as f:
                json.dump(vars(config), f, indent=2)
            print(f"  ★ New best model (loss {best_eval_loss:.4f}) → {best_dir}")

        save_checkpoint(model, tokenizer, optimizer, scheduler, config, epoch, num_opt_steps)

    # ── save final model ─────────────────────────────────────────
    final_dir = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Final model → {final_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main(max_samples=1_000_000, config_path=None)
    else:
        parser = argparse.ArgumentParser(description="Pre-train ~350M LM on FineWeb-Edu")
        parser.add_argument("--config", type=str, default=None)
        parser.add_argument("--max-samples", type=int, default=1_000_000, help="Cap fetched documents to prevent OOM")
        cli_args = parser.parse_args()
        main(max_samples=cli_args.max_samples, config_path=cli_args.config)
