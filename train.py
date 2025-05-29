from argparse import ArgumentParser
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning_gpt import callbacks, models

import tiktoken   # GPT-2 BPE

# -------------------------------------------------------------------- #
#                       DATASET – random-access sampler                #
# -------------------------------------------------------------------- #
class OpenWebTextSampleDataset(Dataset):
    """
    Instead of iterating through the whole 9-billion-token corpus,
    we treat *each __getitem__ call* as:
        1. pick a random offset in train.bin
        2. return the token block (x,y)
    Thus:
        • __len__ is an arbitrary 'samples_per_epoch'
        • torch.randperm never sees the real corpus length
    """
    def __init__(self, data_dir: str, split: str,
                 block_size: int, samples_per_epoch: int = 1_000_000):
        path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found.  Run data/openwebtext/prepare.py first.")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.samples_per_epoch = samples_per_epoch
        self.max_start = len(self.data) - block_size - 1

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _):
        # choose a random starting point each time
        idx = np.random.randint(0, self.max_start, dtype=np.int64)
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# tiny val set can stay sequential; it’s only ~4 M tokens
class OpenWebTextValDataset(Dataset):
    def __init__(self, data_dir: str, block_size: int):
        path = os.path.join(data_dir, "val.bin")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) // block_size  # ≈ 140 k batches of 32 tokens

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# -------------------------------------------------------------------- #
#                               MAIN                                   #
# -------------------------------------------------------------------- #
def main(args):
    enc = tiktoken.get_encoding("gpt2")        # 50 257 tokens incl. <|endoftext|>

    # ------------ Data ------------
    train_dataset = OpenWebTextSampleDataset(
        args.data_dir, "train",
        args.block_size, samples_per_epoch=args.samples_per_epoch)
    val_dataset = OpenWebTextValDataset(args.data_dir, args.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,          # now only shuffles 1 M indices, not 9 B 🙂
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # ------------ Model selection ------------
    extra = {}
    if args.implementation == "mingpt":
        GPT = models.MinGPT
        extra.update(dict(embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1))
    elif args.implementation == "nanogpt":
        GPT = models.NanoGPT
        extra["dropout"] = 0.0
    else:
        raise ValueError(f"Unsupported implementation {args.implementation}")
    
    # after you pick GPT = models.MinGPT or models.NanoGPT …
    if args.strategy.startswith("fsdp"):
        if args.implementation == "nanogpt":
            GPT = models.FSDPNanoGPT
        else:
            raise ValueError("FSDP is only supported for NanoGPT in this codebase.")
            
    model = GPT(
        vocab_size=enc.n_vocab,
        block_size=args.block_size,
        model_type=args.model_type,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        weight_decay=0.1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        **extra,
    )

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"torch {torch.__version__} lacks compile(); install ≥ 2.0 or drop --compile")
        model = torch.compile(model)

    # ------------ Trainer ----------
    cb = []
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        cb.append(callbacks.CUDAMetricsCallback())

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",    # ← add this line
        callbacks=cb,
        accelerator="auto",
        devices="auto",
        strategy=args.strategy,
        precision="16-mixed",
        logger=WandbLogger(),      # keeps your WandB project unchanged
    )

    trainer.fit(model, train_loader, val_loader)

    # quick sanity-check generation
    ctx = "Friends of my soul"
    x = torch.tensor(enc.encode_ordinary(ctx), dtype=torch.long,
                     device=model.device)[None, :]
    y = model.generate(x, max_new_tokens=200, temperature=1.0,
                       do_sample=True, top_k=40)
    print(enc.decode(y[0].tolist()))

if __name__ == "__main__":
    L.seed_everything(42)

    p = ArgumentParser()
    p.add_argument("--strategy", default="fsdp", type=str)
    p.add_argument("--model_type", default="gpt2")
    p.add_argument("--n_layer", type=int, default=16)
    p.add_argument("--n_head", type=int, default=16)
    p.add_argument("--n_embd", type=int, default=1024)
    p.add_argument("--learning_rate", default=1e-3, type=float)
    p.add_argument("--max_epochs", default=10, type=int)

    p.add_argument("--data_dir", default="data/openwebtext")
    p.add_argument("--block_size", default=1024, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--num_workers", default=4, type=int)

    # NEW: how many random samples constitute *one epoch*
    p.add_argument("--samples_per_epoch", default=1_000_000, type=int)

    p.add_argument("--compile", default="dynamo", choices=[None, "dynamo"])
    p.add_argument("--implementation", default="nanogpt", choices=["mingpt", "nanogpt"])

    args = p.parse_args()
    main(args)