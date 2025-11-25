"""
AIM:
Train a subword-level (SentencePiece BPE) Kannada → English Transformer model
using train_kn_250k.txt and train_en_250k.txt, with checkpointing so training
can resume after interruption.
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_DIR = os.path.join(BASE_DIR, "vocab_tf")
MODEL_DIR = os.path.join(BASE_DIR, "models_tf")

# ---------- FIXED DATASET PATHS ----------
SRC_FILE = os.path.join(BASE_DIR, "train_kn_250k.txt")
TGT_FILE = os.path.join(BASE_DIR, "train_en_250k.txt")

# ---------- TRAINING CONFIG ----------
EPOCHS = 20
RESUME_FROM_CHECKPOINT = True  # set False if you want to restart from scratch
BATCH_SIZE = 16                # smaller batch to avoid OOM
MAX_SRC_LEN = 128              # truncate long Kannada sentences
MAX_TGT_LEN = 128              # truncate long English sentences

# ---------- SentencePiece ----------
import sentencepiece as spm


def load_pairs(max_lines=None):
    with open(SRC_FILE, "r", encoding="utf-8") as f1, open(TGT_FILE, "r", encoding="utf-8") as f2:
        src = [l.strip() for l in f1]
        tgt = [l.strip() for l in f2]

    pairs = [(s, t) for s, t in zip(src, tgt) if s and t]
    return pairs if max_lines is None else pairs[:max_lines]


def train_or_load_spm_models(src_path, tgt_path, src_vocab=8000, tgt_vocab=8000):
    os.makedirs(VOCAB_DIR, exist_ok=True)

    src_prefix = os.path.join(VOCAB_DIR, "spm_src")
    tgt_prefix = os.path.join(VOCAB_DIR, "spm_tgt")

    if not os.path.exists(src_prefix + ".model"):
        print("Training SentencePiece SRC model...")
        spm.SentencePieceTrainer.Train(
            input=src_path,
            model_prefix=src_prefix,
            vocab_size=src_vocab,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )

    if not os.path.exists(tgt_prefix + ".model"):
        print("Training SentencePiece TGT model...")
        spm.SentencePieceTrainer.Train(
            input=tgt_path,
            model_prefix=tgt_prefix,
            vocab_size=tgt_vocab,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )

    sp_src = spm.SentencePieceProcessor(model_file=src_prefix + ".model")
    sp_tgt = spm.SentencePieceProcessor(model_file=tgt_prefix + ".model")

    print("Loaded SP models")
    print(" SRC vocab =", sp_src.vocab_size())
    print(" TGT vocab =", sp_tgt.vocab_size())

    return sp_src, sp_tgt


class TranslationDS(Dataset):
    def __init__(self, pairs, sp_src, sp_tgt):
        self.pairs = pairs
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s, t = self.pairs[idx]

        # Encode
        src_ids = self.sp_src.encode(s, out_type=int)
        tgt_ids_core = self.sp_tgt.encode(t, out_type=int)

        # Truncate to max length (reserve 1 spot for EOS/BOS)
        src_ids = src_ids[: MAX_SRC_LEN - 1]
        tgt_ids_core = tgt_ids_core[: MAX_TGT_LEN - 2]  # because we add BOS and EOS

        # Add special tokens
        src_ids = src_ids + [self.sp_src.eos_id()]
        tgt_ids = [self.sp_tgt.bos_id()] + tgt_ids_core + [self.sp_tgt.eos_id()]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    max_s = max(len(s) for s in srcs)
    max_t = max(len(t) for t in tgts)

    src_pad = torch.zeros(len(batch), max_s, dtype=torch.long)
    tgt_pad = torch.zeros(len(batch), max_t, dtype=torch.long)

    for i, s in enumerate(srcs):
        src_pad[i, :len(s)] = s
    for i, t in enumerate(tgts):
        tgt_pad[i, :len(t)] = t

    return src_pad, tgt_pad


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerMT(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model)
        self.trans = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt):
        device = src.device
        src = self.pos(self.src_emb(src))
        tgt = self.pos(self.tgt_emb(tgt))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        out = self.trans(src, tgt, tgt_mask=tgt_mask)
        return self.fc(out)


def train():
    print("Using files:")
    print(" SRC =", SRC_FILE)
    print(" TGT =", TGT_FILE)

    pairs = load_pairs()
    print("Loaded pairs:", len(pairs))

    sp_src, sp_tgt = train_or_load_spm_models(SRC_FILE, TGT_FILE)

    ds = TranslationDS(pairs, sp_src, sp_tgt)
    val_size = int(0.1 * len(ds))
    tr, va = random_split(ds, [len(ds) - val_size, val_size])

    tr_loader = DataLoader(tr, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(va, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    model = TransformerMT(sp_src.vocab_size(), sp_tgt.vocab_size()).to(device)
    pad_id = sp_tgt.pad_id()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, "checkpoint_last.pth")
    best_model_path = os.path.join(MODEL_DIR, "kn_en_transformer_spm_best.pth")

    best = 1e9
    start_epoch = 1

    # ---------- RESUME FROM CHECKPOINT ----------
    if RESUME_FROM_CHECKPOINT and os.path.exists(ckpt_path):
        print("Resuming from checkpoint:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best = ckpt.get("best_val_loss", best)
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best:.4f}")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_tokens = 0

        for src, tgt in tr_loader:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            out = model(src, tgt_in)
            vocab = out.size(-1)

            loss = loss_fn(out.reshape(-1, vocab), tgt_out.reshape(-1))
            loss.backward()
            opt.step()

            mask = tgt_out != pad_id
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

        train_loss = total_loss / total_tokens

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0
        val_tok = 0

        with torch.no_grad():
            for src, tgt in va_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                out = model(src, tgt_in)
                vocab = out.size(-1)
                loss = loss_fn(out.reshape(-1, vocab), tgt_out.reshape(-1))

                mask = tgt_out != pad_id
                val_loss += loss.item() * mask.sum().item()
                val_tok += mask.sum().item()

        val_loss /= val_tok

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

        # ---------- SAVE LAST CHECKPOINT EVERY EPOCH ----------
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_val_loss": best,
            },
            ckpt_path,
        )

        # ---------- SAVE BEST MODEL ----------
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model.")

    print("Training done.")
    print("Last checkpoint saved at:", ckpt_path)
    print("Best model saved at:", best_model_path)


if __name__ == "__main__":
    train()
