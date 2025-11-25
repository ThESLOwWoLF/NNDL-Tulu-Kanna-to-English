print(">>> bigguy.py STARTED (VS Code version)")

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import amp
from tqdm import tqdm
import sentencepiece as spm

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_DIR = os.path.join(BASE_DIR, "vocab_tf")
MODEL_DIR = os.path.join(BASE_DIR, "models_bigguy")

SRC_FILE = os.path.join(BASE_DIR, "train_kn_250k.txt")
TGT_FILE = os.path.join(BASE_DIR, "train_en_250k.txt")

# ---------- TRAINING CONFIG ----------
EPOCHS = 12
RESUME_FROM_CHECKPOINT = True
BATCH_SIZE = 8
MAX_SRC_LEN = 96
MAX_TGT_LEN = 96
USE_AMP = True
LABEL_SMOOTHING = 0.1
DROPOUT = 0.1
EARLY_STOP_PATIENCE = 3

# ---------- TRANSFORMER SIZE ----------
D_MODEL = 384
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 1536

# ---------- BLEU ----------
try:
    from sacrebleu.metrics import BLEU
    BLEU_METRIC = BLEU(effective_order=True)
    print("BLEU: ENABLED")
except Exception:
    BLEU_METRIC = None
    print("BLEU: DISABLED (install sacrebleu to enable)")


# -------------------- UTILITY --------------------
def load_pairs(max_lines=None):
    with open(SRC_FILE, "r", encoding="utf-8") as f1, open(TGT_FILE, "r", encoding="utf-8") as f2:
        src = [l.strip() for l in f1]
        tgt = [l.strip() for l in f2]
    pairs = [(s, t) for s, t in zip(src, tgt) if s and t]
    return pairs if max_lines is None else pairs[:max_lines]


def train_or_load_spm_models(src, tgt, src_vocab=8000, tgt_vocab=8000):
    os.makedirs(VOCAB_DIR, exist_ok=True)

    src_prefix = os.path.join(VOCAB_DIR, "spm_src")
    tgt_prefix = os.path.join(VOCAB_DIR, "spm_tgt")

    if not os.path.exists(src_prefix + ".model"):
        print("Training SPM SRC...")
        spm.SentencePieceTrainer.Train(
            input=src,
            model_prefix=src_prefix,
            vocab_size=src_vocab,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )

    if not os.path.exists(tgt_prefix + ".model"):
        print("Training SPM TGT...")
        spm.SentencePieceTrainer.Train(
            input=tgt,
            model_prefix=tgt_prefix,
            vocab_size=tgt_vocab,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )

    sp_src = spm.SentencePieceProcessor(model_file=src_prefix + ".model")
    sp_tgt = spm.SentencePieceProcessor(model_file=tgt_prefix + ".model")

    print("Loaded SentencePiece")
    print(" SRC vocab =", sp_src.vocab_size())
    print(" TGT vocab =", sp_tgt.vocab_size())

    return sp_src, sp_tgt


def ids_to_text(ids, sp, pad, bos, eos):
    out = []
    for i in ids:
        if i == pad or i == bos:
            continue
        if i == eos:
            break
        out.append(i)
    return sp.decode(out) if out else ""


# -------------------- DATASET --------------------
class TranslationDS(Dataset):
    def __init__(self, pairs, sp_src, sp_tgt):
        self.pairs = pairs
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s, t = self.pairs[idx]

        src = self.sp_src.encode(s, out_type=int)[:MAX_SRC_LEN - 1]
        tgt_core = self.sp_tgt.encode(t, out_type=int)[:MAX_TGT_LEN - 2]

        src = src + [self.sp_src.eos_id()]
        tgt = [self.sp_tgt.bos_id()] + tgt_core + [self.sp_tgt.eos_id()]

        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    max_s = max(len(x) for x in srcs)
    max_t = max(len(x) for x in tgts)

    s_pad = torch.zeros(len(batch), max_s, dtype=torch.long)
    t_pad = torch.zeros(len(batch), max_t, dtype=torch.long)

    for i, s in enumerate(srcs):
        s_pad[i, :len(s)] = s
    for i, t in enumerate(tgts):
        t_pad[i, :len(t)] = t

    return s_pad, t_pad


# -------------------- MODEL --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class BigGuyTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, D_MODEL)
        self.tgt_emb = nn.Embedding(tgt_vocab, D_MODEL)
        self.pos = PositionalEncoding(D_MODEL)
        self.drop = nn.Dropout(DROPOUT)

        self.trans = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True
        )

        self.fc = nn.Linear(D_MODEL, tgt_vocab)

    def forward(self, src, tgt):
        device = src.device
        src = self.drop(self.pos(self.src_emb(src)))
        tgt = self.drop(self.pos(self.tgt_emb(tgt)))

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)

        out = self.trans(src, tgt, tgt_mask=tgt_mask)
        return self.fc(out)


# -------------------- TRAIN --------------------
def train():
    print("Loading dataset...")
    pairs = load_pairs()
    print("Loaded lines:", len(pairs))

    sp_src, sp_tgt = train_or_load_spm_models(SRC_FILE, TGT_FILE)

    dataset = TranslationDS(pairs, sp_src, sp_tgt)
    val_size = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, BATCH_SIZE, False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = BigGuyTransformer(sp_src.vocab_size(), sp_tgt.vocab_size()).to(device)
    pad_id = sp_tgt.pad_id()
    bos_id = sp_tgt.bos_id()
    eos_id = sp_tgt.eos_id()

    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=LABEL_SMOOTHING)
    scaler = amp.GradScaler("cuda", enabled=USE_AMP)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=2)

    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt = os.path.join(MODEL_DIR, "bigguy_ckpt.pth")
    best_path = os.path.join(MODEL_DIR, "bigguy_best.pth")

    best_loss = 9999.0
    patience = 0
    start_epoch = 1

    # Resume
    if RESUME_FROM_CHECKPOINT and os.path.exists(ckpt):
        print("Resuming from checkpoint…")
        data = torch.load(ckpt, map_location=device)
        model.load_state_dict(data["model"])
        opt.load_state_dict(data["opt"])
        scaler.load_state_dict(data["scaler"])
        start_epoch = data["epoch"] + 1
        best_loss = data["best"]
        patience = data["patience"]

    # -------------------- EPOCHS --------------------
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        t0 = time.time()

        # ---------- TRAIN ----------
        model.train()
        train_loss = 0.0
        train_tokens = 0
        train_correct = 0

        total_train_batches = len(train_loader)

        for batch_idx, (src, tgt) in enumerate(
            tqdm(train_loader, desc=f"Training {epoch}/{EPOCHS}", ncols=100)
        ):
            src, tgt = src.to(device), tgt.to(device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            opt.zero_grad(set_to_none=True)

            with amp.autocast("cuda", enabled=USE_AMP):
                out = model(src, tgt_in)
                vocab = out.size(-1)
                loss = loss_fn(out.reshape(-1, vocab), tgt_out.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            mask = tgt_out != pad_id
            tokens = mask.sum().item()

            preds = out.argmax(-1)
            correct = (preds[mask] == tgt_out[mask]).sum().item()

            train_loss += loss.item() * tokens
            train_correct += correct
            train_tokens += tokens

            # Percent inside this epoch
            percent = (batch_idx + 1) / total_train_batches * 100.0
            tqdm.write(f"Train epoch {epoch}: {percent:.1f}% | batch_loss={loss.item():.4f}", end="\r")

        train_loss /= train_tokens
        train_acc = train_correct / train_tokens

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        val_correct = 0

        refs, hyps = [], []
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for batch_idx, (src, tgt) in enumerate(
                tqdm(val_loader, desc=f"Validating {epoch}/{EPOCHS}", ncols=100)
            ):
                src, tgt = src.to(device), tgt.to(device)
                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                with amp.autocast("cuda", enabled=USE_AMP):
                    out = model(src, tgt_in)
                    vocab = out.size(-1)
                    loss = loss_fn(out.reshape(-1, vocab), tgt_out.reshape(-1))

                mask = tgt_out != pad_id
                tokens = mask.sum().item()
                correct = (out.argmax(-1)[mask] == tgt_out[mask]).sum().item()

                val_loss += loss.item() * tokens
                val_correct += correct
                val_tokens += tokens

                # Percent in validation
                percent_val = (batch_idx + 1) / total_val_batches * 100.0
                tqdm.write(f"Val epoch {epoch}: {percent_val:.1f}% | batch_loss={loss.item():.4f}", end="\r")

                if BLEU_METRIC:
                    for b in range(src.size(0)):
                        ref = ids_to_text(tgt[b].tolist(), sp_tgt, pad_id, bos_id, eos_id)
                        hyp = ids_to_text(out.argmax(-1)[b].tolist(), sp_tgt, pad_id, bos_id, eos_id)
                        if ref and hyp:
                            refs.append(ref)
                            hyps.append(hyp)

        val_loss /= val_tokens
        val_acc = val_correct / val_tokens

        bleu = None
        if BLEU_METRIC is not None and refs:
            bleu = BLEU_METRIC.corpus_score(hyps, [refs]).score
        bleu_str = f"{bleu:.2f}" if bleu is not None else "N/A"

        scheduler.step(val_loss)

        t = time.time() - t0
        print(
            f"\n[Epoch {epoch}] time={t/60:.1f}m LR={opt.param_groups[0]['lr']:.6f} "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc*100:.1f}% "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc*100:.1f}% BLEU={bleu_str}"
        )

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "best": best_loss,
            "patience": patience
        }, ckpt)

        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), best_path)
            print(">> Saved BEST model.")
        else:
            patience += 1
            print(f">> No improvement. Patience {patience}/{EARLY_STOP_PATIENCE}")
            if patience >= EARLY_STOP_PATIENCE:
                print(">> EARLY STOP TRIGGERED.")
                break

    print("\n🤖 bigguy training completed.")
    print("Best model:", best_path)


if __name__ == "__main__":
    train()
