"""
AIM:
Use the trained Kannada → English Transformer model (SentencePiece BPE)
to translate input Kannada sentences into English using greedy decoding.
"""

import os
import math
import torch
import torch.nn as nn

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_DIR = os.path.join(BASE_DIR, "vocab_tf")
MODEL_DIR = os.path.join(BASE_DIR, "models_tf")

SRC_SPM_PATH = os.path.join(VOCAB_DIR, "spm_src.model")
TGT_SPM_PATH = os.path.join(VOCAB_DIR, "spm_tgt.model")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "kn_en_transformer_spm_best.pth")
CKPT_PATH = os.path.join(MODEL_DIR, "checkpoint_last.pth")

# ---------- SentencePiece ----------
import sentencepiece as spm


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
    """
    Same architecture as in modelT.py:
    d_model=256, nhead=4, num_layers=3
    """
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
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt):
        device = src.device
        src = self.pos(self.src_emb(src))
        tgt = self.pos(self.tgt_emb(tgt))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        out = self.trans(src, tgt, tgt_mask=tgt_mask)
        return self.fc(out)


def load_model_and_sp():
    if not os.path.exists(SRC_SPM_PATH) or not os.path.exists(TGT_SPM_PATH):
        raise FileNotFoundError("SentencePiece models not found in 'vocab_tf'. Train with modelT.py first.")

    sp_src = spm.SentencePieceProcessor(model_file=SRC_SPM_PATH)
    sp_tgt = spm.SentencePieceProcessor(model_file=TGT_SPM_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = TransformerMT(sp_src.vocab_size(), sp_tgt.vocab_size()).to(device)

    # Prefer best-model file, else fall back to checkpoint
    if os.path.exists(BEST_MODEL_PATH):
        print("Loading best model from:", BEST_MODEL_PATH)
        state = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
    elif os.path.exists(CKPT_PATH):
        print("Loading last checkpoint from:", CKPT_PATH)
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise FileNotFoundError("No trained model found in 'models_tf'. Run modelT.py first.")

    model.eval()
    return model, sp_src, sp_tgt, device


@torch.no_grad()
def translate_sentence(model, sp_src, sp_tgt, device, kn_text, max_len=80):
    """
    Simple greedy decoding:
    - Encode Kannada sentence with sp_src
    - Start target with BOS token
    - Step by step predict next token until EOS or max_len
    """
    # Encode source
    src_ids = sp_src.encode(kn_text, out_type=int)
    src_ids = src_ids + [sp_src.eos_id()]  # add EOS just like training

    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, src_len)

    # Prepare decoder start with BOS
    bos_id = sp_tgt.bos_id()
    eos_id = sp_tgt.eos_id()

    tgt_ids = [bos_id]  # growing target sequence (list of ids)

    for _ in range(max_len):
        tgt = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, cur_len)

        # Forward pass
        out = model(src, tgt)           # (1, cur_len, vocab)
        next_token_logits = out[0, -1]  # last time step
        next_id = int(next_token_logits.argmax().item())

        if next_id == eos_id:
            break

        tgt_ids.append(next_id)

    # Remove BOS before decoding to text
    decoded_ids = [i for i in tgt_ids if i != bos_id]
    en_text = sp_tgt.decode(decoded_ids) if decoded_ids else ""
    return en_text.strip()


def main():
    print("Loading model and SentencePiece vocab...")
    model, sp_src, sp_tgt, device = load_model_and_sp()

    print("\nReady for inference!")
    print("Type Kannada sentences and press Enter to translate.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            kn = input("KN > ").strip()
        except EOFError:
            break

        if not kn:
            continue
        if kn.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        en = translate_sentence(model, sp_src, sp_tgt, device, kn)
        print("EN >", en)
        print("-" * 40)


if __name__ == "__main__":
    main()
