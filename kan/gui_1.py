"""
AIM:
Use the trained Kannada → English Transformer model (SentencePiece BPE)
to translate input Kannada sentences into English using beam search decoding,
with a Gradio web UI.
"""

import os
import math
import torch
import torch.nn as nn
import sentencepiece as spm
import gradio as gr

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_DIR = os.path.join(BASE_DIR, "vocab_tf")
MODEL_DIR = os.path.join(BASE_DIR, "models_tf")

SRC_SPM_PATH = os.path.join(VOCAB_DIR, "spm_src.model")
TGT_SPM_PATH = os.path.join(VOCAB_DIR, "spm_tgt.model")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "kn_en_transformer_spm_best.pth")
CKPT_PATH = os.path.join(MODEL_DIR, "checkpoint_last.pth")


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
    # must match modelT.py
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


# ---------- GLOBALS (lazy-loaded) ----------
_model = None
_sp_src = None
_sp_tgt = None
_device = None


def load_model_and_sp():
    global _model, _sp_src, _sp_tgt, _device
    if _model is not None:
        return _model, _sp_src, _sp_tgt, _device

    if not os.path.exists(SRC_SPM_PATH) or not os.path.exists(TGT_SPM_PATH):
        raise FileNotFoundError("SentencePiece models not found in 'vocab_tf'. Train with modelT.py first.")

    _sp_src = spm.SentencePieceProcessor(model_file=SRC_SPM_PATH)
    _sp_tgt = spm.SentencePieceProcessor(model_file=TGT_SPM_PATH)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", _device)

    _model = TransformerMT(_sp_src.vocab_size(), _sp_tgt.vocab_size()).to(_device)

    if os.path.exists(BEST_MODEL_PATH):
        print("Loading best model from:", BEST_MODEL_PATH)
        state = torch.load(BEST_MODEL_PATH, map_location=_device)
        _model.load_state_dict(state)
    elif os.path.exists(CKPT_PATH):
        print("Loading last checkpoint from:", CKPT_PATH)
        ckpt = torch.load(CKPT_PATH, map_location=_device)
        _model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise FileNotFoundError("No trained model found in 'models_tf'. Run modelT.py first.")

    _model.eval()
    return _model, _sp_src, _sp_tgt, _device


@torch.no_grad()
def translate_beam(model, sp_src, sp_tgt, device, kn_text, beam_size=4, max_len=80, len_penalty=0.7):
    """
    Beam search decoding for better translations.
    len_penalty < 1.0 favours slightly longer outputs, >1.0 shorter.
    """
    kn_text = kn_text.strip()
    if not kn_text:
        return ""

    # Encode source
    src_ids = sp_src.encode(kn_text, out_type=int)
    src_ids = src_ids + [sp_src.eos_id()]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, src_len)

    bos_id = sp_tgt.bos_id()
    eos_id = sp_tgt.eos_id()

    # each beam: (tokens_list, log_prob)
    beams = [([bos_id], 0.0)]
    completed = []

    for step in range(max_len):
        new_beams = []
        for tokens, score in beams:
            if tokens[-1] == eos_id:
                completed.append((tokens, score))
                continue

            tgt = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, cur_len)
            out = model(src, tgt)  # (1, cur_len, vocab)
            logits = out[0, -1]    # last step: (vocab,)
            log_probs = torch.log_softmax(logits, dim=-1)

            topk_log_probs, topk_idx = torch.topk(log_probs, beam_size)

            for lp, idx in zip(topk_log_probs.tolist(), topk_idx.tolist()):
                new_tokens = tokens + [idx]
                new_score = score + lp
                new_beams.append((new_tokens, new_score))

        if not new_beams:
            break

        # sort by score (higher better)
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    completed.extend(beams)

    # length penalty
    def lp(seq_len):
        return ((5 + seq_len) / 6) ** len_penalty

    scored = []
    for tokens, score in completed:
        # ignore initial BOS for length
        seq_len = max(1, len(tokens) - 1)
        scored.append((tokens, score / lp(seq_len)))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_tokens, best_score = scored[0]

    # remove BOS, cut at EOS
    bos = bos_id
    eos = eos_id
    final_ids = []
    for t in best_tokens:
        if t == bos:
            continue
        if t == eos:
            break
        final_ids.append(t)

    en = sp_tgt.decode(final_ids) if final_ids else ""
    return en.strip()


def gradio_translate(kn_text, beam_size, max_len, len_penalty):
    model, sp_src, sp_tgt, device = load_model_and_sp()
    try:
        en = translate_beam(
            model, sp_src, sp_tgt, device,
            kn_text,
            beam_size=int(beam_size),
            max_len=int(max_len),
            len_penalty=float(len_penalty),
        )
    except Exception as e:
        return f"Error during translation: {e}"
    return en


def main():
    with gr.Blocks(title="Kannada → English Translator") as demo:
        gr.Markdown(
            """
            # Kannada → English Translator (Transformer + SentencePiece)

            - Type a sentence in **Kannada**.
            - Model uses **subword BPE** and **beam search** for better translations.
            - You can tweak beam size, max length, and length penalty.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                kn_input = gr.Textbox(
                    label="Kannada input",
                    placeholder="Type Kannada text here...",
                    lines=4,
                )
                beam_slider = gr.Slider(
                    minimum=1, maximum=8, value=4, step=1,
                    label="Beam size"
                )
                maxlen_slider = gr.Slider(
                    minimum=20, maximum=120, value=80, step=5,
                    label="Max output tokens"
                )
                lenp_slider = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Length penalty"
                )
                btn = gr.Button("Translate")

            with gr.Column(scale=1):
                en_output = gr.Textbox(
                    label="English translation",
                    lines=4,
                )

        btn.click(
            fn=gradio_translate,
            inputs=[kn_input, beam_slider, maxlen_slider, lenp_slider],
            outputs=en_output,
        )

    demo.launch()


if __name__ == "__main__":
    main()
