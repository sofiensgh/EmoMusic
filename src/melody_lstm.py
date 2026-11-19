"""Mood-conditioned LSTM melody model (Chapter 3 - RNN).

At every time step the LSTM sees the previous note token AND a learned mood embedding,
so the same network can compose in 4 different moods.
"""
import torch
import torch.nn as nn

from .config import MOODS, VOCAB_SIZE


class MelodyLSTM(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, n_moods=len(MOODS), emb=128, mood_emb=16,
                 hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.hparams = dict(vocab=vocab, n_moods=n_moods, emb=emb, mood_emb=mood_emb,
                            hidden=hidden, layers=layers, dropout=dropout)
        self.tok = nn.Embedding(vocab, emb)
        self.mood = nn.Embedding(n_moods, mood_emb)
        self.lstm = nn.LSTM(emb + mood_emb, hidden, layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x, mood, state=None):
        # x: (B, T) tokens, mood: (B,) mood index
        m = self.mood(mood).unsqueeze(1).expand(-1, x.size(1), -1)
        h, state = self.lstm(torch.cat([self.tok(x), m], dim=-1), state)
        return self.head(self.drop(h)), state


@torch.no_grad()
def sample(model, mood_idx, seed_tokens, n_tokens, temperature=1.0, top_k=12, device="cpu", generator=None):
    """Autoregressive generation: feed the seed, then sample one token at a time."""
    model.eval()
    mood = torch.tensor([mood_idx], device=device)
    x = torch.tensor([seed_tokens], device=device)
    logits, state = model(x, mood)
    out = list(seed_tokens)
    for _ in range(n_tokens):
        lg = logits[0, -1] / max(temperature, 1e-3)
        if top_k:
            kth = torch.topk(lg, top_k).values[-1]
            lg = lg.masked_fill(lg < kth, float("-inf"))
        nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=generator).item()
        out.append(nxt)
        logits, state = model(torch.tensor([[nxt]], device=device), mood, state)
    return out


def load_melody_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = MelodyLSTM(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
