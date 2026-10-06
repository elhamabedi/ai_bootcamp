"""
Mini GPT — PyTorch implementation exercise.

ALLOWED PyTorch APIs:
    torch tensor operations, torch.nn.Parameter, torch.nn.Linear, torch.nn.Embedding,
    torch.nn.Module, torch.autograd, torch.optim.

BANNED PyTorch APIs (you must implement these yourself):
    torch.nn.LayerNorm, torch.nn.functional.layer_norm,
    torch.nn.MultiheadAttention, torch.nn.functional.scaled_dot_product_attention,
    torch.nn.Transformer / nn.TransformerEncoderLayer / nn.TransformerDecoderLayer,
    torch.nn.functional.softmax, torch.softmax, Tensor.softmax,
    torch.nn.functional.cross_entropy, torch.nn.functional.log_softmax,
    torch.nn.CrossEntropyLoss.

Gradients are handled entirely by autograd — you never write a backward pass. Every
forward pass you write must therefore stay differentiable: build outputs from tensor
operations on the inputs, and never call .detach(), .item(), .numpy(), or wrap
anything in torch.no_grad() except where a docstring explicitly says so.

All tensors are float32 unless stated otherwise, except `token_ids`, which is
torch.long. Do not hardcode dtypes or devices inside forward passes: derive them from
the incoming tensors, so the same code runs unchanged in float64.
"""

import torch
import torch.nn as nn
import os
import re

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        """
        Token and positional embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of embedding vectors.
            max_seq_len (int): Maximum sequence length supported by positional embeddings.

        Attributes:
            token_embed (nn.Embedding): Token embedding table. Weight shape: (vocab_size, embed_dim)
            pos_embed (nn.Embedding): Positional embedding table. Weight shape: (max_seq_len, embed_dim)
        """
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        """
        Computes combined token and positional embeddings for input token sequences.

        Args:
            token_ids (torch.Tensor): Token indices, dtype torch.long.
                Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Sum of token embeddings and the positional embeddings for
                positions 0..seq_len-1, broadcast across the batch.
                Shape: (batch_size, seq_len, embed_dim)
        """
        seq_len = token_ids.size(1)
        
        token_emb = self.token_embed(token_ids)
        
        positions = torch.arange(seq_len, dtype=torch.long, device=token_ids.device)
        pos_emb = self.pos_embed(positions).unsqueeze(0)
        
        return token_emb + pos_emb


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        """
        Layer Normalization across the feature dimension.

        Args:
            dim (int): Feature/embedding dimension to normalize.
            eps (float): Epsilon added to the variance for numerical stability.

        Attributes:
            gamma (nn.Parameter): Learnable scale, initialized to ones. Shape: (dim,)
            beta (nn.Parameter): Learnable shift, initialized to zeros. Shape: (dim,)
            eps (float): Stored epsilon value.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        """
        Normalizes the last dimension of the input tensor and applies scale and shift.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: (..., dim)

        Returns:
            torch.Tensor: Layer-normalized tensor with the same shape as the input.
                The mean and the biased variance are computed over the last axis only.
                Shape: (..., dim)
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Causal Multi-Head Attention module.

        Args:
            embed_dim (int): Total dimensionality of input and output features.
            num_heads (int): Number of parallel attention heads. Must divide embed_dim.

        Attributes:
            num_heads (int): Number of attention heads.
            head_dim (int): embed_dim // num_heads.
            W_q (nn.Linear): Query projection, no bias. Weight shape: (embed_dim, embed_dim)
            W_k (nn.Linear): Key projection, no bias. Weight shape: (embed_dim, embed_dim)
            W_v (nn.Linear): Value projection, no bias. Weight shape: (embed_dim, embed_dim)
            W_out (nn.Linear): Output projection, no bias. Weight shape: (embed_dim, embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)
        for layer in (self.W_q, self.W_k, self.W_v, self.W_out):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        """
        Multi-head projection, scaled dot-product attention with optional additive
        masking, and output projection. The softmax must be implemented by hand.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Additive attention mask, added to the
                attention scores before the softmax. Allowed positions hold 0.0, masked
                positions hold a large negative value (see `causal_mask`).
                Shape: (seq_len, seq_len) or broadcastable to
                (batch_size, num_heads, seq_len, seq_len). Defaults to None (no masking).

        Returns:
            torch.Tensor: Attention output after the heads are recombined and passed
                through W_out.
                Shape: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask

        scores = scores - scores.max(dim=-1, keepdim=True).values
        weights = torch.exp(scores)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.W_out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        """
        Position-wise Feed-Forward Network (MLP).

        Args:
            embed_dim (int): Model embedding feature dimension.
            ff_dim (int): Hidden feature dimension of the expansion layer.

        Attributes:
            fc1 (nn.Linear): Expansion layer. Weight shape: (ff_dim, embed_dim), bias: (ff_dim,)
            fc2 (nn.Linear): Contraction layer. Weight shape: (embed_dim, ff_dim), bias: (embed_dim,)
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        for layer in (self.fc1, self.fc2):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Two-layer feed-forward transformation with a ReLU activation in between.

        Args:
            x (torch.Tensor): Input hidden states.
                Shape: (..., embed_dim)

        Returns:
            torch.Tensor: Transformed features, projected up to ff_dim and back down.
                Shape: (..., embed_dim)
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        """
        Pre-LayerNorm Transformer block.

        Args:
            embed_dim (int): Embedding feature dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Intermediate feed-forward layer dimension.

        Attributes:
            ln1 (LayerNorm): Normalization applied before attention.
            attn (MultiHeadAttention): Causal self-attention sub-layer.
            ln2 (LayerNorm): Normalization applied before the feed-forward network.
            ffn (FeedForward): Feed-forward sub-layer.
        """
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        """
        Passes the input through the attention and feed-forward sub-layers, each with
        pre-normalization and a residual connection.

        Args:
            x (torch.Tensor): Input representation.
                Shape: (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Additive causal mask forwarded to attention.
                Shape: (seq_len, seq_len) or broadcastable. Defaults to None.

        Returns:
            torch.Tensor: Output representation with the same shape as the input.
                Shape: (batch_size, seq_len, embed_dim)
        """
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


def causal_mask(seq_len, dtype=torch.float32, device=None):
    """
    Builds the additive causal (autoregressive) attention mask.

    Args:
        seq_len (int): Sequence length.
        dtype (torch.dtype): Data type of the returned mask. Defaults to torch.float32.
        device (torch.device, optional): Device of the returned mask. Defaults to None (CPU).

    Returns:
        torch.Tensor: Square additive mask whose entry [i, j] is 0.0 when position i is
            allowed to attend to position j (j <= i) and a large negative value
            otherwise. Use torch.finfo(dtype).min rather than a hardcoded constant so
            the mask stays valid in float64.
            Shape: (seq_len, seq_len)
    """
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    upper_triangle = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1
    )
    mask = mask.masked_fill(upper_triangle, torch.finfo(dtype).min)
    return mask


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        """
        Full MiniGPT causal language model.

        Args:
            vocab_size (int): Size of the vocabulary. Defaults to 50257.
            embed_dim (int): Hidden dimension size. Defaults to 768.
            num_heads (int): Number of attention heads. Defaults to 12.
            num_layers (int): Number of stacked Transformer blocks. Defaults to 12.
            max_seq_len (int): Maximum sequence context length. Defaults to 1024.
            ff_dim (int): Expansion dimension for the feed-forward network. Defaults to 3072.

        Attributes:
            embedding (Embedding): Joint token and positional embedding layer.
            blocks (nn.ModuleList): Stack of `num_layers` TransformerBlock modules.
            ln_f (LayerNorm): Final normalization applied before the output projection.
            vocab_size (int): Stored vocabulary size.
            embed_dim (int): Stored embedding dimension.
            max_seq_len (int): Stored maximum sequence length.
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

    def forward(self, token_ids):
        """
        Forward pass converting token sequences into next-token logits.

        Args:
            token_ids (torch.Tensor): Batch of token index sequences, dtype torch.long.
                Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Unnormalized scores over the vocabulary (logits), produced by
                weight tying with the token embedding matrix. A causal mask built from
                `seq_len` must prevent every position from attending to later positions.
                Shape: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        x = self.embedding(token_ids)
        
        mask = causal_mask(seq_len, dtype=x.dtype, device=x.device)
        
        for block in self.blocks:
            x = block(x, mask=mask)
        
        x = self.ln_f(x)
        
        logits = torch.matmul(x, self.embedding.token_embed.weight.t())
        
        return logits

    def count_parameters(self):
        """
        Computes the total number of trainable parameters from the architecture itself.

        Count the token and positional embedding tables, the four attention projections,
        both feed-forward weights and biases, both LayerNorm gains and shifts of every
        block, and the final LayerNorm parameters — deriving each size from the
        hyper-parameters. Do NOT use self.parameters(); the test compares your result
        against it.

        Returns:
            int: Grand total parameter count across all components.
        """
        num_layers = len(self.blocks)
        ff_dim = self.blocks[0].ffn.fc1.out_features if num_layers else 0

        token_emb = self.vocab_size * self.embed_dim
        pos_emb = self.max_seq_len * self.embed_dim

        per_block_attn = 4 * self.embed_dim * self.embed_dim
        per_block_ff = 2 * self.embed_dim * ff_dim + self.embed_dim + ff_dim
        per_block_ln = 4 * self.embed_dim
        per_block = per_block_attn + per_block_ff + per_block_ln

        final_ln = 2 * self.embed_dim

        total = token_emb + pos_emb + num_layers * per_block + final_ln

        return total


def cross_entropy_loss(logits, targets):
    """
    Computes the average cross-entropy loss over a batch of sequences.

    Must be implemented with a numerically stable log-softmax written by hand, and must
    stay differentiable: the returned tensor is what `.backward()` is called on during
    training, so do not detach it or convert it to a Python float.

    Args:
        logits (torch.Tensor): Model output logits before softmax.
            Shape: (batch_size, seq_len, vocab_size)
        targets (torch.Tensor): Ground-truth target token indices, dtype torch.long.
            Shape: (batch_size, seq_len)

    Returns:
        torch.Tensor: Scalar (0-dimensional) loss tensor, averaged over all
            batch_size * seq_len positions.
            Shape: ()
    """
    logits_max = logits.max(dim=-1, keepdim=True).values
    logits_shifted = logits - logits_max
    log_sum_exp = torch.log(torch.exp(logits_shifted).sum(dim=-1, keepdim=True))
    log_probs = logits_shifted - log_sum_exp
    
    targets_expanded = targets.unsqueeze(-1)
    target_log_probs = log_probs.gather(dim=-1, index=targets_expanded).squeeze(-1)
    
    loss = -target_log_probs.mean()
    
    return loss


def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    """
    Autoregressively generates new tokens from a prompt using temperature sampling.

    Runs without gradient tracking. Sampling must use torch.multinomial, so that seeding
    with torch.manual_seed makes the output reproducible.

    Args:
        model (MiniGPT): The language model instance.
        prompt_tokens (list[int]): Initial prompt token IDs.
        max_new_tokens (int): Number of new tokens to generate. Defaults to 100.
        temperature (float): Divisor applied to the logits before the softmax. Lower
            values sharpen the distribution, higher values flatten it. Defaults to 0.8.

    Returns:
        list[int]: The prompt followed by the generated tokens, of total length
            len(prompt_tokens) + max_new_tokens. The context fed to the model at each
            step must be truncated to the model's maximum sequence length.
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = list(prompt_tokens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = tokens[-model.max_seq_len:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1, :] / temperature

            logits = logits - logits.max()
            probs = torch.exp(logits)
            probs = probs / probs.sum()

            next_token = int(torch.multinomial(probs, num_samples=1))
            tokens.append(next_token)

    return tokens


def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=1500, lr=3e-4, batch_size=4):
    """
    Runs an end-to-end training loop for MiniGPT on raw text.

    The text is encoded as UTF-8 bytes, so the vocabulary is the 256 possible byte
    values. Each step samples `batch_size` random windows of length seq_len + 1, uses
    the first seq_len bytes of each window as input and the last seq_len bytes as
    targets, computes the loss, backpropagates with autograd and updates every
    parameter with a torch.optim.AdamW optimizer. Print the loss every 20 steps so
    training is visible.

    Sanity check: with the defaults, the loss starts near ln(256) = 5.55 and should
    fall below 0.5 within roughly 1500 steps (about a minute on a CPU). If it plateaus
    above 3.0, something in your forward pass or loss is wrong. Note that the optimizer
    choice is part of the specification: plain SGD at this learning rate barely moves
    the loss at all.

    Args:
        text (str): Raw input text corpus used for training.
        vocab_size (int): Size of the byte vocabulary. Defaults to 256.
        embed_dim (int): Model embedding dimensionality. Defaults to 128.
        num_heads (int): Attention head count. Defaults to 4.
        num_layers (int): Transformer depth. Defaults to 4.
        seq_len (int): Training context length window, also used as the model's
            maximum sequence length. Defaults to 64.
        num_steps (int): Total gradient update steps. Defaults to 1500.
        lr (float): AdamW learning rate. Defaults to 3e-4.
        batch_size (int): Number of sequences sampled per step. Defaults to 4.

    Returns:
        MiniGPT: The trained model instance, left in eval mode.
    """
    data = torch.tensor(
        list(text.encode("utf-8")),
        dtype=torch.long
    )

    if len(data) <= seq_len:
        raise ValueError(
            f"Text is too short. Need more than {seq_len} bytes."
        )

    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len,
        ff_dim=4 * embed_dim
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    model.train()

    for step in range(num_steps):
        starts = torch.randint(
            0,
            len(data) - seq_len,
            (batch_size,)
        )

        inputs = torch.stack([
            data[start:start + seq_len]
            for start in starts
        ])

        targets = torch.stack([
            data[start + 1:start + seq_len + 1]
            for start in starts
        ])

        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")
    model.eval()
    return model


def parameter_breakdown():
    """
    Prints parameter counts for the standard GPT-2 configuration sizes.

    Returns:
        None
    """
    configs = [
        ("GPT-2 Small", 50257, 768, 12, 12, 1024, 3072),
        ("GPT-2 Medium", 50257, 1024, 16, 24, 1024, 4096),
        ("GPT-2 Large", 50257, 1280, 20, 36, 1024, 5120),
        ("GPT-2 XL", 50257, 1600, 25, 48, 1024, 6400),
    ]
    print("GPT-2 Family Parameter Counts")
    print("=" * 65)
    print(f"{'Model':<16} {'Layers':>6} {'Heads':>6} {'Dims':>6} {'Params':>14}")
    print("-" * 65)
    for name, vocab, dim, heads, layers, seq_len, ff in configs:
        token_emb = vocab * dim
        pos_emb = seq_len * dim
        per_block_attn = 4 * dim * dim
        per_block_ff = 2 * dim * ff + dim + ff
        per_block_ln = 4 * dim
        per_block = per_block_attn + per_block_ff + per_block_ln
        final_ln = 2 * dim
        total = token_emb + pos_emb + layers * per_block + final_ln
        print(f"{name:<16} {layers:>6} {heads:>6} {dim:>6} {total:>14,}")
    print()


def memory_estimate():
    """
    Prints theoretical FP16 inference memory consumption for several modern models.

    Returns:
        None
    """
    print("Memory Requirements for Inference (FP16)")
    print("=" * 65)
    models = [
        ("GPT-2 Small (124M)", 124e6, 12, 12, 64, 1024),
        ("Llama 3 8B", 8e9, 32, 32, 128, 8192),
        ("Llama 3 70B", 70e9, 80, 64, 128, 8192),
        ("Llama 3 405B", 405e9, 126, 128, 128, 131072),
    ]
    print(f"{'Model':<24} {'Weights':>10} {'KV Cache':>12} {'Total':>10}")
    print("-" * 65)
    for name, params, layers, heads, head_dim, max_seq in models:
        weight_bytes = params * 2
        kv_per_token = 2 * layers * heads * head_dim * 2
        kv_full = kv_per_token * max_seq
        total = weight_bytes + kv_full

        def fmt(b):
            if b >= 1e9:
                return f"{b / 1e9:.1f} GB"
            return f"{b / 1e6:.0f} MB"

        print(f"{name:<24} {fmt(weight_bytes):>10} {fmt(kv_full):>12} {fmt(total):>10}")
    print()


# ===== Adding the data_pipeline.py file and doctors' comments (comments_raw_canonical.csv) =====

DEFAULT_CSV_PATH = "comments_raw_canonical.csv"
DEFAULT_MAX_CHARS = 200_000
QUALITY_FILTER_KWARGS = {"min_words": 3, "max_ratio_caps": 0.3, "max_ratio_special": 0.3}
DEDUP_THRESHOLD = 0.8

# Arabic letter/digit variants -> their Persian equivalents.
_ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه",
    "٠": "۰", "١": "۱", "٢": "۲", "٣": "۳", "٤": "۴",
    "٥": "۵", "٦": "۶", "٧": "۷", "٨": "۸", "٩": "۹",
})

# Diacritics (harakat), superscript alef and kashida (tatweel).
_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
# Zero-width space/joiner, directional marks and BOM. The ZWNJ (\u200c) is
# intentionally kept because it is meaningful in Persian orthography.
_INVISIBLE_RE = re.compile(r"[\u200b\u200d\u200e\u200f\ufeff]")
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")
# Three or more repeats of the same letter ("خیلییییی") collapse to one letter.
# Digits and underscores are excluded, so numbers such as 1000000 are unchanged.
_REPEATED_LETTER_RE = re.compile(r"([^\W\d_])\1{2,}")
_REPEATED_PUNCT_RE = re.compile(r"([!؟?.،,])\1+")

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_persian(text: str) -> str:
    """
    Normalizes Persian text so that equivalent spellings map to the same bytes.

    Unifies Arabic/Persian letters and digits, strips diacritics, kashida,
    invisible characters and emojis, and collapses exaggerated repetitions
    of letters and punctuation marks.

    Args:
        text (str): Raw Persian text.

    Returns:
        str: Normalized text (whitespace is not collapsed here).
    """
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = _DIACRITICS_RE.sub("", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _REPEATED_LETTER_RE.sub(r"\1", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return text

def _clean_comment(text: str) -> str:
    """Removes markup, links and control characters, then normalizes Persian text."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = normalize_persian(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def load_simple_comments_corpus(csv_path: str = DEFAULT_CSV_PATH,
                                min_words: int = 5,
                                max_words: int = 12,
                                top_words: int = 2000,
                                min_rate: float = None,
                                max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """
    Builds a simple, readable corpus of short doctor-review comments for a small
    byte-level model, using the data pipeline (data_pipeline.py): quality filtering
    and near-duplicate removal.


    A small model spends its capacity on spelling rare words. This function removes
    that burden: it keeps only comments of moderate length whose every word is among
    the `top_words` most frequent words, so the model can focus on sentence patterns.


    The pipeline's own `clean_text` is not used here because it removes every
    non-ASCII character and would erase all Persian text. `_clean_comment`, a
    Persian-aware cleaning step, is applied instead.


    Args:
        csv_path (str): Path to the raw canonical comments CSV file.
        min_words (int): Minimum number of words per comment.
        max_words (int): Maximum number of words per comment.
        top_words (int): Size of the allowed vocabulary (most frequent words).
        min_rate (float, optional): Keep only comments rated at least this value.
        max_chars (int): Character budget; the corpus is cut at a comment boundary.


    Returns:
        str: Newline-separated simple comments for byte-level training.
    """
    import pandas as pd
    from collections import Counter
    from data_pipeline import quality_filter, deduplicate


    df = pd.read_csv(csv_path)
    if min_rate is not None:
        df = df[df["rate"] >= min_rate]


    docs, seen = [], set()
    for raw in df["text"].dropna().astype(str):
        text = _clean_comment(raw)
        if not (min_words <= len(text.split()) <= max_words) or text in seen:
            continue
        if quality_filter(text, **QUALITY_FILTER_KWARGS):
            seen.add(text)
            docs.append(text)


    counts = Counter(word for doc in docs for word in doc.split())
    allowed = {word for word, _ in counts.most_common(top_words)}
    docs = [doc for doc in docs if all(word in allowed for word in doc.split())]


    docs, removed = deduplicate(docs, threshold=DEDUP_THRESHOLD)
    corpus = "\n".join(docs)
    if len(corpus) > max_chars:
        corpus = corpus[:max_chars].rsplit("\n", 1)[0]
    print(f"Simple corpus: {len(corpus.splitlines())} comments, {len(allowed)} distinct words, {removed} near-duplicates removed")
    return corpus


if __name__ == "__main__":
    torch.manual_seed(42)
    parameter_breakdown()
    memory_estimate()
    corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""
    print("Training Mini GPT")
    print("=" * 65)
    model = train_mini_gpt(corpus, num_steps=1500)
    prompt = list("The transformer".encode("utf-8"))
    print(f"\nPrompt: 'The transformer'")
    print("Generating...")
    output_tokens = generate(model, prompt, max_new_tokens=100, temperature=0.6)
    generated_text = bytes(output_tokens).decode("utf-8", errors="replace")
    print(f"Generated: {generated_text}")

    # ===== train on doctor comments (data_pipeline.py and comments_raw_canonical.csv) =====

    if os.path.exists("comments_raw_canonical.csv") and os.path.exists("data_pipeline.py"):
        torch.manual_seed(42)
        print("\nExtra credit: Doctor comments (simple corpus)")
        print("=" * 65)
        simple_corpus = load_simple_comments_corpus(min_rate=4.0)
        simple_model = train_mini_gpt(simple_corpus,
                                      embed_dim=192,
                                      num_heads=6,
                                      num_layers=4,
                                      seq_len=128,
                                      batch_size=16,
                                      num_steps=3000,
                                      lr=1e-3)
        simple_prompt = list("دکتر".encode("utf-8"))
        print("\nPrompt: 'دکتر'")
        print("Generating...")
        simple_tokens = generate(simple_model, simple_prompt, max_new_tokens=200, temperature=0.4)
        simple_text = bytes(simple_tokens).decode("utf-8", errors="replace")
        print(f"Generated: {simple_text.split(chr(10))[0]}")