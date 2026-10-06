import re
import unicodedata
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Union

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )


def pre_tokenize(text: str) -> List[str]:
    """
    Split input text into initial word/symbol chunks using the GPT-2 regex pattern.
    """
    return GPT2_PATTERN.findall(text)


def apply_merge(byte_seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """
    Replace consecutive occurrences of a specific pair of token IDs in a sequence with a new token ID.
    """
    result = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(byte_seq[i])
            i += 1
    return result


class SpecialTokenHandler:
    """Manages registration and regex-based splitting of special tokens during tokenization."""

    def __init__(self) -> None:
        self.special_tokens: Dict[str, int] = {}
        self.pattern: Union[re.Pattern, None] = None

    def add_token(self, token_str: str, token_id: int) -> None:
        """Register a special token and update the combined regular expression pattern."""
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(tok) for tok in self.special_tokens.keys()]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text: str) -> List[Tuple[str, bool]]:
        """Segment text into a list of tuples containing text chunks and boolean flags."""
        if not self.pattern:
            return [(text, False)] if text else []

        result = []
        last_end = 0
        for match in self.pattern.finditer(text):
            start, end = match.span()
            if start > last_end:
                result.append((text[last_end:start], False))
            result.append((match.group(), True))
            last_end = end
        if last_end < len(text):
            result.append((text[last_end:], False))
        return result if result else [(text, False)]


class ProductionTokenizer:
    """Byte-Pair Encoding (BPE) tokenizer supporting training, normalization, special tokens, encoding, and decoding."""

    def __init__(self) -> None:
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_handler: SpecialTokenHandler = SpecialTokenHandler()
        self.next_id: int = 256

    def normalize(self, text: str) -> str:
        """Normalize input text using Unicode NFKC normalization."""
        return unicodedata.normalize("NFKC", text)

    def train(self, text: str, num_merges: int) -> None:
        """Train the BPE tokenizer by identifying frequent adjacent pairs and iteratively merging them."""
        if not text:
            return
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        byte_seqs = [list(chunk.encode("utf-8")) for chunk in chunks]

        for _ in range(num_merges):
            pair_counts = Counter()
            for seq in byte_seqs:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            new_id = self.next_id
            self.next_id += 1
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            byte_seqs = [apply_merge(seq, best_pair, new_id) for seq in byte_seqs]

    def add_special_token(self, token_str: str) -> int:
        """Register a new special token into the tokenizer vocabulary and special token handler."""
        new_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, new_id)
        self.vocab[new_id] = token_str.encode("utf-8")
        return new_id

    def encode(self, text: str) -> List[int]:
        """Encode raw text into a sequence of vocabulary token IDs using trained merges and special tokens."""
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        result = []
        for part, is_special in parts:
            if is_special:
                result.append(self.special_handler.special_tokens[part])
            else:
                chunks = pre_tokenize(part)
                for chunk in chunks:
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    result.extend(byte_seq)
        return result

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back into a UTF-8 string."""
        byte_list = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_list.extend(self.vocab[token_id])
        return bytes(byte_list).decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        """Get current total size of vocabulary including base bytes, merges, and special tokens."""
        return len(self.vocab)

    def get_token_bytes(self, token_id: int) -> bytes:
        """Retrieve underlying byte representation of a given token ID."""
        return self.vocab.get(token_id, b"")

    def save(self, path: str) -> None:
        """Save tokenizer state (vocab, merges, special tokens) to a binary file using pickle."""
        with open(path, "wb") as f:
            pickle.dump({
                "vocab": self.vocab,
                "merges": self.merges,
                "next_id": self.next_id,
                "special_tokens": self.special_handler.special_tokens,
            }, f)

    @classmethod
    def load(cls, path: str) -> "ProductionTokenizer":
        """Load tokenizer state from a binary pickle file, restoring vocab, merges, and special tokens."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.merges = data["merges"]
        tok.next_id = data["next_id"]
        # Restore special tokens directly (without re-assigning new IDs)
        tok.special_handler.special_tokens = data["special_tokens"]
        if data["special_tokens"]:
            escaped = [re.escape(t) for t in data["special_tokens"].keys()]
            tok.special_handler.pattern = re.compile("|".join(escaped))
        return tok


# ===== Demo Functions (UNCHANGED) =====

def demo_byte_encoding() -> None:
    print("=" * 60)
    print("Byte-Level Encoding")
    print("=" * 60)
    texts = [
        ("English", "hello"),
        ("Chinese", "你好"),
        ("Japanese", "こんにちは"),
        ("Emoji", "🔥🌍"),
        ("Mixed", "hello你好🔥"),
        ("Code", "def f(x):"),
    ]
    for label, text in texts:
        b = list(text.encode("utf-8"))
        print(f"{label:10s}: {len(text):2d} chars -> {len(b):2d} bytes -> {b[:16]}{'...' if len(b) > 16 else ''}")


def demo_pre_tokenization() -> None:
    print("\n" + "=" * 60)
    print("Pre-Tokenization (GPT-2 Regex)")
    print("=" * 60)
    texts = [
        "Hello, world! Don't stop.",
        "def train(model, data):",
        "The price is $3.14 per unit.",
        "  multiple   spaces   here  ",
    ]
    for text in texts:
        chunks = pre_tokenize(text)
        print(f"\n'{text}'")
        print(f"  -> {chunks}")


def demo_full_tokenizer() -> None:
    print("\n" + "=" * 60)
    print("Training Production Tokenizer")
    print("=" * 60)
    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox runs through the forest. "
        "Machine learning models process natural language. "
        "Machine learning transforms how we build software. "
        "Deep learning models need large datasets to train. "
        "def train(model, data): return model.fit(data) "
        "def predict(model, x): return model(x) "
        "for i in range(100): print(i) "
    )
    tok = ProductionTokenizer()
    tok.train(corpus, num_merges=50)
    bos_id = tok.add_special_token("<|begin|>")
    eos_id = tok.add_special_token("<|end|>")
    user_id = tok.add_special_token("<|user|>")
    asst_id = tok.add_special_token("<|assistant|>")
    print(f"\nVocab size: {tok.vocab_size()}")
    print(f"Special tokens: <|begin|>={bos_id}, <|end|>={eos_id}, <|user|>={user_id}, <|assistant|>={asst_id}")
    print("\n" + "=" * 60)
    print("Encoding Tests")
    print("=" * 60)
    test_texts = [
        "The quick brown fox.",
        "你好世界 Hello World",
        "🔥🌍🚀",
        "def foo(x): return x + 1",
        "<|begin|><|user|>Hello<|end|>",
        "Machine learning is powerful.",
    ]
    for text in test_texts:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        raw_bytes = len(text.encode("utf-8"))
        print(f"\nInput:   {text}")
        print(f"IDs:     {ids[:20]}{'...' if len(ids) > 20 else ''}")
        print(f"Tokens:  {len(ids)} (from {raw_bytes} bytes, ratio: {len(ids)/raw_bytes:.2f})")
        print(f"Decoded: {decoded}")
        roundtrip = "PASS" if decoded == text else "FAIL"
        print(f"Round-trip: {roundtrip}")


def demo_save_load() -> None:
    """Demonstrate saving and loading a trained tokenizer."""
    print("\n" + "=" * 60)
    print("Save & Load Tokenizer")
    print("=" * 60)
    
    # Train a tokenizer
    corpus = "The quick brown fox jumps over the lazy dog. " * 5
    tok = ProductionTokenizer()
    tok.train(corpus, num_merges=30)
    bos_id = tok.add_special_token("<|begin|>")
    eos_id = tok.add_special_token("<|end|>")
    
    # Save to file
    save_path = "tokenizer_checkpoint.pkl"
    tok.save(save_path)
    print(f"Tokenizer saved to: {save_path}")
    print(f"Original vocab size: {tok.vocab_size()}")
    print(f"Original special tokens: {tok.special_handler.special_tokens}")
    
    # Load from file
    loaded_tok = ProductionTokenizer.load(save_path)
    print(f"\nLoaded vocab size: {loaded_tok.vocab_size()}")
    print(f"Loaded special tokens: {loaded_tok.special_handler.special_tokens}")
    
    # Verify round-trip consistency
    test_text = "<|begin|>The quick fox<|end|>"
    original_ids = tok.encode(test_text)
    loaded_ids = loaded_tok.encode(test_text)
    
    print(f"\nTest text: {test_text}")
    print(f"Original encode: {original_ids}")
    print(f"Loaded encode:   {loaded_ids}")
    print(f"IDs match: {'PASS' if original_ids == loaded_ids else 'FAIL'}")
    print(f"Vocab match: {'PASS' if tok.vocab == loaded_tok.vocab else 'FAIL'}")
    print(f"Merges match: {'PASS' if tok.merges == loaded_tok.merges else 'FAIL'}")
    
    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"\nCleaned up: {save_path}")


def demo_tiktoken_comparison() -> None:
    try:
        import tiktoken
    except ImportError:
        print("\ntiktoken not installed. Run: pip install tiktoken")
        return
    print("\n" + "=" * 60)
    print("Comparison with tiktoken (GPT-4)")
    print("=" * 60)
    enc = tiktoken.get_encoding("cl100k_base")
    test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"
    tokens = enc.encode(test_paragraph)
    pieces = [enc.decode([t]) for t in tokens]
    print(f"\nInput: {test_paragraph}")
    print(f"GPT-4 tokens ({len(tokens)}): {pieces}")
    languages = [
        ("English", "The quick brown fox jumps over the lazy dog."),
        ("Chinese", "快速的棕色狐狸跳过了懒狗。"),
        ("Japanese", "素早い茶色のキツネが怠け者の犬を飛び越えた。"),
        ("Korean", "빠른 갈색 여우가 게으른 개를 뛰어넘었다."),
        ("Code", "def quicksort(arr): return sorted(arr)"),
        ("Emoji", "🎉🎊🎈🎁🎂🎄🎃🎆🎇✨"),
    ]
    print(f"\n{'Language':<10} {'Chars':<6} {'Tokens':<7} {'Fertility':<10}")
    print("-" * 35)
    for label, text in languages:
        toks = enc.encode(text)
        words = len(text.split())
        fertility = len(toks) / max(words, 1)
        print(f"{label:<10} {len(text):<6} {len(toks):<7} {fertility:<10.2f}")


# ===== UNIT TESTS =====

def test_pre_tokenize() -> None:
    res = pre_tokenize("Hello world! 123 ")
    assert isinstance(res, list), "Pre-tokenize output must be a list of strings."
    assert len(res) > 0, "Pre-tokenize output should not be empty for non-empty text."
    assert "".join(res) == "Hello world! 123 ", "Concatenated pre-tokenized chunks must reconstruct original text."
    empty_res = pre_tokenize("")
    assert empty_res == [], "Edge Case Failed: Empty string must return an empty list."


def test_apply_merge() -> None:
    seq = [10, 20, 10, 20, 30]
    merged = apply_merge(seq, (10, 20), 100)
    assert isinstance(merged, list), "apply_merge must return a list."
    assert len(merged) == 3, f"Expected merged length of 3, got {len(merged)}."
    assert merged == [100, 100, 30], "Merged output values do not match expected replaced token IDs."
    single = apply_merge([10], (10, 20), 100)
    assert single == [10], "Edge Case Failed: Single element list should remain unchanged."


def test_special_token_handler() -> None:
    handler = SpecialTokenHandler()
    handler.add_token("<|end|>", 500)
    parts = handler.split_with_specials("Hello <|end|>World ")
    assert isinstance(parts, list), "split_with_specials must return a list."
    assert len(parts) == 3, f"Expected 3 parts after splitting, got {len(parts)}."
    assert parts[1] == ("<|end|>", True), "Special token tag flag or value is incorrect."
    no_specials = handler.split_with_specials("Plain text")
    assert no_specials == [("Plain text", False)], "Edge Case Failed: Plain text splitting mismatched."


def test_production_tokenizer_train() -> None:
    tok = ProductionTokenizer()
    initial_vocab_size = tok.vocab_size()
    tok.train("abc abc abc", num_merges=2)
    assert tok.vocab_size() > initial_vocab_size, "Vocabulary size should increase after training BPE merges."
    assert len(tok.merges) <= 2, "Number of recorded merges should not exceed requested num_merges."
    tok_empty = ProductionTokenizer()
    tok_empty.train("", num_merges=5)
    assert tok_empty.vocab_size() == 256, "Edge Case Failed: Vocabulary size should stay at 256 for empty training text."


def test_production_tokenizer_encode_decode() -> None:
    tok = ProductionTokenizer()
    tok.train("The quick brown fox jumps over the lazy dog.", num_merges=10)
    tok.add_special_token("<|special|>")
    text = "The quick fox <|special|>"
    encoded = tok.encode(text)
    assert isinstance(encoded, list), "Encoder output must be a list of integers."
    assert len(encoded) > 0, "Encoded token list should not be empty."
    assert all(isinstance(idx, int) for idx in encoded), "All token IDs in encoded list must be integers."
    decoded = tok.decode(encoded)
    assert isinstance(decoded, str), "Decoder output must be a string."
    assert decoded == text, f"Round-trip decoded text '{decoded}' does not match original text '{text}'."
    single_char_ids = tok.encode("A")
    assert len(single_char_ids) == 1, "Edge Case Failed: Single byte character should yield exactly 1 token ID."


def test_save_load() -> None:
    """Test save/load round-trip preserves tokenizer state."""
    import os
    tok = ProductionTokenizer()
    tok.train("hello world hello world", num_merges=5)
    bos_id = tok.add_special_token("<|bos|>")
    eos_id = tok.add_special_token("<|eos|>")
    
    save_path = "test_tokenizer.pkl"
    try:
        tok.save(save_path)
        loaded = ProductionTokenizer.load(save_path)
        
        assert loaded.vocab_size() == tok.vocab_size(), "Vocab size mismatch after load."
        assert loaded.merges == tok.merges, "Merges mismatch after load."
        assert loaded.next_id == tok.next_id, "next_id mismatch after load."
        assert loaded.special_handler.special_tokens == tok.special_handler.special_tokens, \
            "Special tokens mismatch after load."
        
        # Verify IDs are preserved exactly
        assert loaded.special_handler.special_tokens["<|bos|>"] == bos_id, \
            "Special token ID not preserved."
        assert loaded.special_handler.special_tokens["<|eos|>"] == eos_id, \
            "Special token ID not preserved."
        
        # Verify encoding produces same output
        test_text = "<|bos|>hello world<|eos|>"
        assert tok.encode(test_text) == loaded.encode(test_text), \
            "Encoding mismatch after load."
        assert tok.decode(tok.encode(test_text)) == loaded.decode(loaded.encode(test_text)), \
            "Decoding mismatch after load."
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == "__main__":
    print("Running Unit Tests...\n")
    test_pre_tokenize()
    print("✓ test_pre_tokenize passed")
    test_apply_merge()
    print("✓ test_apply_merge passed")
    test_special_token_handler()
    print("✓ test_special_token_handler passed")
    test_production_tokenizer_train()
    print("✓ test_production_tokenizer_train passed")
    test_production_tokenizer_encode_decode()
    print("✓ test_production_tokenizer_encode_decode passed")
    test_save_load()
    print("✓ test_save_load passed")
    print("\nAll tests passed!\n")
    
    demo_byte_encoding()
    demo_pre_tokenization()
    demo_full_tokenizer()
    demo_save_load()
    demo_tiktoken_comparison()