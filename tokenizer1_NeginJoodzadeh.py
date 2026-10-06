from collections import Counter
from typing import List, Dict, Tuple, Any, Optional


class CharTokenizer:
    """A simple character-level tokenizer mapping ASCII/Unicode characters to integer values."""

    def encode(self, text: str) -> List[int]:
        """Convert an input string into a list of integer character codes.

        Args:
            text (str): Input text to encode.

        Returns:
            List[int]: List of character integer codes.
        """
        return [ord(char) for char in text]

    def decode(self, tokens: List[int]) -> str:
        """Convert a list of integer character codes back into a string.

        Args:
            tokens (List[int]): List of integer character codes.

        Returns:
            str: Decoded text string.
        """
        return "".join(chr(token) for token in tokens)


class BPETokenizer:
    """Byte-Pair Encoding (BPE) Tokenizer implementation starting from 256 base byte tokens."""

    def __init__(self) -> None:
        """Initialize vocabulary and merge rules storage."""
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}

    def _get_pairs(self, tokens: List[int]) -> Counter:
        """Count frequencies of adjacent token pairs in a sequence.

        Args:
            tokens (List[int]): List of current integer tokens. Shape: (seq_len,)

        Returns:
            Counter: Mapping of (token_a, token_b) tuples to their occurrence counts.
        """
        # TODO: Count frequency of adjacent pairs across the token sequence
        pairs = Counter()

        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1

        return pairs



    def _merge_pair(
        self, tokens: List[int], pair: Tuple[int, int], new_token: int
    ) -> List[int]:
        """Replace non-overlapping occurrences of a target adjacent pair with a new merged token ID.

        Args:
            tokens (List[int]): Current token sequence. Shape: (seq_len,)
            pair (Tuple[int, int]): Target pair of token IDs to replace.
            new_token (int): New token ID to insert for the target pair.

        Returns:
            List[int]: Updated sequence with non-overlapping target pair merged.
        """
        # TODO: Iterate through sequence and substitute target pair occurrences with new token without overlapping
        merged = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                        merged.append(new_token)
                        i += 2
            else:
                        merged.append(tokens[i])
                        i += 1

        return merged

    

    def train(self, text: str, num_merges: int) -> "BPETokenizer":
        """Train BPE vocabulary starting from 256 base bytes and learn merge rules from text.

        Args:
            text (str): Raw text corpus for training.
            num_merges (int): Total number of BPE merge operations to execute.

        Returns:
            BPETokenizer: Self instance after training.
        """
        # TODO: Initialize 256 base byte tokens in vocab, compute pair frequencies, and iteratively merge top pair
        self.merges = {}

        self.vocab = {i: bytes([i]) for i in range(256)}

        tokens = list(text.encode("utf-8"))
        next_token_id = 256

        for _ in range(num_merges):
            pairs = self._get_pairs(tokens)

            if not pairs:
                break
            

            best_pair, _ = pairs.most_common(1)[0]

            tokens = self._merge_pair(
                tokens,
                best_pair,
                next_token_id
            )
      

            left_bytes = self.vocab[best_pair[0]]
            right_bytes = self.vocab[best_pair[1]]

            self.vocab[next_token_id] = left_bytes + right_bytes
            self.merges[best_pair] = next_token_id

            next_token_id += 1

        return self

    def encode(self, text: str) -> List[int]:
        """Encode string text into BPE token IDs using learned merge rules.

        Args:
            text (str): Input text string.

        Returns:
            List[int]: Encoded list of BPE token IDs.
        """
        # TODO: Convert text into raw UTF-8 bytes and iteratively apply learned merge rules in order
        tokens = list(text.encode("utf-8"))

        for pair, new_token in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_token)
        

        return tokens

    

    def decode(self, tokens: List[int]) -> str:
        """Decode a list of BPE token IDs back into a text string.

        Args:
            tokens (List[int]): List of BPE token IDs.

        Returns:
            str: Reconstructed text string.
        """
        # TODO: Lookup byte sequences for tokens, concatenate them, and decode UTF-8 bytes into text
        byte_sequence = b"".join(self.vocab[token] for token in tokens)

        return byte_sequence.decode("utf-8")

    

    def vocab_size(self) -> int:
        """Get current vocabulary size.

        Returns:
            int: Total number of unique tokens in the vocabulary.
        """
        # TODO: Return total number of items in vocabulary
        return len(self.vocab)
    

    def token_to_str(self, token_id: int) -> str:
        """Convert a single token ID into its string representation for visualization.

        Args:
            token_id (int): Token ID to inspect.

        Returns:
            str: String representation of the token bytes.
        """
        # TODO: Fetch byte mapping for token ID and decode into printable string format
        if token_id not in self.vocab:
            return "<UNK>"
        

        return self.vocab[token_id].decode("utf-8", errors="replace")
    


def compression_ratio(tokenizer: Any, text: str) -> float:
    """Calculate token compression ratio relative to raw UTF-8 byte count.

    Handle empty strings gracefully by returning 0.0 when byte length is zero.

    Args:
        tokenizer (Any): Tokenizer instance providing encode().
        text (str): Input text string.

    Returns:
        float: Ratio of encoded token count to raw byte length (0.0 if empty).
    """
    # TODO: Calculate encoded token count relative to raw UTF-8 byte length, returning 0.0 for empty input
    byte_length = len(text.encode("utf-8"))

    if byte_length == 0:
        return 0.0

    return len(tokenizer.encode(text)) / byte_length



def vocabulary_stats(tokenizer: Any, texts: List[str]) -> None:
    """Compute and display usage statistics across a list of text corpora.

    Args:
        tokenizer (Any): Tokenizer instance providing encode(), vocab_size(), and token_to_str().
        texts (List[str]): List of text strings to analyze.

    Returns:
        None
    """
    # TODO: Compute token frequency metrics, average token lengths per word, and output summary stats
    if not texts:
        print("No texts provided.")
        return

    token_counts = Counter()
    total_tokens = 0
    total_words = 0
    total_token_bytes = 0

    for text in texts:
        tokens = tokenizer.encode(text)

        token_counts.update(tokens)
        total_tokens += len(tokens)

        words = text.split()
        total_words += len(words)

        for word in words:
            word_tokens = tokenizer.encode(word)
            total_token_bytes += len(word_tokens)

    total_bytes = sum(
        len(text.encode("utf-8"))
        for text in texts
    )

    avg_tokens_per_word = (
        total_token_bytes / total_words
        if total_words
        else 0.0
    )

    avg_tokens_per_byte = (
        total_tokens / total_bytes
        if total_bytes
        else 0.0
    )

    print(f"  Vocabulary size: {tokenizer.vocab_size()}")
    print(f"  Total texts: {len(texts)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total UTF-8 bytes: {total_bytes}")
    print(f"  Average tokens per word: {avg_tokens_per_word:.2f}")
    print(f"  Token/byte ratio: {avg_tokens_per_byte:.2f}")

    print("  Most frequent tokens:")

    for token_id, count in token_counts.most_common(10):
        print(
            f"    {token_id}: {count} "
            f"-> {tokenizer.token_to_str(token_id)!r}"
        )


# [KEEP_IMPLEMENTATION]
def demo_char_tokenizer() -> None:
    """Demonstrate basic character-level tokenizer operations."""
    print("=" * 60)
    print("STEP 1: Character-Level Tokenizer")
    print("=" * 60)

    ct = CharTokenizer()

    texts = ["hello", "Hello, world!", "GPT-4"]
    for text in texts:
        encoded = ct.encode(text)
        decoded = ct.decode(encoded)
        print(f"  '{text}' -> {encoded}")
        print(f"  Roundtrip: {'PASS' if decoded == text else 'FAIL'}")
        print(f"  Tokens: {len(encoded)}")
        print()


# [KEEP_IMPLEMENTATION]
def demo_bpe_training() -> Tuple[BPETokenizer, str]:
    """Demonstrate BPE training process on a sample text corpus."""
    print("=" * 60)
    print("STEP 2: BPE Training")
    print("=" * 60)

    corpus = (
        "The cat sat on the mat. The cat ate the rat. "
        "The dog sat on the log. The dog ate the frog. "
        "Natural language processing is the study of how computers "
        "understand and generate human language. "
        "Tokenization is the first step in any NLP pipeline. "
        "Language models read tokens, not words. "
        "The tokenizer converts text into a sequence of integers. "
        "Each integer maps to a subword in the vocabulary."
    )

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, num_merges=50)

    print(f"\nVocabulary size after training: {tokenizer.vocab_size()}")
    print(f"Number of merges learned: {len(tokenizer.merges)}")

    return tokenizer, corpus


# [KEEP_IMPLEMENTATION]
def demo_encode_decode(tokenizer: BPETokenizer) -> None:
    """Demonstrate BPE encoding and decoding functionality across sentences."""
    print("\n" + "=" * 60)
    print("STEP 3: Encode and Decode")
    print("=" * 60)

    test_sentences = [
        "The cat sat on the mat.",
        "Natural language processing",
        "tokenization pipeline",
        "unhappiness",
        "The dog ate the frog.",
    ]

    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        raw_bytes = len(sentence.encode("utf-8"))
        ratio = len(encoded) / raw_bytes
        roundtrip = "PASS" if decoded == sentence else "FAIL"
        print(f"\n  '{sentence}'")
        print(f"  Encoded: {encoded[:15]}{'...' if len(encoded) > 15 else ''}")
        print(f"  Tokens: {len(encoded)} (from {raw_bytes} bytes)")
        print(f"  Compression ratio: {ratio:.2f}")
        print(f"  Roundtrip: {roundtrip}")


# [KEEP_IMPLEMENTATION]
def demo_tiktoken_comparison(tokenizer: BPETokenizer) -> None:
    """Compare custom BPE tokenizer outputs with OpenAI's tiktoken implementation."""
    print("\n" + "=" * 60)
    print("STEP 4: Compare with tiktoken")
    print("=" * 60)

    try:
        import tiktoken
    except ImportError:
        print("  tiktoken not installed. Run: pip install tiktoken")
        return

    enc = tiktoken.get_encoding("cl100k_base")

    texts = [
        "The cat sat on the mat.",
        "unhappiness",
        "Hello, world!",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "Geschwindigkeitsbegrenzung",
    ]

    for text in texts:
        our_tokens = tokenizer.encode(text)
        tk_tokens = enc.encode(text)
        tk_pieces = [enc.decode([t]) for t in tk_tokens]
        print(f"\n  '{text}'")
        print(f"  Our BPE:  {len(our_tokens)} tokens")
        print(f"  tiktoken: {len(tk_tokens)} tokens -> {tk_pieces}")
        ratio = len(our_tokens) / len(tk_tokens) if len(tk_tokens) > 0 else 0
        print(f"  Ours / tiktoken: {ratio:.1f}x")


# [KEEP_IMPLEMENTATION]
def demo_vocabulary_analysis(tokenizer: BPETokenizer, corpus: str) -> None:
    """Demonstrate vocabulary statistic evaluation and compression ratio checks."""
    print("\n" + "=" * 60)
    print("STEP 5: Vocabulary Analysis")
    print("=" * 60)

    test_texts = [
        corpus,
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is the most popular language for data science.",
    ]

    vocabulary_stats(tokenizer, test_texts)

    print(f"\nCompression ratios:")
    for text in test_texts[:3]:
        preview = text[:50] + "..." if len(text) > 50 else text
        ratio = compression_ratio(tokenizer, text)
        print(f"  {ratio:.2f} -- '{preview}'")


# ===== UNIT TESTS =====


def test_char_tokenizer() -> None:
    ct = CharTokenizer()

    # Standard functionality test
    sample = "Hello World!"
    encoded = ct.encode(sample)
    decoded = ct.decode(encoded)
    assert isinstance(encoded, list), "CharTokenizer.encode should return a list."
    assert (
        len(encoded) == len(sample)
    ), f"CharTokenizer output length mismatch. Expected {len(sample)}, got {len(encoded)}"
    assert (
        decoded == sample
    ), f"Roundtrip decoding failed. Expected '{sample}', got '{decoded}'"

    # Edge Case: Unicode Characters
    unicode_sample = "Hello 🌍!"
    u_encoded = ct.encode(unicode_sample)
    u_decoded = ct.decode(u_encoded)
    assert (
        u_decoded == unicode_sample
    ), f"Unicode decoding failed. Expected '{unicode_sample}', got '{u_decoded}'"

    # Edge Case: Empty String
    empty_encoded = ct.encode("")
    assert empty_encoded == [], "Encoding empty string should yield an empty list."
    assert ct.decode([]) == "", "Decoding empty list should yield an empty string."


def test_bpe_tokenizer_get_pairs() -> None:
    bpe = BPETokenizer()

    tokens = [1, 2, 1, 2, 3]
    pairs = bpe._get_pairs(tokens)
    assert pairs[(1, 2)] == 2, "Pair counting incorrect for pair (1, 2)."
    assert pairs[(2, 3)] == 1, "Pair counting incorrect for pair (2, 3)."

    # Edge Case: Single element / empty list
    assert (
        len(bpe._get_pairs([1])) == 0
    ), "Single element input should yield 0 pairs."
    assert len(bpe._get_pairs([])) == 0, "Empty list input should yield 0 pairs."


def test_bpe_tokenizer_merge_pair() -> None:
    bpe = BPETokenizer()

    tokens = [1, 2, 1, 2, 3]
    merged = bpe._merge_pair(tokens, (1, 2), 99)
    assert merged == [
        99,
        99,
        3,
    ], f"Merge output mismatch. Expected [99, 99, 3], got {merged}"

    # Non-overlapping merge edge case test
    overlapping_tokens = [1, 1, 1]
    non_overlap_merged = bpe._merge_pair(overlapping_tokens, (1, 1), 99)
    assert non_overlap_merged == [
        99,
        1,
    ], f"Merge must be non-overlapping. Expected [99, 1], got {non_overlap_merged}"

    # Edge Case: Pair not present / Sequence length 1
    assert bpe._merge_pair([1, 2, 3], (4, 5), 99) == [
        1,
        2,
        3,
    ], "Merging non-existent pair should return unchanged list."
    assert bpe._merge_pair([1], (1, 2), 99) == [
        1
    ], "Merging sequence of length 1 should return unchanged list."


def test_bpe_tokenizer_train_and_encode_decode() -> None:
    bpe = BPETokenizer()
    corpus = "aaabcaaab"

    bpe.train(corpus, num_merges=2)

    # Base byte requirement: Must initialize 256 base bytes + 2 merges = 258 vocab size
    assert (
        bpe.vocab_size() == 258
    ), f"BPE must start with 256 base bytes. Expected vocab_size 258 after 2 merges, got {bpe.vocab_size()}"
    assert (
        len(bpe.merges) == 2
    ), f"Learned merges count mismatch. Expected 2, got {len(bpe.merges)}"

    encoded = bpe.encode(corpus)
    decoded = bpe.decode(encoded)

    assert (
        len(encoded) < len(corpus.encode("utf-8"))
    ), "BPE compression failed to reduce token length."
    assert (
        decoded == corpus
    ), f"Roundtrip decoding mismatch. Expected '{corpus}', got '{decoded}'"

    # Edge Case: Unseen character at encoding
    unseen = "z"
    unseen_encoded = bpe.encode(unseen)
    assert bpe.decode(unseen_encoded) == unseen, "Failed to encode/decode unseen character correctly."


def test_bpe_tokenizer_token_to_str() -> None:
    bpe = BPETokenizer()
    bpe.vocab = {0: b"a", 256: b"ab"}

    res_a = bpe.token_to_str(0)
    res_ab = bpe.token_to_str(256)
    res_unknown = bpe.token_to_str(999)

    assert res_a == "a", f"Expected 'a', got '{res_a}'"
    assert res_ab == "ab", f"Expected 'ab', got '{res_ab}'"
    assert isinstance(res_unknown, str), "token_to_str should return string even for unknown token ID."


def test_compression_ratio() -> None:
    ct = CharTokenizer()
    text = "Test string"
    ratio = compression_ratio(ct, text)

    assert isinstance(ratio, float), "Compression ratio must return a float value."
    assert (
        0.0 <= ratio <= 2.0
    ), f"Compression ratio out of reasonable bounds: {ratio}"

    # Edge Case: Empty String (Division by zero check)
    empty_ratio = compression_ratio(ct, "")
    assert empty_ratio == 0.0, f"Expected 0.0 for empty string input, got {empty_ratio}"


def test_vocabulary_stats() -> None:
    class DummyTokenizer:
        def encode(self, text: str) -> List[int]:
            return [1, 2, 3]

        def vocab_size(self) -> int:
            return 10

        def token_to_str(self, token_id: int) -> str:
            return f"tok_{token_id}"

    dt = DummyTokenizer()
    # vocabulary_stats prints output and shouldn't crash
    try:
        vocabulary_stats(dt, ["hello world", "test corpus"])
    except Exception as e:
        assert False, f"vocabulary_stats raised an unexpected exception: {e}"


def run_all_tests() -> None:
    """Run hidden unit tests across student functions."""
    print("Running Unit Tests...")
    test_char_tokenizer()
    test_bpe_tokenizer_get_pairs()
    test_bpe_tokenizer_merge_pair()
    test_bpe_tokenizer_train_and_encode_decode()
    test_bpe_tokenizer_token_to_str()
    test_compression_ratio()
    test_vocabulary_stats()
    print("All unit tests passed successfully!\n")


def run_all_demos() -> None:
    """Run step-by-step demonstration functions."""
    demo_char_tokenizer()
    tokenizer, corpus = demo_bpe_training()
    demo_encode_decode(tokenizer)
    demo_tiktoken_comparison(tokenizer)
    demo_vocabulary_analysis(tokenizer, corpus)


if __name__ == "__main__":
    # Unit tests and pipeline demos are explicitly separated
    run_all_tests()
    run_all_demos()