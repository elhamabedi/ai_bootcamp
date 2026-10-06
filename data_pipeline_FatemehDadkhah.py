import re
import hashlib
import random
import time
from collections import Counter, defaultdict
from typing import List, Tuple, Set, Dict, Any, Generator, Optional


def clean_text(text: str) -> str:
    """
    Cleans raw document text by stripping HTML tags, URLs, non-ASCII characters,
    and normalizing whitespace.

    Args:
        text (str): Input raw text document.

    Returns:
        str: Cleaned and normalized text string.
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def quality_filter(
    text: str, 
    min_words: int = 50, 
    max_ratio_caps: float = 0.3, 
    max_ratio_special: float = 0.1
) -> bool:
    """
    Filters out low-quality documents based on length, capitalization ratio, and special character density.

    Args:
        text (str): Cleaned document text.
        min_words (int): Minimum required word count.
        max_ratio_caps (float): Maximum allowed ratio of ALL-CAPS words.
        max_ratio_special (float): Maximum allowed ratio of non-alphanumeric special characters.

    Returns:
        bool: True if the document meets quality criteria, False otherwise.
    """
    words = text.split()
    if len(words) < min_words:
        return False

    alpha_words = [w for w in words if w.isalpha()]
    caps_words = [w for w in alpha_words if w.isupper()]
    ratio_caps = len(caps_words) / len(alpha_words) if alpha_words else 0.0
    if ratio_caps > max_ratio_caps:
        return False

    if text:
        special_chars = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        ratio_special = special_chars / len(text)
        if ratio_special > max_ratio_special:
            return False

    return True


def get_shingles(text: str, k: int = 5) -> Set[str]:
    """
    Extracts k-shingles (word n-grams) from a text string.

    Args:
        text (str): Input document text.
        k (int): Size of word shingle (n-gram length).

    Returns:
        Set[str]: Unique set of k-word shingles.
    """
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash_signature(shingles: Set[str], num_hashes: int = 128) -> List[int]:
    """
    Computes MinHash signature for a set of shingles using hash function permutations.

    Args:
        shingles (Set[str]): Set of unique word shingles.
        num_hashes (int): Length of the MinHash signature vector.

    Returns:
        List[int]: MinHash signature list of length (num_hashes,).
    """
    if not shingles:
        return [0] * num_hashes

    signature = []
    for seed in range(num_hashes):
        min_hash = min(
            int(hashlib.md5(f"{seed}_{s}".encode("utf-8")).hexdigest(), 16)
            for s in shingles
        )
        signature.append(min_hash)
    return signature



def lsh_buckets(signature: List[int], bands: int = 16) -> List[Tuple[int, str]]:
    """
    Divides MinHash signature into bands and maps each band to a bucket identifier using LSH.

    Args:
        signature (List[int]): MinHash signature vector of shape (num_hashes,).
        bands (int): Number of bands to divide the signature into.

    Returns:
        List[Tuple[int, str]]: List of (band_id, bucket_hash) tuples of length (bands,).
    """
    rows_per_band = max(len(signature) // bands, 1)
    buckets = []
    for band_id in range(bands):
        start = band_id * rows_per_band
        end = start + rows_per_band
        band_values = tuple(signature[start:end])
        bucket_hash = hashlib.md5(str(band_values).encode("utf-8")).hexdigest()
        buckets.append((band_id, bucket_hash))
    return buckets


def deduplicate(
    documents: List[str], 
    threshold: float = 0.8, 
    num_hashes: int = 128, 
    bands: int = 16
) -> Tuple[List[str], int]:
    """
    Finds and removes near-duplicate documents using MinHash LSH and Jaccard similarity evaluation.

    Args:
        documents (List[str]): List of clean document strings.
        threshold (float): Jaccard similarity threshold for considering two docs as duplicates.
        num_hashes (int): Total hash functions for MinHash computation.
        bands (int): Number of bands for LSH partitioning.

    Returns:
        Tuple[List[str], int]: Tuple containing (list of deduplicated documents, count of removed duplicates).
    """
    shingle_sets = [get_shingles(doc) for doc in documents]
    signatures = [minhash_signature(s, num_hashes) for s in shingle_sets]

    band_buckets: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for doc_idx, signature in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(signature, bands):
            band_buckets[(band_id, bucket_hash)].append(doc_idx)

    candidate_pairs: Set[Tuple[int, int]] = set()
    for doc_indices in band_buckets.values():
        if len(doc_indices) < 2:
            continue
        for i in range(len(doc_indices)):
            for j in range(i + 1, len(doc_indices)):
                candidate_pairs.add((doc_indices[i], doc_indices[j]))

    removed: Set[int] = set()
    for i, j in sorted(candidate_pairs):
        if i in removed or j in removed:
            continue
        set_i, set_j = shingle_sets[i], shingle_sets[j]
        union = set_i | set_j
        if not union:
            continue
        jaccard = len(set_i & set_j) / len(union)
        if jaccard >= threshold:
            removed.add(j)

    deduped_docs = [doc for idx, doc in enumerate(documents) if idx not in removed]
    return deduped_docs, len(removed)


class SimpleTokenizer:
    """
    A minimal Byte Pair Encoding (BPE) tokenizer implementation operating over byte sequences.
    """
    def __init__(self, vocab_size: int = 256):
        """
        Initializes tokenizer vocabulary and internal mapping structures.

        Args:
            vocab_size (int): Target vocabulary capacity.
        """
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.next_id: int = 256
        self.eos_id: Optional[int] = None
        self.pad_id: int = 0

    def train_bpe(self, text: str, num_merges: int) -> None:
        """
        Iteratively finds the most frequent pair of tokens and merges them into a new token.

        Args:
            text (str): Training text corpus.
            num_merges (int): Number of BPE merge operations to execute.

        Returns:
            None
        """
        tokens = list(text.encode("utf-8"))

        for _ in range(num_merges):
            pairs: Counter = Counter()
            for a, b in zip(tokens, tokens[1:]):
                pairs[(a, b)] += 1

            if not pairs:
                break

            best_pair = max(pairs.items(), key=lambda item: item[1])[0]
            new_id = self.next_id
            self.next_id += 1

            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = new_id

            merged = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        self.eos_id = self.next_id
        self.vocab[self.eos_id] = b""
        self.next_id += 1

    def encode(self, text: str) -> List[int]:
        """
        Encodes input text into a list of token IDs using pre-learned BPE merge rules.

        Args:
            text (str): Input text string.

        Returns:
            List[int]: Encoded list of token IDs.
        """
        tokens = list(text.encode("utf-8"))

        while len(tokens) > 1:
            pairs = list(zip(tokens, tokens[1:]))
            ranked = [(self.merges[pair], pair) for pair in pairs if pair in self.merges]
            if not ranked:
                break
            _, best_pair = min(ranked)
            new_id = self.merges[best_pair]

            merged = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        return tokens

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into a UTF-8 string.

        Args:
            ids (List[int]): List of token IDs.

        Returns:
            str: Decoded text representation.
        """
        special_ids = {self.eos_id, self.pad_id} if self.eos_id is not None else {self.pad_id}
        raw_bytes = b"".join(
            self.vocab.get(token_id, b"") for token_id in ids if token_id not in special_ids or self.vocab.get(token_id) != b""
        )
        return raw_bytes.decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        """
        Returns the total vocabulary size including base bytes and merges.

        Returns:
            int: Total vocabulary size.
        """
        return len(self.vocab)


def tokenize_corpus(documents: List[str], tokenizer: SimpleTokenizer) -> List[int]:
    """
    Tokenizes a list of documents into a single flat stream of token IDs separated by EOS markers.

    Args:
        documents (List[str]): List of document strings.
        tokenizer (SimpleTokenizer): Trained BPE tokenizer instance.

    Returns:
        List[int]: Flat list containing combined token sequence.
    """
    flat_tokens: List[int] = []
    for doc in documents:
        flat_tokens.extend(tokenizer.encode(doc))
        flat_tokens.append(tokenizer.eos_id)
    return flat_tokens


def pack_sequences(
    token_ids: List[int], 
    seq_length: int, 
    pad_id: int = 0
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Packs continuous token stream into fixed-length sequence chunks with corresponding attention masks.

    Args:
        token_ids (List[int]): Continuous flat list of token IDs.
        seq_length (int): Target length per sequence block.
        pad_id (int): Token ID used for padding incomplete trailing sequences.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: Tuple of (padded_sequences, attention_masks)
            where each element has dynamic batch outer dimension and sequence shape (seq_length,).
    """
    if not token_ids:
        return [], []

    sequences: List[List[int]] = []
    masks: List[List[int]] = []

    for start in range(0, len(token_ids), seq_length):
        chunk = token_ids[start:start + seq_length]
        valid_len = len(chunk)
        if valid_len < seq_length:
            chunk = chunk + [pad_id] * (seq_length - valid_len)
        mask = [1] * valid_len + [0] * (seq_length - valid_len)
        sequences.append(chunk)
        masks.append(mask)

    return sequences, masks


class PreTrainingDataLoader:
    """
    Data loader providing mini-batch iteration and shuffling over packed token sequences.
    """
    def __init__(
        self, 
        sequences: List[List[int]], 
        attention_masks: List[List[int]], 
        batch_size: int, 
        shuffle: bool = True
    ):
        """
        Initializes dataloader properties.

        Args:
            sequences (List[List[int]]): Packed input sequences of shape (num_samples, seq_length).
            attention_masks (List[List[int]]): Attention masks of shape (num_samples, seq_length).
            batch_size (int): Size of individual mini-batches.
            shuffle (bool): Whether to shuffle sample indices prior to batching.
        """
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """
        Calculates total number of batches per epoch.

        Returns:
            int: Number of available mini-batches.
        """
        n = len(self.sequences)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Generator[Tuple[List[List[int]], List[List[int]]], None, None]:
        """
        Iterates over the dataset and yields mini-batches of sequences and attention masks.

        Returns:
            Generator[Tuple[List[List[int]], List[List[int]]], None, None]: Batch generator yielding 
            (batch_sequences, batch_attention_masks) where each tensor component has shape (batch_size, seq_length).
        """
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_indices]
            batch_masks = [self.attention_masks[i] for i in batch_indices]
            yield batch_seqs, batch_masks


def compute_statistics(
    documents: List[str], 
    token_ids: List[int], 
    sequences: List[List[int]], 
    tokenizer_vocab_size: int
) -> Dict[str, Any]:
    """
    Computes overall corpus, vocabulary, compression, and sequence padding efficiency metrics.

    Args:
        documents (List[str]): Corpus of cleaned documents.
        token_ids (List[int]): Continuous flat list of token IDs.
        sequences (List[List[int]]): Packed token sequences.
        tokenizer_vocab_size (int): Size of the tokenizer vocabulary.

    Returns:
        Dict[str, Any]: Dictionary containing pre-training dataset analytical statistics.
    """
    total_documents = len(documents)
    total_characters = sum(len(doc) for doc in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))

    compression_ratio = total_characters / total_tokens if total_tokens else 0.0
    avg_doc_length_words = (
        sum(len(doc.split()) for doc in documents) / total_documents if total_documents else 0.0
    )

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    total_positions = sum(len(seq) for seq in sequences)
    non_pad_positions = sum(1 for seq in sequences for tok in seq if tok != 0)
    sequence_utilization = non_pad_positions / total_positions if total_positions else 0.0

    vocab_utilization = unique_tokens / tokenizer_vocab_size if tokenizer_vocab_size else 0.0

    return {
        "total_documents": total_documents,
        "total_characters": total_characters,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": vocab_utilization,
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length_words,
        "num_sequences": len(sequences),
        "sequence_utilization": sequence_utilization,
        "top_tokens": top_tokens,
    }


# [KEEP_IMPLEMENTATION]
def generate_sample_corpus() -> List[str]:
    """
    Generates a pre-defined synthetic corpus containing standard, duplicate, HTML, and spam documents.

    Returns:
        List[str]: List of raw document strings.
    """
    base_docs = [
        "Machine learning is a subset of artificial intelligence that provides systems the ability "
        "to automatically learn and improve from experience without being explicitly programmed. "
        "Machine learning focuses on the development of computer programs that can access data and "
        "use it to learn for themselves. The process of learning begins with observations or data, "
        "such as examples, direct experience, or instruction, in order to look for patterns in data "
        "and make better decisions in the future based on the examples that we provide.",

        "Deep learning is part of a broader family of machine learning methods based on artificial "
        "neural networks with representation learning. Learning can be supervised, semi-supervised "
        "or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural "
        "networks, convolutional neural networks and transformers have been applied to fields "
        "including natural language processing, speech recognition, computer vision, and many other tasks.",

        "Natural language processing is a subfield of linguistics, computer science, and artificial "
        "intelligence concerned with the interactions between computers and human language, in "
        "particular how to program computers to process and analyze large amounts of natural language "
        "data. The result is a computer capable of understanding the contents of documents, including "
        "the contextual nuances of the language within them.",

        "Transformers are a type of neural network architecture that has become the dominant approach "
        "for natural language processing tasks. The key innovation is the self-attention mechanism, "
        "which allows the model to weigh the importance of different parts of the input when producing "
        "each part of the output. This enables transformers to capture long-range dependencies in text "
        "much more effectively than previous recurrent approaches.",

        "The attention mechanism in neural networks allows the model to focus on relevant parts of "
        "the input sequence when generating each element of the output. In the transformer architecture, "
        "multi-head attention computes attention in parallel across multiple representation subspaces, "
        "enabling the model to jointly attend to information from different representation subspaces "
        "at different positions in the sequence.",

        "Reinforcement learning is an area of machine learning concerned with how intelligent agents "
        "ought to take actions in an environment in order to maximize the notion of cumulative reward. "
        "Reinforcement learning is one of three basic machine learning paradigms, alongside supervised "
        "learning and unsupervised learning. It differs from supervised learning in that correct input "
        "and output pairs need not be presented.",

        "Computer vision is an interdisciplinary scientific field that deals with how computers can "
        "gain high-level understanding from digital images or videos. From the perspective of "
        "engineering, it seeks to understand and automate tasks that the human visual system can do. "
        "Computer vision tasks include methods for acquiring, processing, analyzing and understanding "
        "digital images, and extraction of high-dimensional data from the real world.",

        "Convolutional neural networks are a class of deep learning architecture commonly applied to "
        "analyze visual imagery. They use a variation of multilayer perceptrons designed to require "
        "minimal preprocessing. They are also known as shift invariant or space invariant artificial "
        "neural networks based on their shared-weights architecture and translation invariance "
        "characteristics. Convolutional networks were inspired by biological processes.",

        "Generative adversarial networks consist of two neural networks that contest with each other "
        "in the form of a zero-sum game, where one agent gain is another agent loss. Given a training "
        "set, this technique learns to generate new data with the same statistics as the training set. "
        "For example, a generative adversarial network trained on photographs can generate new "
        "photographs that look authentic to human observers.",

        "Transfer learning is a machine learning method where a model developed for a task is reused "
        "as the starting point for a model on a second task. It is a popular approach in deep learning "
        "where pre-trained models are used as the starting point on computer vision and natural language "
        "processing tasks given the vast compute and time resources required to develop neural network "
        "models on these problems and the large improvements they provide.",
    ]

    near_dup_1 = (
        "Machine learning is a subset of artificial intelligence that provides systems the ability "
        "to automatically learn and improve from experience. Machine learning focuses on developing "
        "computer programs that can access data and use it to learn for themselves. The learning "
        "process begins with observations or data, such as examples or direct experience, in order "
        "to look for patterns and make better decisions based on the examples provided."
    )

    near_dup_2 = (
        "Deep learning is part of a broader family of machine learning methods based on artificial "
        "neural networks with representation learning. Learning can be supervised, semi-supervised "
        "or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural "
        "networks, convolutional neural networks and transformers have been applied to fields "
        "including natural language processing, speech recognition, computer vision, and many other tasks."
    )

    short_doc = "This is too short to be useful."

    html_doc = (
        "<html><body><h1>Title</h1><p>Machine learning is transforming how we build software. "
        "Deep neural networks can learn complex patterns from data. The transformer architecture "
        "has become the dominant approach for language tasks. Self-attention allows models to capture "
        "long-range dependencies. Pre-training on large corpora produces strong foundation models. "
        "Fine-tuning adapts these models to specific tasks with minimal additional data.</p></body></html>"
    )

    spam_doc = "BUY NOW CLICK HERE FREE MONEY GUARANTEED RESULTS " * 20

    docs = base_docs + [near_dup_1, near_dup_2, short_doc, html_doc, spam_doc]
    return docs


# [KEEP_IMPLEMENTATION]
def run_pipeline():
    """
    Executes end-to-end data processing pipeline for pre-training dataset preparation.
    """
    print("=" * 60)
    print("Data Pipeline for Pre-Training")
    print("=" * 60)

    raw_docs = generate_sample_corpus()
    print(f"\nRaw documents: {len(raw_docs)}")

    print("\n--- Stage 1: Cleaning ---")
    cleaned_docs = [clean_text(doc) for doc in raw_docs]
    print(f"After HTML stripping: {len(cleaned_docs)} documents")

    print("\n--- Stage 2: Quality Filtering ---")
    filtered_docs = [doc for doc in cleaned_docs if quality_filter(doc)]
    removed_quality = len(cleaned_docs) - len(filtered_docs)
    print(f"Removed {removed_quality} low-quality documents")
    print(f"Remaining: {len(filtered_docs)} documents")

    print("\n--- Stage 3: Deduplication (MinHash + LSH) ---")
    start = time.time()
    deduped_docs, num_removed = deduplicate(filtered_docs, threshold=0.8)
    dedup_time = time.time() - start
    print(f"Removed {num_removed} near-duplicates in {dedup_time:.2f}s")
    print(f"Remaining: {len(deduped_docs)} documents")

    print("\n--- Stage 4: Tokenization ---")
    all_text = " ".join(deduped_docs)
    tokenizer = SimpleTokenizer()
    start = time.time()
    tokenizer.train_bpe(all_text, num_merges=100)
    train_time = time.time() - start
    print(f"Trained tokenizer with {tokenizer.vocab_size()} tokens in {train_time:.2f}s")

    start = time.time()
    token_ids = tokenize_corpus(deduped_docs, tokenizer)
    tok_time = time.time() - start
    print(f"Tokenized {len(token_ids):,} tokens in {tok_time:.2f}s ({len(token_ids)/max(tok_time, 0.001):,.0f} tokens/sec)")

    print("\n--- Stage 5: Sequence Packing ---")
    seq_length = 128
    sequences, masks = pack_sequences(token_ids, seq_length, pad_id=0)
    print(f"Packed into {len(sequences)} sequences of length {seq_length}")

    print("\n--- Stage 6: DataLoader ---")
    batch_size = 4
    loader = PreTrainingDataLoader(sequences, masks, batch_size)
    print(f"DataLoader: {len(loader)} batches of size {batch_size}")

    batch_count = 0
    total_tokens_served = 0
    for batch_seqs, batch_masks in loader:
        batch_count += 1
        total_tokens_served += sum(sum(m) for m in batch_masks)
        if batch_count <= 2:
            print(f"\n  Batch {batch_count}:")
            print(f"    Sequences: {len(batch_seqs)}")
            print(f"    First seq (first 20 tokens): {batch_seqs[0][:20]}...")
            print(f"    First mask (first 20): {batch_masks[0][:20]}...")
    print(f"\n  Total batches served: {batch_count}")
    print(f"  Total non-padding tokens served: {total_tokens_served:,}")

    print("\n--- Dataset Statistics ---")
    stats = compute_statistics(deduped_docs, token_ids, sequences, tokenizer.vocab_size())
    print(f"  Documents:           {stats['total_documents']}")
    print(f"  Total characters:    {stats['total_characters']:,}")
    print(f"  Total tokens:        {stats['total_tokens']:,}")
    print(f"  Unique tokens:       {stats['unique_tokens']}")
    print(f"  Vocab utilization:  {stats['vocab_utilization']:.1%}")
    print(f"  Compression ratio:  {stats['compression_ratio']:.2f} chars/token")
    print(f"  Avg doc length:     {stats['avg_doc_length_words']:.0f} words")
    print(f"  Num sequences:      {stats['num_sequences']}")
    print(f"  Seq utilization:    {stats['sequence_utilization']:.1%}")

    print("\n--- Pipeline Summary ---")
    print(f"  Raw documents:       {len(raw_docs)}")
    print(f"  After cleaning:      {len(cleaned_docs)}")
    print(f"  After quality filter: {len(filtered_docs)} (-{removed_quality})")
    print(f"  After dedup:         {len(deduped_docs)} (-{num_removed})")
    print(f"  Final tokens:        {len(token_ids):,}")
    print(f"  Training sequences:  {len(sequences)}")
    print(f"  Training batches:    {len(loader)}")


# ===== UNIT TESTS =====

def test_clean_text():
    sample = "<html><body>Hello    World! http://example.com</body></html>"
    cleaned = clean_text(sample)
    assert isinstance(cleaned, str), "Output must be a string."
    assert "<html>" not in cleaned, "HTML tags were not removed."
    assert "http" not in cleaned, "URLs were not stripped."
    assert "  " not in cleaned, "Multiple spaces were not collapsed."
    
    # Edge case: Empty input string
    assert clean_text("") == "", "Edge case failed: Empty string must return an empty string."


def test_quality_filter():
    good_text = " ".join(["word"] * 60)
    bad_text_short = "short text"
    bad_text_caps = " ".join(["WORD"] * 60)
    
    assert quality_filter(good_text) is True, "Valid document failed quality filter."
    assert quality_filter(bad_text_short) is False, "Short document incorrectly passed quality filter."
    assert quality_filter(bad_text_caps) is False, "Overly-capitalized document passed quality filter."
    
    # Edge case: Boundary word count
    exact_words = " ".join(["word"] * 50)
    assert quality_filter(exact_words, min_words=50) is True, "Edge case failed: Exact min_words boundary missed."


def test_get_shingles():
    text = "the quick brown fox jumps over the lazy dog"
    shingles = get_shingles(text, k=3)
    assert isinstance(shingles, set), "Output should be a set."
    assert len(shingles) == 7, "Shingle extraction length mismatch."
    
    # Edge case: Text shorter than k
    short_text = "quick brown"
    assert get_shingles(short_text, k=5) == set(), "Edge case failed: Text shorter than k must return an empty set."


def test_minhash_signature():
    shingles = {"quick brown fox", "brown fox jumps"}
    sig = minhash_signature(shingles, num_hashes=32)
    assert isinstance(sig, list), "MinHash signature must be a list."
    assert len(sig) == 32, "Signature length must match requested num_hashes."
    
    # Edge case: Empty shingle set
    empty_sig = minhash_signature(set(), num_hashes=16)
    assert len(empty_sig) == 16, "Edge case failed: MinHash length mismatched for empty shingles."
    assert all(val == 0 for val in empty_sig), "Edge case failed: Empty shingle signature must default to 0s."


def test_lsh_buckets():
    random.seed(42)
    sig = [random.randint(0, 1000) for _ in range(128)]
    buckets = lsh_buckets(sig, bands=16)
    assert isinstance(buckets, list), "Output must be a list."
    assert len(buckets) == 16, "Number of generated buckets must match bands count."
    assert len(buckets[0]) == 2, "Each bucket element must be a (band_id, bucket_hash) tuple."


def test_deduplicate():
    docs = [
        "Natural language processing is a subfield of linguistics and computer science.",
        "Natural language processing is a subfield of linguistics and computer science.",
        "Computer vision allows computers to derive meaningful information from digital images."
    ]
    deduped, removed = deduplicate(docs, threshold=0.8)
    assert isinstance(deduped, list), "Deduplicated docs must be returned as a list."
    assert len(deduped) == 2, "Duplicate document was not removed."
    assert removed == 1, "Removed duplicate count is incorrect."
    
    # Edge case: Single document input
    single_doc = ["Single document text."]
    d_out, r_out = deduplicate(single_doc)
    assert len(d_out) == 1 and r_out == 0, "Edge case failed: Single document should yield no duplicates."


def test_simple_tokenizer():
    tok = SimpleTokenizer()
    text = "hello world hello world"
    tok.train_bpe(text, num_merges=2)
    
    encoded = tok.encode("hello world")
    assert isinstance(encoded, list), "Encoded tokens must be a list of IDs."
    assert len(encoded) > 0, "Encoded sequence should not be empty."
    
    decoded = tok.decode(encoded)
    assert isinstance(decoded, str), "Decoded output must be a string."
    assert decoded == "hello world", "Decoded text does not match original."
    
    # Edge case: Out of vocabulary / raw unseen bytes
    unseen = tok.encode("xyz123")
    assert tok.decode(unseen) == "xyz123", "Edge case failed: Unseen characters were not decoded back accurately."


def test_tokenize_corpus():
    tok = SimpleTokenizer()
    tok.train_bpe("test document", num_merges=1)
    docs = ["test", "document"]
    tokens = tokenize_corpus(docs, tok)
    
    assert isinstance(tokens, list), "Corpus tokens must be a list."
    assert tokens.count(tok.eos_id) == 2, "Corpus token sequence must include one EOS token per document."


def test_pack_sequences():
    tokens = list(range(10))
    seqs, masks = pack_sequences(tokens, seq_length=4, pad_id=0)
    
    assert len(seqs) == 3, "Packed sequence count mismatch."
    assert len(seqs[0]) == 4, "Sequence block length mismatch."
    assert seqs[2] == [8, 9, 0, 0], "Padding incorrect on last sequence block."
    assert masks[2] == [1, 1, 0, 0], "Attention mask incorrect on padded sequence block."
    
    # Edge case: Empty token input
    e_seqs, e_masks = pack_sequences([], seq_length=4)
    assert len(e_seqs) == 0 and len(e_masks) == 0, "Edge case failed: Empty tokens should return empty lists."


def test_pre_training_data_loader():
    seqs = [[1, 2, 3, 4] for _ in range(10)]
    masks = [[1, 1, 1, 1] for _ in range(10)]
    loader = PreTrainingDataLoader(seqs, masks, batch_size=4, shuffle=False)
    
    assert len(loader) == 3, "DataLoader batch count mismatch."
    
    batches = list(loader)
    first_b_seqs, first_b_masks = batches[0]
    assert len(first_b_seqs) == 4, "Batch size mismatch."
    assert len(first_b_masks) == 4, "Attention mask batch size mismatch."
    
    # Edge case: Batch size greater than dataset size
    loader_large = PreTrainingDataLoader(seqs[:2], masks[:2], batch_size=8)
    large_batches = list(loader_large)
    assert len(large_batches) == 1, "Edge case failed: Oversized batch should return exactly 1 mini-batch."
    assert len(large_batches[0][0]) == 2, "Edge case failed: Single batch size should match total available samples."


def test_compute_statistics():
    docs = ["one two three", "four five"]
    token_ids = [1, 2, 3, 4, 5]
    sequences = [[1, 2, 3], [4, 5, 0]]
    
    stats = compute_statistics(docs, token_ids, sequences, tokenizer_vocab_size=256)
    assert isinstance(stats, dict), "Statistics output must be a dictionary."
    assert stats["total_documents"] == 2, "Stat mismatch: Total documents."
    assert stats["total_tokens"] == 5, "Stat mismatch: Total tokens."
    assert "compression_ratio" in stats, "Missing metric: compression_ratio."
    assert 0.0 <= stats["sequence_utilization"] <= 1.0, "Sequence utilization ratio out of valid bounds [0, 1]."


if __name__ == "__main__":
    run_pipeline()