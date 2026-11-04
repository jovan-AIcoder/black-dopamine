import os
import re
import json
import numpy as np

# Configuration parameters
DATASET_PATH = "dataset/raw_youtube_comments.csv"
OUTPUT_DIR = "processed"
MIN_WORD_FREQ = 1       # Minimum word frequency to retain in vocabulary
MAX_ALLOWED_LEN = 30    # Cap sequence length to prevent excessive RNN unrolling

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"     # Special token for stopping generation

def clean_text(text: str) -> str:
    """Clean raw text string: lowercase, remove URLs, HTML tags, mentions, timestamps, and unwanted symbols."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)                                        # Remove HTML tags
    text = re.sub(r"@\w+", "", text)                                         # Remove @mentions completely
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)                # Remove timestamps (e.g. 03:45, 1:12:00)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)                               # Retain alphanumeric and spaces
    text = re.sub(r"\s+", " ", text).strip()                                 # Remove extra spaces
    return text

def main():
    print("=== Starting Sequence Generation & Preprocessing ===")
    
    # 1. Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load raw text dataset
    if not os.path.exists(DATASET_PATH):
        print(f"[Error] Dataset file not found at '{DATASET_PATH}'. Please check the path.")
        return

    print(f"Loading raw dataset from '{DATASET_PATH}'...")
    raw_comments = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = clean_text(line)
            if cleaned:
                raw_comments.append(cleaned)
    
    print(f"Loaded {len(raw_comments)} non-empty lines after cleaning.")

    # 3. Build vocabulary (Word Frequency Mapping)
    print("Building vocabulary map...")
    word_counts = {}
    for comment in raw_comments:
        for word in comment.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    # Define Vocabulary Mapping (0: PAD, 1: UNK, 2: EOS)
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, EOS_TOKEN: 2}
    idx = 3
    for word, count in word_counts.items():
        if count >= MIN_WORD_FREQ:
            word2idx[word] = idx
            idx += 1

    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(word2idx)
    print(f"Vocabulary size created: {vocab_size} tokens (including <PAD>, <UNK>, and <EOS>).")

    # 4. Generate N-gram Sequences with EOS Token & Length Truncation
    print(f"Generating expanding N-gram sequences (truncated to max {MAX_ALLOWED_LEN} tokens)...")
    sequences = []
    eos_idx = word2idx[EOS_TOKEN]

    for comment in raw_comments:
        # Append EOS token index at the end of each comment's token list
        tokens = [word2idx.get(w, word2idx[UNK_TOKEN]) for w in comment.split()] + [eos_idx]
        
        if len(tokens) < 2:
            continue
        
        # Expanding sequence extraction with tail truncation
        for i in range(1, len(tokens)):
            n_gram_seq = tokens[: i + 1]
            
            # Truncate to keep only the most recent MAX_ALLOWED_LEN tokens if too long
            if len(n_gram_seq) > MAX_ALLOWED_LEN:
                n_gram_seq = n_gram_seq[-MAX_ALLOWED_LEN:]
                
            sequences.append(n_gram_seq)

    print(f"Total N-gram samples generated: {len(sequences)}")

    # 5. Determine Actual Sequence Length & Perform Pre-Padding
    effective_max_len = max(len(seq) for seq in sequences)
    print(f"Effective sequence length for padding: {effective_max_len}")

    # Pad sequences with 0 (PAD_TOKEN) at the front
    padded_sequences = []
    for seq in sequences:
        pad_len = effective_max_len - len(seq)
        padded_seq = [0] * pad_len + seq
        padded_sequences.append(padded_seq)

    padded_sequences = np.array(padded_sequences, dtype=np.int32)

    # 6. Separate Input Features (X) and Target Labels (Y)
    X = padded_sequences[:, :-1]
    Y = padded_sequences[:, -1]

    print(f"Final Feature Matrix X Shape: {X.shape}")
    print(f"Final Target Array Y Shape: {Y.shape}")

    # 7. Save Arrays and Metadata to Disk
    print(f"Saving processed datasets and metadata into '{OUTPUT_DIR}/'...")
    np.save(os.path.join(OUTPUT_DIR, "X_data.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "Y_data.npy"), Y)

    metadata = {
        "max_seq_len": int(effective_max_len),
        "vocab_size": int(vocab_size),
        "total_samples": int(len(X)),
        "word2idx": word2idx,
        "idx2word": idx2word
    }

    with open(os.path.join(OUTPUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=== Processing Completed Successfully! ===")

if __name__ == "__main__":
    main()