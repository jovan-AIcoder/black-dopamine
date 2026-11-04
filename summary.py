import json
import os
import tensorflow as tf

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
MODEL_PATH = "model/black_dopamine.keras"  # Adjust format if using .keras or SavedModel directory
METADATA_PATH = "model/model_metadata.json"
VOCAB_PATH = "processed/vocab.json"
X_TEST_PATH = "processed/X_data.npy"
Y_TEST_PATH = "processed/Y_data.npy"

print("=" * 60)
print("       BLACK DOPAMINE — MODEL & ARTIFACT SUMMARY       ")
print("=" * 60)

# ------------------------------------------
# 1. LOAD MODEL & DISPLAY SUMMARY
# ------------------------------------------
print("\n[1/4] Loading Model Architecture & Summary...")
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("-" * 60)
    model.summary()
    print("-" * 60)
else:
    print(f"⚠️ Warning: Model file '{MODEL_PATH}' was not found.")

# ------------------------------------------
# 2. EVALUATE MODEL PERFORMANCE
# ------------------------------------------
print("\n[2/4] Evaluating Model Performance...")
if os.path.exists(MODEL_PATH) and os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH):
    import numpy as np
    
    x_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1, batch_size=512)
    print(f"  • Test Loss     : {loss:.4f}")
    print(f"  • Test Accuracy : {accuracy * 100:.2f}%")
else:
    print("⚠️ Skip Evaluation: Test dataset (.npy) or model file was not found.")

# ------------------------------------------
# 3. DISPLAY SAVED METADATA
# ------------------------------------------
print("\n[3/4] Fetching Execution Metadata...")
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print("-" * 40)
    for key, value in metadata.items():
        print(f"  • {key:<20} : {value}")
    print("-" * 40)
else:
    print(f"⚠️ Warning: Metadata file '{METADATA_PATH}' was not found.")

# ------------------------------------------
# 4. VOCAB.JSON CHARACTERISTICS 
# ------------------------------------------
print("\n[4/4] Analyzing vocab.json Characteristics...")
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    
    # Extract actual word dictionary if stored under 'word2idx' key
    if isinstance(vocab_data, dict) and "word2idx" in vocab_data:
        actual_vocab = vocab_data["word2idx"]
    else:
        actual_vocab = vocab_data

    vocab_words = list(actual_vocab.keys())
    vocab_size = len(vocab_words)
    
    word_lengths = [len(str(w)) for w in vocab_words]
    avg_word_len = sum(word_lengths) / vocab_size if vocab_size > 0 else 0
    longest_word = max(vocab_words, key=len) if vocab_size > 0 else ""
    shortest_word = min(vocab_words, key=len) if vocab_size > 0 else ""
    
    special_tokens = [w for w in vocab_words if str(w).startswith("<") and str(w).endswith(">")]

    print("-" * 40)
    print(f"  • Total Vocab Size      : {vocab_size:,} tokens")
    print(f"  • Special Tokens        : {special_tokens}")
    print(f"  • Average Word Length   : {avg_word_len:.2f} characters")
    print(f"  • Longest Token         : '{longest_word}' ({len(longest_word)} chars)")
    print(f"  • Shortest Token        : '{shortest_word}' ({len(shortest_word)} chars)")
    print("-" * 40)
else:
    print(f"⚠️ Warning: Vocabulary file '{VOCAB_PATH}' was not found.")

print("\n" + "=" * 60)
print("                SUMMARY PROCESS COMPLETED               ")
print("=" * 60)