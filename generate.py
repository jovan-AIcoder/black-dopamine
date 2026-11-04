import os
import re
import json
import numpy as np
import tensorflow as tf

# Configuration parameters matching sequencing.py & train.py
PROCESSED_DIR = "processed"
MODEL_PATH = "model/black_dopamine.keras"
VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab.json")

def sample_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Apply temperature scaling to raw logits and sample a token index."""
    logits = np.asarray(logits, dtype=np.float64)
    
    # Low temperature limit: deterministic / greedy selection
    if temperature <= 0.001:
        return int(np.argmax(logits))
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Compute numerically stable Softmax probabilities
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Random weighted choice based on the scaled probability distribution
    return int(np.random.choice(len(probabilities), p=probabilities))

def clean_input(text: str) -> str:
    """Clean prompt text to match training pipeline preprocessing."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_text(
    prompt: str,
    model: tf.keras.Model,
    word2idx: dict,
    idx2word: dict,
    max_seq_len: int,
    max_gen_len: int = 20,
    temperature: float = 0.7
) -> str:
    """Autoregressively generate text completion from a prompt using temperature scaling."""
    cleaned_prompt = clean_input(prompt)
    tokens = [word2idx.get(w, word2idx["<UNK>"]) for w in cleaned_prompt.split()]
    
    if not tokens:
        tokens = [word2idx["<UNK>"]]

    eos_idx = word2idx["<EOS>"]
    generated_tokens = list(tokens)

    # Calculate input sequence length expected by the model (max_seq_len - 1)
    input_len = max_seq_len - 1

    for _ in range(max_gen_len):
        # Extract sliding context window
        current_seq = generated_tokens[-input_len:]
        
        # Pre-padding with 0 (<PAD>)
        pad_len = input_len - len(current_seq)
        padded_seq = [0] * pad_len + current_seq
        
        # Reshape to (1, input_len) for Keras batch dimension
        input_array = np.array([padded_seq], dtype=np.int32)
        
        # Predict logits for the next token
        predictions = model.predict(input_array, verbose=0)
        logits = predictions[0]
        
        # Sample next token with temperature parameter
        next_token_id = sample_with_temperature(logits, temperature=temperature)
        
        # Check for stopping cue (<EOS>)
        if next_token_id == eos_idx:
            break
            
        generated_tokens.append(next_token_id)

    # Decode token IDs back into string words
    result_words = [idx2word.get(str(t), idx2word.get(t, "<UNK>")) for t in generated_tokens]
    
    # Filter out special metadata tokens from output display
    output_words = [w for w in result_words if w not in ["<PAD>", "<UNK>", "<EOS>"]]
    return " ".join(output_words)

def main():
    print('\n=== WELCOME TO BLACK DOPAMINE MODEL ===\n')

    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file '{MODEL_PATH}' not found. Please train the model first.")
        return
        
    if not os.path.exists(VOCAB_PATH):
        print(f"[Error] Vocabulary file '{VOCAB_PATH}' not found. Run sequencing.py first.")
        return

    print("Loading pre-trained model and metadata...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    word2idx = metadata["word2idx"]
    idx2word = metadata["idx2word"]
    max_seq_len = metadata["max_seq_len"]

    print(f"Model loaded successfully! Vocabulary size: {len(word2idx)} tokens.")
    print('\nWARNING: This model is trained in INDONESIAN [ID] language. Please input prompt IN INDONESIAN [ID].\n')
    print("=== Interactive Prompt Generation Ready! ===")
    print("Type 'quit' or 'exit' to stop the execution.\n")
    
    while True:
        try:
            prompt = input("Prompt > ")
            if prompt.lower().strip() in ["quit", "exit"]:
                print("Exiting BLACK DOPAMINE. Good luck!")
                break
                
            if not prompt.strip():
                print("[Warning] Prompt cannot be empty. Please enter a word or phrase.\n")
                continue
                
            temp_input = input("Temperature (default 0.001) > ")
            try:
                temperature = float(temp_input) if temp_input.strip() else 0.001
            except ValueError:
                print("[Warning] Invalid temperature input. Falling back to default (0.001).")
                temperature = 0.001

            gen_len_input = input("Max Words to Generate (default 20) > ")
            try:
                max_gen_len = int(gen_len_input) if gen_len_input.strip() else 20
            except ValueError:
                max_gen_len = 20

            print("\nGenerating text completion...")
            response = generate_text(
                prompt=prompt,
                model=model,
                word2idx=word2idx,
                idx2word=idx2word,
                max_seq_len=max_seq_len,
                max_gen_len=max_gen_len,
                temperature=temperature
            )
            
            print("\n" + "="*50)
            print(f"Generated Output:\n{response}")
            print("="*50 + "\n")

        except KeyboardInterrupt:
            print("\nExecution interrupted by user. Exiting...")
            break

if __name__ == "__main__":
    main()