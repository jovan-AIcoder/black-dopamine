import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Hyperparameter Configurations
PROCESSED_DIR = "processed"
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 20
MODEL_SAVE_PATH = "model/black_dopamine.keras"

def main():
    print("=== Initializing TensorFlow LSTM Training Pipeline ===")

    # 1. Check artifact dependencies
    x_path = os.path.join(PROCESSED_DIR, "X_data.npy")
    y_path = os.path.join(PROCESSED_DIR, "Y_data.npy")
    vocab_path = os.path.join(PROCESSED_DIR, "vocab.json")

    if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(vocab_path)):
        print(f"[Error] Required preprocessed artifacts not found in '{PROCESSED_DIR}/'.")
        print("Please execute 'sequencing.py' first.")
        return

    # 2. Load preprocessed datasets and metadata
    print("Loading preprocessed arrays and vocabulary metadata...")
    X = np.load(x_path)
    Y = np.load(y_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vocab_size = metadata["vocab_size"]
    max_seq_len = metadata["max_seq_len"]
    input_length = X.shape[1]  # max_seq_len - 1

    print(f"Dataset successfully loaded: {len(X)} samples, Vocabulary Size: {vocab_size}")
    print(f"Input sequence length: {input_length}")

    # 3. Build Model Architecture using Keras Sequential API
    print("Building TensorFlow LSTM model architecture... (The Model Name is Black Dopamine!)")
    model = Sequential()
    
    # Embedding Layer with zero mask for padding token (index 0)
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        input_length=input_length,
        mask_zero=True
    ))

    # Stacking LSTM Layers based on NUM_LAYERS configuration
    for i in range(NUM_LAYERS):
        is_last_layer = (i == NUM_LAYERS - 1)
        model.add(LSTM(
            units=HIDDEN_DIM,
            return_sequences=not is_last_layer  # Only last layer returns a 2D vector for final prediction
        ))

    # Dense Output Layer with Softmax activation for vocabulary probability distribution
    model.add(Dense(units=vocab_size, activation="softmax"))

    # 4. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n--- Model Summary ---")
    model.summary()
    print("---------------------\n")

    # 5. Define Model Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="loss",
        save_best_only=True,
        verbose=1
    )

    # 6. Model Training Loop
    print(f"Starting training for {EPOCHS} epochs with Batch Size {BATCH_SIZE}...")
    history = model.fit(
        X, Y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback]
    )

    # 7. Save Final Trained Model Weights & Configs
    print(f"\nSaving final model state to '{MODEL_SAVE_PATH}'...")
    model.save(MODEL_SAVE_PATH)

    # Save training metadata alongside for inference reference
    model_metadata = {
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len
    }
    with open("model/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    print("=== Training Completed Successfully! ===")


if __name__ == "__main__":
    main()