# train_model.py
import numpy as np
import os
import random
import struct
from PIL import Image

# --- Configuration ---
PATCH_SIZE = 16  # Size of image patches (e.g., 16x16)
INPUT_DIM = PATCH_SIZE * PATCH_SIZE # Flattened patch size (grayscale)
HIDDEN_DIM = 64   # Size of the hidden layer (bottleneck)
OUTPUT_DIM = INPUT_DIM # Output dimension must match input for autoencoder
LEARNING_RATE = 0.01
EPOCHS = 5 # Number of training epochs (keep low for demo)
BATCH_SIZE = 32
IMAGES_DIR = './data/images/'
MODEL_FILENAME = 'model.jack'

# --- Activation Function ---
def sigmoid(x):
    # Add epsilon to prevent overflow/underflow issues in exp
    epsilon = 1e-8
    x_clipped = np.clip(x, -500, 500) # Prevent extreme values in exp
    return 1 / (1 + np.exp(-x_clipped) + epsilon)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# --- Loss Function ---
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# --- Autoencoder Model ---
class Autoencoder:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim # Autoencoder specific

        # Xavier/Glorot initialization (helps with training stability)
        limit_w1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        limit_w2 = np.sqrt(6.0 / (hidden_dim + self.output_dim))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, (hidden_dim, self.output_dim))
        self.b2 = np.zeros((1, self.output_dim))

        # For storing intermediate values during forward pass for backprop
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None # This is the reconstructed output

    def forward(self, X):
        """Performs the forward pass."""
        # Input to Hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1) # Hidden layer activation (bottleneck)

        # Hidden to Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2) # Output layer activation (reconstruction)

        return self.a2 # Return the reconstructed output

    def backward(self, X, y_true, y_pred, learning_rate):
        """Performs backpropagation and updates weights."""
        m = X.shape[0] # Number of examples in batch

        # --- Calculate Gradients ---
        # Output layer error
        delta_output = mean_squared_error_derivative(y_true, y_pred) * sigmoid_derivative(self.z2)
        dW2 = (1/m) * np.dot(self.a1.T, delta_output)
        db2 = (1/m) * np.sum(delta_output, axis=0, keepdims=True)

        # Hidden layer error
        delta_hidden = np.dot(delta_output, self.W2.T) * sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, delta_hidden)
        db1 = (1/m) * np.sum(delta_hidden, axis=0, keepdims=True)

        # --- Update Weights and Biases ---
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train_step(self, X_batch, learning_rate):
        """Performs one training step: forward pass, loss calc, backward pass."""
        # Input to the autoencoder is the data itself
        y_true = X_batch
        y_pred = self.forward(X_batch)
        loss = mean_squared_error(y_true, y_pred)
        self.backward(X_batch, y_true, y_pred, learning_rate)
        return loss

# --- Data Handling ---
def load_and_preprocess_image(image_path):
    """Loads an image, converts to grayscale, normalizes to [0, 1]."""
    try:
        img = Image.open(image_path).convert('L') # Convert to grayscale
        img_arr = np.array(img, dtype=np.float32) / 255.0
        return img_arr
    except Exception as e:
        print(f"Warning: Could not load image {image_path}. Error: {e}")
        return None

def extract_random_patches(image_arr, patch_size, num_patches):
    """Extracts random patches from a single image."""
    if image_arr is None:
        return []
    h, w = image_arr.shape
    patches = []
    if h < patch_size or w < patch_size:
        # print(f"Warning: Image smaller than patch size ({h}x{w} vs {patch_size}x{patch_size}). Skipping.")
        return [] # Skip images smaller than patch size

    for _ in range(num_patches):
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        patch = image_arr[top:top+patch_size, left:left+patch_size]
        patches.append(patch.flatten()) # Flatten the patch
    return patches

# --- Model Serialization (.jack format) ---
# Format Specification:
# 1. Magic Bytes (4 bytes): b'JACK'
# 2. Version (1 byte): e.g., 1
# 3. Input Dim (uint32): Number of input neurons
# 4. Hidden Dim (uint32): Number of hidden neurons
# 5. Output Dim (uint32): Number of output neurons (should == Input Dim)
# 6. W1 shape (2 x uint32): rows, cols
# 7. W1 data (float64 array bytes)
# 8. b1 shape (2 x uint32): rows, cols
# 9. b1 data (float64 array bytes)
# 10. W2 shape (2 x uint32): rows, cols
# 11. W2 data (float64 array bytes)
# 12. b2 shape (2 x uint32): rows, cols
# 13. b2 data (float64 array bytes)

def save_model(model, filename):
    """Saves the model weights and architecture to a .jack file."""
    print(f"Saving model to {filename}...")
    try:
        with open(filename, 'wb') as f:
            # 1. Magic Bytes & Version
            f.write(b'JACK')
            f.write(struct.pack('B', 1)) # Version 1

            # 2. Dims
            f.write(struct.pack('<I', model.input_dim))  # Use '<' for little-endian
            f.write(struct.pack('<I', model.hidden_dim))
            f.write(struct.pack('<I', model.output_dim))

            # --- Save Weights and Biases ---
            for weight_array in [model.W1, model.b1, model.W2, model.b2]:
                shape = weight_array.shape
                # Write shape (rows, cols) - use uint32
                f.write(struct.pack('<I', shape[0]))
                f.write(struct.pack('<I', shape[1]))
                # Write array data as float64 bytes
                f.write(weight_array.astype(np.float64).tobytes())

        print(f"Model successfully saved to {filename}")

    except IOError as e:
        print(f"Error saving model: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")


# --- Main Training Loop ---
if __name__ == "__main__":
    print("Starting training...")
    if not os.path.isdir(IMAGES_DIR):
        print(f"Error: Image directory '{IMAGES_DIR}' not found.")
        print("Please create it and place training images inside.")
        exit(1)

    image_files = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Error: No images found in '{IMAGES_DIR}'.")
        exit(1)

    print(f"Found {len(image_files)} images for training.")

    # 1. Load and preprocess images into patches
    print("Extracting patches...")
    all_patches = []
    # Extract fewer patches per image to speed up demo
    patches_per_image = max(1, 500 // len(image_files))
    for img_path in image_files:
        img_arr = load_and_preprocess_image(img_path)
        if img_arr is not None:
            # Adjust num_patches based on image size if needed, or keep fixed
            num_to_extract = min(patches_per_image, (img_arr.shape[0]-PATCH_SIZE+1)*(img_arr.shape[1]-PATCH_SIZE+1))
            if num_to_extract > 0:
                 all_patches.extend(extract_random_patches(img_arr, PATCH_SIZE, num_to_extract))

    if not all_patches:
        print("Error: No patches could be extracted. Check image sizes and formats.")
        exit(1)

    # Convert to NumPy array for batching
    dataset = np.array(all_patches, dtype=np.float32)
    print(f"Total patches extracted: {len(dataset)}")
    np.random.shuffle(dataset) # Shuffle patches

    # 2. Initialize the model
    autoencoder = Autoencoder(INPUT_DIM, HIDDEN_DIM)
    print(f"Initialized Autoencoder: Input={INPUT_DIM}, Hidden={HIDDEN_DIM}")

    # 3. Training loop
    num_batches = len(dataset) // BATCH_SIZE
    print(f"Starting training for {EPOCHS} epochs with batch size {BATCH_SIZE}...")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        # Mini-batch training
        for i in range(0, len(dataset), BATCH_SIZE):
            X_batch = dataset[i : i + BATCH_SIZE]
            if len(X_batch) == 0:
                continue

            loss = autoencoder.train_step(X_batch, LEARNING_RATE)
            epoch_loss += loss * len(X_batch) # Weighted average

            # Print progress
            batch_num = i // BATCH_SIZE + 1
            if batch_num % 5 == 0 or batch_num == num_batches:
                 print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {batch_num}/{num_batches}, Batch Loss: {loss:.6f}", end='\r')


        avg_epoch_loss = epoch_loss / len(dataset)
        print(f"\nEpoch {epoch+1}/{EPOCHS} completed. Average Training Loss: {avg_epoch_loss:.6f}")


    # 4. Save the trained model
    save_model(autoencoder, MODEL_FILENAME)

    print("Training finished.")
