# screen_detector.py
import numpy as np
import time
import struct
import ctypes
import ctypes.wintypes as wintypes # Use wintypes for convenience on Windows
from PIL import Image, ImageDraw, ImageGrab # Pillow for Image I/O and drawing

# --- Configuration ---
MODEL_FILENAME = 'model.jack'
PATCH_SIZE = 16  # Must match training
DETECTION_THRESHOLD = 0.02  # Lower reconstruction error means better match (adjust based on training)
NMS_THRESHOLD = 0.1 # IoU threshold for Non-Maximum Suppression
SCAN_STRIDE = 8 # How many pixels to jump between patches (lower = denser, slower)

# --- Activation Function (needed for forward pass) ---
def sigmoid(x):
    epsilon = 1e-8
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped) + epsilon)

# --- Loss Function (only MSE needed for error calc) ---
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# --- Model Deserialization (.jack format loader) ---
def load_model(filename):
    """Loads the model weights and architecture from a .jack file."""
    print(f"Loading model from {filename}...")
    weights = {}
    dims = {}
    try:
        with open(filename, 'rb') as f:
            # 1. Magic Bytes & Version
            magic = f.read(4)
            if magic != b'JACK':
                raise ValueError("Not a valid .jack file (magic bytes mismatch)")
            version = struct.unpack('B', f.read(1))[0]
            if version != 1:
                raise ValueError(f"Unsupported .jack version: {version}")

            # 2. Dims
            dims['input'] = struct.unpack('<I', f.read(4))[0]
            dims['hidden'] = struct.unpack('<I', f.read(4))[0]
            dims['output'] = struct.unpack('<I', f.read(4))[0]
            print(f"  Model Dims: Input={dims['input']}, Hidden={dims['hidden']}, Output={dims['output']}")

            # --- Load Weights and Biases ---
            for name in ['W1', 'b1', 'W2', 'b2']:
                # Read shape (rows, cols)
                rows = struct.unpack('<I', f.read(4))[0]
                cols = struct.unpack('<I', f.read(4))[0]
                shape = (rows, cols)
                # Calculate number of elements and bytes to read (float64 = 8 bytes)
                num_elements = rows * cols
                num_bytes = num_elements * 8
                # Read data and reshape
                data_bytes = f.read(num_bytes)
                weights[name] = np.frombuffer(data_bytes, dtype=np.float64).reshape(shape)
                print(f"  Loaded {name} with shape {shape}")

        print(f"Model successfully loaded from {filename}")
        return weights, dims

    except FileNotFoundError:
        print(f"Error: Model file '{filename}' not found.")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# --- Forward Pass Implementation ---
def model_forward_pass(X, weights):
    """Performs a forward pass using loaded weights."""
    W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']

    # Input to Hidden
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    # Hidden to Output
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2) # Reconstructed output
    return a2

# --- Screen Capture (Windows Specific using ctypes) ---
# NOTE: This requires Pillow to convert the raw bitmap data.
#       Error handling is minimal for brevity.

# Necessary Windows API functions and constants
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# Get screen dimensions
screen_width = user32.GetSystemMetrics(0) # SM_CXSCREEN
screen_height = user32.GetSystemMetrics(1) # SM_CYSCREEN

# Create Device Contexts (DC)
hwnd = user32.GetDesktopWindow()
screen_dc = user32.GetDC(hwnd)      # DC for the entire screen
mem_dc = gdi32.CreateCompatibleDC(screen_dc) # Memory DC for bitmap

# Create a compatible bitmap
# bitmap_info = BITMAPINFO() # Structure would normally be defined here
# ... (Setup BITMAPINFO structure - complex)

# For simplicity with Pillow, let's use ImageGrab which wraps this logic
def capture_screen():
    """Captures the screen using Pillow's ImageGrab (which uses OS APIs)."""
    try:
        # bbox = (0, 0, screen_width, screen_height) # Define area if needed
        screenshot = ImageGrab.grab()#bbox=bbox)
        return screenshot # Returns a Pillow Image object
    except Exception as e:
        print(f"Error capturing screen: {e}")
        # Fallback or error handling could go here
        # For non-Windows, ImageGrab might use different backends (like scrot)
        # or fail if dependencies are missing.
        return None

# --- Detection Helper ---
def preprocess_patch(patch_img):
    """Converts patch to grayscale, normalizes, flattens."""
    img_arr = np.array(patch_img.convert('L'), dtype=np.float32) / 255.0
    return img_arr.flatten()

# --- Non-Maximum Suppression (NMS) ---
def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two boxes."""
    # box format: [x1, y1, x2, y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    iou = inter_area / union_area
    return iou

def non_max_suppression(boxes, scores, iou_threshold):
    """Applies Non-Maximum Suppression."""
    if not boxes:
        return []

    # Sort boxes by score (lower error is better score here)
    order = np.argsort(scores) # Ascending order for error scores

    keep = []
    while order.size > 0:
        i = order[0] # Index of box with lowest error
        keep.append(i)

        if order.size == 1:
            break

        # Calculate IoU of the current box with all remaining boxes
        current_box = boxes[i]
        remaining_indices = order[1:]
        remaining_boxes = [boxes[j] for j in remaining_indices]

        ious = np.array([calculate_iou(current_box, box) for box in remaining_boxes])

        # Find indices of boxes with IoU greater than threshold
        indices_to_remove = remaining_indices[ious > iou_threshold]

        # Remove overlapping boxes (keep only the one with lowest error)
        order = np.setdiff1d(order, np.concatenate(([i], indices_to_remove)))


    return [boxes[i] for i in keep]


# --- Main Detection Loop ---
if __name__ == "__main__":
    # 1. Load the trained model
    weights, dims = load_model(MODEL_FILENAME)
    if not weights:
        exit(1)

    input_dim = dims['input']
    expected_patch_pixels = PATCH_SIZE * PATCH_SIZE
    if input_dim != expected_patch_pixels:
        print(f"Error: Model input dimension ({input_dim}) does not match expected patch size ({expected_patch_pixels}).")
        exit(1)

    print("Starting screen detection loop (Press Ctrl+C to stop)...")
    last_time = time.time()

    try:
        while True:
            # 2. Capture screen
            screen_img = capture_screen()
            if screen_img is None:
                print("Failed to capture screen, retrying...")
                time.sleep(1)
                continue

            screen_width, screen_height = screen_img.size
            draw = ImageDraw.Draw(screen_img) # For drawing boxes later

            detected_boxes = []
            detection_scores = [] # Store reconstruction error as score

            # 3. Sliding window across the screen
            for y in range(0, screen_height - PATCH_SIZE + 1, SCAN_STRIDE):
                for x in range(0, screen_width - PATCH_SIZE + 1, SCAN_STRIDE):
                    # Extract patch
                    box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
                    patch_img = screen_img.crop(box)

                    # Preprocess
                    patch_flat = preprocess_patch(patch_img)
                    patch_batch = patch_flat.reshape(1, -1) # Reshape for batch dimension

                    # 4. Run inference (forward pass)
                    reconstructed_patch = model_forward_pass(patch_batch, weights)

                    # 5. Calculate reconstruction error
                    error = mean_squared_error(patch_batch, reconstructed_patch)

                    # 6. Check threshold
                    if error < DETECTION_THRESHOLD:
                        # Store box coordinates [x1, y1, x2, y2] and score (error)
                        detected_boxes.append([x, y, x + PATCH_SIZE, y + PATCH_SIZE])
                        detection_scores.append(error)
                        # print(f"Detected pattern at ({x}, {y}) with error {error:.4f}") # Debugging

            # 7. Apply Non-Maximum Suppression
            if detected_boxes:
                final_boxes = non_max_suppression(detected_boxes, detection_scores, NMS_THRESHOLD)
                print(f"Raw detections: {len(detected_boxes)}, After NMS: {len(final_boxes)}")
            else:
                final_boxes = []


            # 8. Draw bounding boxes on the captured image
            for (x1, y1, x2, y2) in final_boxes:
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            # Draw FPS on image
            draw.text((10, 10), f"FPS: {fps:.2f}", fill="lime")


            # 9. Display the result
            # Pillow's show() is basic, opens in default image viewer.
            # For true real-time video, libraries like OpenCV are needed.
            # Alternatively, save frames to disk.
            screen_img.show(title="Screen Detection") # This might open many windows or be slow

            # Optional: Save frame instead of showing
            # screen_img.save(f"output_frame_{int(time.time())}.png")

            # Add a small delay to prevent overwhelming the system and allow viewing
            time.sleep(0.1)


    except KeyboardInterrupt:
        print("\nDetection stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during detection: {e}")
        import traceback
        traceback.print_exc()
