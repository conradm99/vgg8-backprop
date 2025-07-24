import torch
import numpy as np
from vgg8 import VGG8  # replace with your actual filename if needed

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = VGG8().to(device)
model.eval()  # Set to eval mode (no dropout, no batchnorm updates)

# Optionally load trained weights if you have them
# model.load_state_dict(torch.load("vgg8_trained_model.pth", map_location=device))

# Generate a dummy QPSK signal for inference
def generate_qpsk_signal(num_samples=1024):
    """
    Generates a synthetic QPSK signal with I and Q channels.
    Returns shape: (2, num_samples)
    """
    symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])
    indices = np.random.randint(0, 4, num_samples)
    signal = symbols[indices]
    I = np.real(signal)
    Q = np.imag(signal)
    return np.vstack((I, Q))

# Create dummy input data (batch size = 1, 2 channels, 1024 samples)
qpsk_signal = generate_qpsk_signal(1024)
input_tensor = torch.tensor(qpsk_signal, dtype=torch.float32).unsqueeze(0).to(device)  # shape (1, 2, 1024)

# Run inference
with torch.no_grad():
    output = model(input_tensor)

# Process results
predicted_class_index = torch.argmax(output, dim=1).item()
predicted_class_name = model.classes[predicted_class_index]

print(f"Predicted class index: {predicted_class_index}")
print(f"Predicted modulation type: {predicted_class_name}")