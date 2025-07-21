import vgg8.py as vgg8
import torch 

def load_model_and_infer(weight_path, input_tensor, device='cpu'):
    """
    Loads the VGG8 model with pretrained weights and performs inference.

    Args:
        weight_path (str): Path to the .pt or .pth file containing the saved model weights.
        input_tensor (torch.Tensor): The input tensor of shape (batch_size, 2, 1024).
        device (str): 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: Output logits from the model (before softmax).
    """

    # Make sure the device is available
    device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')

    # Instantiate and load the model
    model = VGG8().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Send input tensor to the same device as the model
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    return output