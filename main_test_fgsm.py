import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def fgsm_attack(image, epsilon, data_grad):
    """Generate adversarial image using FGSM."""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def load_image(image_path):
    """Load an image and preprocess it."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def fgsm_example(image_path, epsilon=0.02):
    """Perform FGSM attack on a pre-trained ResNet model."""
    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Load and preprocess the image
    image = load_image(image_path)
    image.requires_grad = True

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Forward pass
    output = model(image)
    _, initial_prediction = torch.max(output, 1)

    print(f"Original Prediction: {initial_prediction.item()}")

    # Calculate loss and perform backward pass
    loss = criterion(output, initial_prediction)
    model.zero_grad()
    loss.backward()

    # Check if gradients are computed
    if image.grad is None:
        print("[Error] Gradients were not computed for the image tensor.")
        return

    # Generate adversarial image
    data_grad = image.grad.data
    perturbed_image = fgsm_attack(image, epsilon, data_grad)

    # Forward pass on perturbed image
    output_perturbed = model(perturbed_image)
    _, perturbed_prediction = torch.max(output_perturbed, 1)

    print(f"Perturbed Prediction: {perturbed_prediction.item()}")

    # Convert perturbed image to numpy and plot
    perturbed_image_np = perturbed_image.squeeze().detach().numpy().transpose(1, 2, 0)
    plt.imshow(perturbed_image_np)
    plt.title(f"Perturbed Image (Epsilon = {epsilon})")
    plt.show()

if __name__ == '__main__':
    image_path = 'dsfd_modified/unnamed.jpg'  # Change to your image path
    fgsm_example(image_path, epsilon=0.02)
