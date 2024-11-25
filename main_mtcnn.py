import torch
from PIL import Image
from MTCNN.mtcnn import MTCNN, fixed_image_standardization
from torchvision import transforms, datasets
import numpy as np
import os

def fgsm_attack(image, epsilon, gradient):
    """Applies FGSM attack by adding perturbation to the image."""
    perturbation = epsilon * gradient.sign()
    perturbed_image = image + perturbation
    return torch.clamp(perturbed_image, 0, 1)

def main():
    # Check for dataset existence
    if not os.path.exists('test_images'):
        print("Dataset directory 'test_images' does not exist.")
        return

    # Initialize MTCNN
    mtcnn = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Transformation for resizing and normalization
    trans = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    # Load dataset
    dataset = datasets.ImageFolder('test_images', transform=trans)
    dataset.idx_to_class = {k: v for v, k in dataset.class_to_idx.items()}

    for img, idx in dataset:
        name = dataset.idx_to_class[idx]
        print(f"Processing image for {name}.")

        print(img.shape)
        # Convert image to tensor and enable gradient
        image_tensor = img.to(mtcnn.device)
        image_tensor.requires_grad = True



        # Debugging: Check the shape and range of image_tensor
        print(f"Input tensor shape: {image_tensor.shape}")
        print(f"Input tensor range: {image_tensor.min().item()} to {image_tensor.max().item()}")

        # Detect faces
        boxes, probs, image_tensor = mtcnn.detect(image_tensor, enable_grad=True)

        if boxes is None or len(boxes) == 0:
            print("No face detected.")
            continue

        print("Detected Boxes:", boxes)
        print("Detection Probabilities:", probs)

        # Compute gradients
        probs = torch.tensor(np.array(probs, dtype=np.float32), requires_grad=True).to(image_tensor.device)
        probs[0].backward()
        gradient = image_tensor.grad

        if gradient is None:
            print("Gradient computation failed.")
            continue

        print(f"Gradient shape: {gradient.shape}, range: {gradient.min().item()} to {gradient.max().item()}")

        # Perform FGSM attack
        epsilon = 0.01
        perturbed_image = fgsm_attack(image_tensor, epsilon, gradient)
        print(f"Perturbed image range: {perturbed_image.min().item()} to {perturbed_image.max().item()}")

        # Re-run detection on perturbed image
        perturbed_boxes, perturbed_probs = mtcnn.detect(perturbed_image, enable_grad=True)
        if perturbed_boxes is None or len(perturbed_boxes) == 0:
            print("No face detected in the perturbed image.")
            continue

        print("Perturbed Boxes:", perturbed_boxes)
        print("Perturbed Detection Probabilities:", perturbed_probs)

if __name__ == "__main__":
    main()
