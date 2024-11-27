from mtcnn import MTCNN
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os

# Define the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Initialize MTCNN
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.5, 0.6, 0.6],  # Lowered thresholds for debugging
    factor=0.709, post_process=True,
    select_largest=True,
    keep_all=False,  # Set to True to get all detections
    device=device
)

def collate_fn(x):
    return x[0]

# Initialize dataset
dataset = datasets.ImageFolder('../data/test_images', transform=transforms.ToTensor())
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)  # workers=0 for Windows compatibility

aligned = []
names = []
adversarial_images = []

for idx, (x, y) in enumerate(loader):
    # Move to device and set requires_grad=True
    x_tensor = x.to(device).unsqueeze(0).requires_grad_(True)  # Shape: [1, C, H, W]
    print(f"Image {idx}: Tensor shape: {x_tensor.shape}, range: [{x_tensor.min()}, {x_tensor.max()}]")

    # Pass through MTCNN
    # Ensure x_tensor is in [0, 255] as expected by MTCNN
    img_input = x_tensor.permute(0, 2, 3, 1) * 255
    faces, probs = mtcnn(img_input, return_prob=True)

    if faces is not None and len(faces) > 0 and faces[0] is not None:
        # Assuming batch_size=1, get the first detection
        if mtcnn.keep_all:
            # Handle multiple detections; for simplicity, choose the first one
            selected_face = faces[0][0]  # [C, H, W]
            selected_prob = probs[0][0]  # Scalar
        else:
            selected_face = faces[0]  # [C, H, W]
            selected_prob = probs[0]  # Scalar

        print(f"Face shape: {selected_face.shape}, prob: {selected_prob}")

        # Define loss: maximize the detection probability
        loss = selected_prob

        # Backpropagate to compute gradients
        loss.backward()

        # Ensure gradients are available
        if x_tensor.grad is not None:
            # FGSM attack: Add perturbation to the input image
            epsilon = 0.01  # Small perturbation value
            perturbed_image = x_tensor + epsilon * x_tensor.grad.sign()

            # Ensure the perturbed image stays in valid range
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

            adversarial_images.append(perturbed_image.detach())
            aligned.append(selected_face.detach())
            names.append(dataset.idx_to_class[y])

            print(adversarial_images)
        else:
            print("Gradient not computed for x_tensor.")
    else:
        print("No face detected.")

    # Zero gradients for the next iteration
    x_tensor.grad.zero_()

# Optionally, save adversarial images and aligned faces
for i, (adv_img, face, name) in enumerate(zip(adversarial_images, aligned, names)):
    # Convert tensors to PIL Images for saving
    adv_img_pil = transforms.ToPILImage()(adv_img.squeeze(0).cpu())
    face_pil = transforms.ToPILImage()(face.permute(2, 0, 1).cpu())

    # Define save paths
    adv_save_path = f'../data/adversarial_images/adv_{i}.png'
    face_save_path = f'../data/aligned_faces/{name}_{i}.png'

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(adv_save_path) + "/", exist_ok=True)
    os.makedirs(os.path.dirname(face_save_path) + "/", exist_ok=True)

    # Save images
    adv_img_pil.save(adv_save_path)
    face_pil.save(face_save_path)

    print(f"Saved adversarial image to {adv_save_path} and aligned face to {face_save_path}")
