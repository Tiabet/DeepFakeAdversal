import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np
from face_ssd import build_ssd

net = build_ssd('test', 640, 2)  # Initialize DSFD model
net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth', map_location=torch.device('cpu')))
  # Load pre-trained weights
net.eval()  # Set the model to evaluation mode

def fgsm_attack(image, epsilon, data_grad):
    # Collect the sign of the gradients of the input
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Clip the values to be in the valid range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))  # Resize to model's expected input size
    image = image.astype(np.float32)
    image -= (104.0, 117.0, 123.0)  # Subtract mean values
    image = image.transpose(2, 0, 1)  # Convert to CHW format
    image = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension
    return image

# Load and preprocess the image
image_path = '04474.jpg'
image = preprocess_image(image_path)
image = Variable(image, requires_grad=True)

# Forward pass
output = net(image)

# Define a loss function (e.g., sum of confidence scores for the 'face' class)
loss = -output[1][:, 1].sum()  # Assuming '1' is the index for 'face' class

# Zero all existing gradients
net.zero_grad()

# Backward pass
loss.backward()

# Collect the gradients of the input image
data_grad = image.grad.data

# Set the epsilon value for FGSM
epsilon = 0.01

# Generate the perturbed image
perturbed_image = fgsm_attack(image, epsilon, data_grad)

# Forward pass with the perturbed image
output_perturbed = net(perturbed_image)

# Process the output as needed (e.g., apply non-maximum suppression, thresholding)
