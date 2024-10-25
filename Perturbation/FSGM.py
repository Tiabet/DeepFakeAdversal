import torch
import torch.nn as nn
import torch.optim as optim

# FGSM Attack
def fgsm_attack(image, epsilon, gradient):
    # Perturb the input in the direction of the gradient's sign
    perturbed_image = image + epsilon * gradient.sign()
    # Clip to maintain valid pixel range [0,1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Example Usage (Assume model, data, and labels are prepared)
model.eval()  # Set model to evaluation mode
data, target = data.to(device), target.to(device)
data.requires_grad = True  # Enable gradient tracking

# Forward pass
output = model(data)
loss = nn.CrossEntropyLoss()(output, target)

# Backprop to get gradients
model.zero_grad()
loss.backward()
data_grad = data.grad.data

# Apply FGSM attack
epsilon = 0.1  # Attack strength
perturbed_data = fgsm_attack(data, epsilon, data_grad)

# Evaluate model on perturbed data
output = model(perturbed_data)
final_pred = output.max(1, keepdim=True)[1]
print(f"Prediction after attack: {final_pred.item()}, True Label: {target.item()}")
