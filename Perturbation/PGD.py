# PGD Attack
def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    # Start with the original image
    perturbed_images = images.clone().detach().to(device)
    perturbed_images.requires_grad = True

    for i in range(num_iter):
        # Forward pass and loss calculation
        output = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(output, labels)

        # Backprop to get gradients
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_images.grad.data

        # Update the image by a small step in the direction of the gradient
        perturbed_images = perturbed_images + alpha * data_grad.sign()

        # Clip to maintain perturbation constraint and pixel range [0,1]
        perturbed_images = torch.clamp(perturbed_images, images - epsilon, images + epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        # Detach gradients for the next iteration
        perturbed_images = perturbed_images.detach()
        perturbed_images.requires_grad = True

    return perturbed_images

# Example Usage (Assume model, data, and labels are prepared)
model.eval()
epsilon = 0.1  # Attack budget
alpha = 0.01  # Step size
num_iter = 40  # Number of iterations

perturbed_data = pgd_attack(model, data, target, epsilon, alpha, num_iter)

# Evaluate model on perturbed data
output = model(perturbed_data)
final_pred = output.max(1, keepdim=True)[1]
print(f"Prediction after attack: {final_pred.item()}, True Label: {target.item()}")
