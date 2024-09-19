import torch
import torch.nn.functional as F

def multi_scale_adversarial_attack(image, face_detector, epsilon=8, max_iterations=10, lambda1=1.0, lambda2=1.0):
    """
    Multi-scale feature-level adversarial attack.

    Args:
    - image (torch.Tensor): The input image to attack.
    - face_detector (nn.Module): The pre-trained face detection model.
    - epsilon (float): Maximum allowed perturbation per pixel.
    - max_iterations (int): Maximum number of iterations for the attack.
    - lambda1 (float): Weight for the error loss.
    - lambda2 (float): Weight for the key loss.

    Returns:
    - adv_image (torch.Tensor): The adversarially perturbed image.
    """
    
    # Step 1: Extract multi-scale intermediate features from the detector for the original image
    features_original = face_detector.get_multi_scale_features(image)  # Function to extract multi-scale features
    
    # Step 2: Initialize adversarial image as a clone of the original
    adv_image = image.clone().detach().requires_grad_(True)
    
    # Optimization loop
    for i in range(max_iterations):
        # Forward pass: Extract multi-scale features from the adversarial image
        features_adv = face_detector.get_multi_scale_features(adv_image)
        
        # Step 3: Calculate multi-scale loss (error loss and key loss for each layer)
        loss_error = compute_multi_scale_error_loss(features_original, features_adv)
        loss_key = compute_multi_scale_key_loss(features_original, features_adv)
        
        # Combine the losses
        total_loss = lambda1 * loss_error + lambda2 * loss_key
        
        # Step 4: Backpropagate to compute gradients
        face_detector.zero_grad()
        total_loss.backward()
        
        # Step 5: Update the adversarial image
        adv_image = update_adversarial_image(adv_image, epsilon)
        
        # Detach the image to avoid gradient accumulation
        adv_image = adv_image.detach().requires_grad_(True)
    
    return adv_image


def compute_multi_scale_error_loss(features_original, features_adv):
    """
    Compute the error loss for multiple scales (layers) using cosine similarity.

    Args:
    - features_original (list of torch.Tensor): Original features from different layers.
    - features_adv (list of torch.Tensor): Adversarial features from different layers.

    Returns:
    - total_loss (torch.Tensor): The total error loss over multiple scales.
    """
    total_loss = 0
    for f_orig, f_adv in zip(features_original, features_adv):
        cos_sim = F.cosine_similarity(f_orig, f_adv, dim=1)
        total_loss += 1 - cos_sim.mean()  # Maximize the difference between original and adversarial features
    return total_loss


def compute_multi_scale_key_loss(features_original, features_adv):
    """
    Compute the key loss for multiple scales, focusing on attacking important feature elements.

    Args:
    - features_original (list of torch.Tensor): Original features from different layers.
    - features_adv (list of torch.Tensor): Adversarial features from different layers.

    Returns:
    - total_loss (torch.Tensor): The total key loss over multiple scales.
    """
    total_loss = 0
    for f_orig, f_adv in zip(features_original, features_adv):
        # Focus on elements of high importance by computing the gradient-based key elements
        key_elements = torch.abs(f_orig - f_adv)
        total_loss += key_elements.mean()  # Attack high-importance regions in the feature map
    return total_loss


def update_adversarial_image(adv_image, epsilon):
    """
    Update the adversarial image by applying the gradient (FGSM-style) within epsilon constraint.

    Args:
    - adv_image (torch.Tensor): The adversarial image to be updated.
    - epsilon (float): Maximum perturbation allowed.

    Returns:
    - adv_image (torch.Tensor): The updated adversarial image.
    """
    step_size = 0.01  # The step size can be adjusted based on experiments
    
    # Apply the Fast Gradient Sign Method (FGSM) to update the adversarial image
    adv_image = adv_image + step_size * adv_image.grad.sign()
    
    # Clip the perturbation to ensure it remains within epsilon bounds
    perturbation = torch.clamp(adv_image - adv_image.detach(), -epsilon, epsilon)
    adv_image = torch.clamp(adv_image + perturbation, 0, 1)  # Keep pixel values in [0, 1]
    
    return adv_image
