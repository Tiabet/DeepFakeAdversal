import torch
import torch.nn.functional as F

def input_diversity(x, prob=0.5, resize_low=224):
    """Applies input diversity with a probability."""
    if torch.rand(1).item() > prob:
        return x
    rnd = torch.randint(resize_low, x.shape[-1], (1,), dtype=torch.int32).item()
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = x.shape[-2] - rnd
    w_rem = x.shape[-1] - rnd
    pad_top = torch.randint(0, h_rem + 1, (1,), dtype=torch.int32).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem + 1, (1,), dtype=torch.int32).item()
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    return padded

def di_fgsm(model, images, labels, epsilon, alpha, num_iter, prob=0.5, resize_low=224):
    """Diverse Inputs FGSM attack."""
    images = images.clone().detach().to(torch.float32)
    labels = labels.clone().detach()
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon).to(images.device)
    delta.requires_grad = True

    for i in range(num_iter):
        outputs = model(input_diversity(images + delta, prob, resize_low))
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        grad_sign = delta.grad.sign()
        delta = delta + alpha * grad_sign
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = torch.clamp(images + delta, 0, 1) - images  # Ensure valid pixel range
        delta.grad.zero_()

    return images + delta.detach()

# Usage example
# model: Your pretrained model
# images: Input tensor (batch_size, channels, height, width)
# labels: True labels tensor (batch_size)
# epsilon: Maximum perturbation
# alpha: Step size
# num_iter: Number of iterations

# perturbed_images = di_fgsm(model, images, labels, epsilon=0.03, alpha=0.005, num_iter=10)
