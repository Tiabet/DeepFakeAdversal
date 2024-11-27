import torch
from torchattacks import PGD
from facenet_pytorch import MTCNN

# Step 1: Load MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True).to(device)


# Step 2: Define a Custom Loss
# Example: Maximize classification loss (1 - detection confidence)
def detection_loss(scores, target_confidence=0.0):

    # scores: confidence scores from MTCNN
    # target_confidence: confidence value to target (e.g., 0 for misdetection)
    return torch.mean((scores - target_confidence) ** 2)


# Step 3: Custom PGD Wrapper for MTCNN
class PGDMTCNN(PGD):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=4):
        super().__init__(model, eps=eps, alpha=alpha, steps=steps)
        self.model = model

    def forward(self, images, target_confidence=0.0):
        images = images.clone().detach().to(self.device)
        images.requires_grad = True

        for _ in range(self.steps):
            # Get MTCNN outputs
            boxes, probs = self.model.detect(images, landmarks=False)

            # Ensure valid outputs (skip invalid images)
            valid_probs = [p for p in probs if p is not None]
            if len(valid_probs) == 0:
                break

            # Compute loss for adversarial attack
            probs_tensor = torch.tensor(valid_probs, requires_grad=True).to(self.device)
            loss = detection_loss(probs_tensor, target_confidence)

            # Backpropagate to compute gradients
            self.model.zero_grad()
            loss.backward()
            grad = images.grad.data

            # Apply PGD update
            adv_images = images + self.alpha * grad.sign()
            eta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            images = torch.clamp(images + eta, min=0, max=1).detach_()
            images.requires_grad = True

        return images


# Step 4: Use PGD for MTCNN
pgd_attack = PGDMTCNN(mtcnn, eps=8 / 255, alpha=2 / 255, steps=4)

# Input image batch
images = torch.rand((1,128, 128,3)).to(device)  # Replace with real images
print(images.shape)
adv_images = pgd_attack(images)
