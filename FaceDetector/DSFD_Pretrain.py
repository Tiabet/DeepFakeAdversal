import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from resnet import resnet152


# ==========================
# Load DSFD Model with ResNet152 Backbone
# ==========================
class FeatureEnhanceModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureEnhanceModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 3, kernel_size=1)
        self.dilation1 = nn.Conv2d(in_channels // 3, in_channels // 3, kernel_size=3, dilation=3, padding=3)
        self.dilation2 = nn.Conv2d(in_channels // 3, in_channels // 3, kernel_size=3, dilation=2, padding=2)
        self.dilation3 = nn.Conv2d(in_channels // 3, in_channels // 3, kernel_size=3, dilation=1, padding=1)
        self.concat = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        d1 = self.dilation1(x1)
        d2 = self.dilation2(x1)
        d3 = self.dilation3(x1)
        out = torch.cat([d1, d2, d3], dim=1)
        return self.concat(out)


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors=6):
        super(DetectionHead, self).__init__()
        self.cls_layer = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=3, padding=1)
        self.reg_layer = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        cls_out = self.cls_layer(x)
        reg_out = self.reg_layer(x)
        return cls_out, reg_out


class DSFDModel(nn.Module):
    def __init__(self):
        super(DSFDModel, self).__init__()
        resnet = resnet152(pretrained=False)

        # Define the backbone layers explicitly
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Additional layers as required by DSFD
        self.fem1 = FeatureEnhanceModule(2048)
        self.fem2 = FeatureEnhanceModule(2048)
        self.head1 = DetectionHead(2048)
        self.head2 = DetectionHead(2048)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print(f"Shape after backbone: {x.shape}")

        fem_out1 = self.fem1(x)
        print(f"Shape after fem1: {fem_out1.shape}")
        cls1, reg1 = self.head1(fem_out1)
        fem_out2 = self.fem2(fem_out1)
        cls2, reg2 = self.head2(fem_out2)
        return (cls1, reg1), (cls2, reg2)


# Load the pretrained DSFD model weights
def load_pretrained_model(model_path):
    model = DSFDModel()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  # Load pretrained weights with strict=False
    model.eval()
    return model


# ==========================
# FGSM Attack Function
# ==========================
def fgsm_attack(model, image, epsilon=0.03):
    image.requires_grad = True
    predictions = model(image)

    # Assume cls_target and loc_target are dummy placeholders for gradient calculation
    cls_target = torch.randint(0, 2, predictions[0][0].shape).to(image.device)
    loc_target = torch.randn(predictions[0][1].shape).to(image.device)

    loss_fn = nn.CrossEntropyLoss()
    cls_loss = loss_fn(predictions[0][0], cls_target) + loss_fn(predictions[1][0], cls_target)
    loc_loss = nn.SmoothL1Loss()(predictions[0][1], loc_target) + nn.SmoothL1Loss()(predictions[1][1], loc_target)
    loss = cls_loss + loc_loss

    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    adv_image = image + epsilon * data_grad.sign()
    adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image


# ==========================
# Adversarial Image Generation for Multiple Images
# ==========================
def generate_adversarial_images(model, input_dir, output_dir, epsilon=0.03):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in tqdm(os.listdir(input_dir)):
        image_path = os.path.join(input_dir, image_name)
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor.requires_grad = True

        # Generate adversarial image
        adv_image_tensor = fgsm_attack(model, image_tensor, epsilon)

        # Save adversarial image
        adv_image_np = adv_image_tensor.squeeze().detach().cpu().numpy()
        adv_image_np = np.transpose(adv_image_np, (1, 2, 0)) * 255
        adv_image_np = adv_image_np.astype(np.uint8)

        adv_image = Image.fromarray(adv_image_np)
        adv_image.save(os.path.join(output_dir, image_name))
        print(f"Adversarial image saved to {output_dir}/{image_name}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    pretrained_model_path = "WIDERFace_DSFD_RES152.pth"  # Replace with your pretrained model path
    input_dir = "Irene"  # Directory with original images
    output_dir = "Irene_adversarial"  # Directory to save adversarial images

    model = load_pretrained_model(pretrained_model_path)
    print("Model loaded successfully!")
    generate_adversarial_images(model, input_dir, output_dir, epsilon=0.03)
