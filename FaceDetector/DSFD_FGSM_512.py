import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
import cv2
from tqdm import tqdm


# ==========================
# Feature Enhancement Module
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


# ====================
# Detection Head
# ====================
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors=6):
        super(DetectionHead, self).__init__()
        self.cls_layer = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=3, padding=1)
        self.reg_layer = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        cls_out = self.cls_layer(x)
        reg_out = self.reg_layer(x)
        return cls_out, reg_out


# ===========================
# Dual Shot Face Detector (DSFD)
# ===========================
class DSFDModel(nn.Module):
    def __init__(self):
        super(DSFDModel, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.backbone = vgg16.features[:-1]
        self.fem1 = FeatureEnhanceModule(512)
        self.fem2 = FeatureEnhanceModule(1024)
        self.head1 = DetectionHead(512)
        self.head2 = DetectionHead(1024)

    def forward(self, x):
        x = self.backbone(x)
        fem_out1 = self.fem1(x)
        cls1, reg1 = self.head1(fem_out1)
        fem_out2 = self.fem2(fem_out1)
        cls2, reg2 = self.head2(fem_out2)
        return (cls1, reg1), (cls2, reg2)


# ==========================
# Loss Function
# ==========================
class DSFDLoss(nn.Module):
    def __init__(self):
        super(DSFDLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.loc_loss = nn.SmoothL1Loss()

    def forward(self, predictions, targets):
        (cls_pred1, loc_pred1), (cls_pred2, loc_pred2) = predictions
        cls_target, loc_target = targets
        loss1 = self.cls_loss(cls_pred1, cls_target) + self.loc_loss(loc_pred1, loc_target)
        loss2 = self.cls_loss(cls_pred2, cls_target) + self.loc_loss(loc_pred2, loc_target)
        return loss1 + loss2


# ==========================
# FGSM Attack Function
# ==========================
def fgsm_attack(model, images, targets, epsilon=0.03):
    images.requires_grad = True
    predictions = model(images)
    loss_fn = DSFDLoss()
    loss = loss_fn(predictions, targets)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    adv_images = images + epsilon * data_grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images


# ==========================
# DataLoader for 512x512 Images
# ==========================
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        cls_target = torch.randint(0, 2, (6, 32, 32))
        loc_target = torch.randn(6, 4, 32, 32)
        return image, (cls_target, loc_target)


# ==========================
# Training Loop with FGSM
# ==========================
def train_model(image_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSFDModel().to(device)
    criterion = DSFDLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    dataset = CustomDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    epsilon = 0.03

    for epoch in range(5):
        model.train()
        total_loss = 0.0
        for images, targets in tqdm(dataloader):
            images, targets = images.to(device), (targets[0].to(device), targets[1].to(device))

            adv_images = fgsm_attack(model, images, targets, epsilon)
            optimizer.zero_grad()
            predictions = model(adv_images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# ==========================
# Main Function
# ==========================
if __name__ == "__main__":
    image_dir = "Irene"
    train_model(image_dir)
