import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import os
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
        self.backbone = vgg16.features[:-1]  # Remove fully connected layers

        # Feature Enhance Module for Dual Shots
        self.fem1 = FeatureEnhanceModule(512)
        self.fem2 = FeatureEnhanceModule(1024)

        # Detection heads for first and second shots
        self.head1 = DetectionHead(512)
        self.head2 = DetectionHead(1024)

    def forward(self, x):
        x = self.backbone(x)
        # First Shot
        fem_out1 = self.fem1(x)
        cls1, reg1 = self.head1(fem_out1)

        # Second Shot
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
# Anchor Generation
# ==========================
def generate_anchors(scales, ratios, feature_map_size, stride):
    anchors = []
    for i in range(feature_map_size):
        for j in range(feature_map_size):
            cx, cy = j * stride, i * stride
            for scale in scales:
                for ratio in ratios:
                    w = scale * ratio[0]
                    h = scale * ratio[1]
                    anchors.append([cx, cy, w, h])
    return torch.tensor(anchors)


# ==========================
# DataLoader (Dummy Example)
# ==========================
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy images and labels
        image = np.random.randint(0, 255, (3, 640, 640), dtype=np.uint8)
        image = self.transform(image)
        cls_target = torch.randint(0, 2, (6, 40, 40))
        loc_target = torch.randn(6, 4, 40, 40)
        return image, (cls_target, loc_target)


# ==========================
# Training Loop
# ==========================
def train_model():
    model = DSFDModel().to(device)
    criterion = DSFDLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(5):
        model.train()
        total_loss = 0.0
        for images, targets in tqdm(dataloader):
            images, targets = images.to(device), (targets[0].to(device), targets[1].to(device))
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# ==========================
# Inference
# ==========================
def inference(model, image_path):
    image = cv2.imread(image_path)
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    print("Inference complete!")


# ==========================
# Main Function
# ==========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model()
