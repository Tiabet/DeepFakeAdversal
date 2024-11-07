import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DSFD(nn.Module):
    def __init__(self):
        super(DSFD, self).__init__()
        base_model = models.resnet50(pretrained=True)

        # Backbone: Use ResNet-50 layers as the feature extractor
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])

        # Additional feature extraction layers (dual shot layers)
        self.extras = nn.ModuleList([
            nn.Conv2d(2048, 512, kernel_size=1, stride=1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        ])

        # Confidence and localization layers
        self.confidence_layers = nn.ModuleList([
            nn.Conv2d(1024, 21, kernel_size=3, padding=1),
            nn.Conv2d(512, 21, kernel_size=3, padding=1)
        ])

        self.localization_layers = nn.ModuleList([
            nn.Conv2d(1024, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        features = []

        # Backbone pass
        x = self.backbone(x)
        features.append(x)

        # Additional layers
        for layer in self.extras:
            x = F.relu(layer(x))
            features.append(x)

        # Confidence and localization predictions
        confs, locs = [], []
        for (x, conf_layer, loc_layer) in zip(features, self.confidence_layers, self.localization_layers):
            confs.append(conf_layer(x).permute(0, 2, 3, 1).contiguous())
            locs.append(loc_layer(x).permute(0, 2, 3, 1).contiguous())

        confs = torch.cat([c.view(c.size(0), -1) for c in confs], 1)
        locs = torch.cat([l.view(l.size(0), -1) for l in locs], 1)

        return confs, locs


# Test model instantiation
if __name__ == "__main__":
    model = DSFD()
    print(model)
    dummy_input = torch.randn(1, 3, 300, 300)
    confidences, locations = model(dummy_input)
    print("Confidences:", confidences.shape)
    print("Locations:", locations.shape)
