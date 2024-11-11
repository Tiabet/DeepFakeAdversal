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

        # Reduce channels from 2048 to 1024
        self.reduce_channels = nn.Conv2d(2048, 1024, kernel_size=1, stride=1)

        # Additional feature extraction layers (dual shot layers)
        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        ])

        # Define anchor boxes count for each feature map layer
        self.num_anchors = [4, 6]  # Assuming 4 and 6 anchors for each feature map

        # Confidence and localization layers
        self.confidence_layers = nn.ModuleList([
            nn.Conv2d(1024, self.num_anchors[0] * 21, kernel_size=3, padding=1),
            nn.Conv2d(512, self.num_anchors[1] * 21, kernel_size=3, padding=1)
        ])

        self.localization_layers = nn.ModuleList([
            nn.Conv2d(1024, self.num_anchors[0] * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, self.num_anchors[1] * 4, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        features = []

        # Backbone pass
        x = self.backbone(x)
        x = F.relu(self.reduce_channels(x))
        features.append(x)

        # Additional layers
        for layer in self.extras:
            x = F.relu(layer(x))
            features.append(x)

        # Confidence and localization predictions
        confs, locs = [], []
        for (x, conf_layer, loc_layer, num_anchors) in zip(
                features, self.confidence_layers, self.localization_layers, self.num_anchors):
            # Confidence predictions
            conf = conf_layer(x).permute(0, 2, 3, 1).contiguous()
            confs.append(conf.view(conf.size(0), -1))

            # Localization predictions
            loc = loc_layer(x).permute(0, 2, 3, 1).contiguous()
            locs.append(loc.view(loc.size(0), -1))

        confs = torch.cat(confs, 1)
        locs = torch.cat(locs, 1)

        return confs, locs


# Test model instantiation
if __name__ == "__main__":
    model = DSFD()
    print(model)
    dummy_input = torch.randn(1, 3, 300, 300)
    confidences, locations = model(dummy_input)
    print("Confidences:", confidences.shape)
    print("Locations:", locations.shape)
