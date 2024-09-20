def get_multi_scale_features(self, x):
    features = []
    # Extract features from different layers in the network
    for layer in self.model.children():
        x = layer(x)
        if isinstance(layer, SomeLayerTypeYouCareAbout):  # E.g., Conv2d, ReLU
            features.append(x)
    return features
