import torch.nn as nn
import torch
from torchvision import models, transforms

class EffNetV2640RGB(nn.Module):
    def __init__(self):
        super(EffNetV2640RGB, self).__init__()
        self.effnetRGB = models.efficientnet_v2_s(weights='DEFAULT')

        # Modify the first convolution layer to accept 1 input channel
        self.effnetRGB.features[0] = nn.Conv2d(3, self.effnetRGB.features[0][0].out_channels, kernel_size=(7, 7),
                                               stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the fully connected layer with two separate output layers
        num_featuresRGB = self.effnetRGB.classifier[1].in_features
        self.effnetRGB.classifier = nn.Identity()

        # Create two linear layers for the two output vectors
        self.fc1 = nn.Linear(num_featuresRGB, 1000)
        self.fc2 = nn.Linear(1000, 1)

        if torch.cuda.is_available():
            self.effnetRGB = self.effnetRGB.to('cuda')
            self.fc1 = self.fc1.to('cuda')
            self.fc2 = self.fc2.to('cuda')
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def forward(self,x):
        preprocessRGB = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 640x640
            transforms.Normalize(  # Normalize the image
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        x_RGB = x[:, :3, :, :]
        x_RGB = preprocessRGB(x_RGB).to(self.device)
        features_RGB = self.effnetRGB(x_RGB)
        new_featuresRGB = self.fc1(features_RGB)
        output = self.fc2(new_featuresRGB)
        return output

class EffNetV2640D(nn.Module):
    def __init__(self):
        super(EffNetV2640D, self).__init__()
        self.effnetD = models.efficientnet_v2_s(weights='DEFAULT')
        self.effnetD.features[0] = nn.Conv2d(1, self.effnetD.features[0][0].out_channels, kernel_size=(7, 7),
                                             stride=(2, 2), padding=(3, 3), bias=False)
        num_featuresD = self.effnetD.classifier[1].in_features
        self.effnetD.classifier = nn.Identity()  # Remove the fully connected layer

        self.fc1 = nn.Linear(num_featuresD, 1000)
        self.fc2 = nn.Linear(1000, 1)

        if torch.cuda.is_available():
            self.effnetD = self.effnetD.to('cuda')
            self.fc1 = self.fc1.to('cuda')
            self.fc2 = self.fc2.to('cuda')
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def forward(self, x):
        preprocessD = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 640x640
            transforms.Normalize(  # Normalize the image
                mean=[0.5],
                std=[0.5]
            )
        ])
        x_D = x[:, 3, :, :].unsqueeze(1)
        x_D = preprocessD(x_D).to(self.device)
        features_D = self.effnetD(x_D)
        new_featuresD = self.fc1(features_D)
        output = self.fc2(new_featuresD)
        return output

class EffNetV2640_C(nn.Module):
    def __init__(self, rgb_model, depth_model):
        super(EffNetV2640_C, self).__init__()

        # Use the feature extractors from both the RGB and Depth models
        self.rgb_feature_extractor = rgb_model.effnetRGB
        self.depth_feature_extractor = depth_model.effnetD

        # Remove the regression heads (fc2 layers)
        self.rgb_fc1 = rgb_model.fc1
        self.depth_fc1 = depth_model.fc1

        # Final fully connected layer after concatenation
        self.fc1_comb = nn.Linear(2000, 1000)
        self.fc2_comb = nn.Linear(1000, 1)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)

    def forward(self, x):
        preprocessRGB = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 640x640
            transforms.Normalize(  # Normalize the image
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        preprocessD = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 640x640
            transforms.Normalize(  # Normalize the image
                mean=[0.5],
                std=[0.5]
            )
        ])
        x_RGB = x[:, :3, :, :]
        x_RGB = preprocessRGB(x_RGB).to(self.device)
        x_D = x[:, 3, :, :].unsqueeze(1)
        x_D = preprocessD(x_D).to(self.device)

        # Forward pass through the RGB model
        rgb_features = self.rgb_feature_extractor(x_RGB)  # Features before the regression head
        rgb_features = self.rgb_fc1(rgb_features)  # Passing through fc1

        # Forward pass through the Depth model
        depth_features = self.depth_feature_extractor(x_D)  # Features before the regression head
        depth_features = self.depth_fc1(depth_features)  # Passing through fc1

        # Concatenate the features from RGB and Depth models
        combined_features = torch.cat((rgb_features, depth_features), dim=1)

        # Final regression layer
        layer_1 = self.fc1_comb(combined_features)
        output = self.fc2_comb(layer_1)

        return output
