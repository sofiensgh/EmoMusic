"""Emotion classifiers (Chapter 2 - CNN).

* EmotionCNN : a VGG-style CNN built like the ASL CNN from the NVIDIA course, but deeper.
* resnet18   : transfer learning option (like the doggy-door notebook).
"""
import torch
import torch.nn as nn

from .config import IMG_SIZE, MOODS


def conv_block(c_in, c_out, drop=0.25):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout(drop),
    )


class EmotionCNN(nn.Module):
    """Input: 1x48x48 grayscale face. Output: logits over the moods."""

    def __init__(self, n_classes=len(MOODS)):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),     # 48 -> 24
            conv_block(32, 64),    # 24 -> 12
            conv_block(64, 128),   # 12 -> 6
            conv_block(128, 256),  # 6  -> 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (IMG_SIZE // 16) ** 2, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResNetEmotion(nn.Module):
    """Pretrained ResNet18 adapted to 1-channel 48x48 faces (transfer learning)."""

    def __init__(self, n_classes=len(MOODS), pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=96, mode="bilinear", align_corners=False)
        return self.net(x.repeat(1, 3, 1, 1))


def build_model(arch="cnn", pretrained=True):
    if arch == "cnn":
        return EmotionCNN()
    if arch == "resnet18":
        return ResNetEmotion(pretrained=pretrained)
    raise ValueError(f"unknown arch {arch}")


def load_emotion_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
