import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, resnet34, resnet50

# the dataset is provided as a .npz file (compressed numpy archive)
# it contains two arrays:
# images: uint8 array of shape (N, 3, 32, 32), values in [0, 255]
# labels: integer class labels in range [0, 8]
# we divide images by 255.0 to get float values in [0, 1]

data = np.load("train.npz")
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

print("Dataset size:", len(dataset))
print("Image shape:", images.shape)
print("Label range:", labels.min().item(), "to", labels.max().item())

NUM_CLASSES = 9

# pick one of: resnet18, resnet34, resnet50
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# resnet34 example
# model = resnet34(weights=None)
# model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# resnet50 example
# model = resnet50(weights=None)
# model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# sanity check -- output shape must be (1, 9)
model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 32, 32))
print("Output shape:", out.shape)

# save only the state dict, not the full model instance
torch.save(model.state_dict(), "model.pt")