import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder("dataset", transform=transform)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = torchvision.models.resnet50(weights="IMAGENET1K_V2")

model.fc = nn.Linear(2048, len(dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):

    for imgs, labels in loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)

        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch", epoch, "loss", loss.item())

save_path = "models/european_birds.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

torch.save({
    "model": model.state_dict(),
    "classes": dataset.classes
}, save_path)