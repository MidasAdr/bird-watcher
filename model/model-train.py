import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_dataset = ImageFolder("dataset_cropped", transform=train_transform)
val_dataset_base = ImageFolder("dataset_cropped", transform=val_transform)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_indices = random_split(full_dataset, [train_size, val_size], generator=generator)
_, val_dataset = random_split(val_dataset_base, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(2048, len(full_dataset.classes))

# Freeze all layers except layer4 and fc
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model.to(device)

optimizer = torch.optim.Adam(
    [
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4},
    ]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, verbose=True
)
loss_fn = nn.CrossEntropyLoss()

save_path = "models/european_birds.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

best_val_loss = float("inf")
best_val_acc = 0.0

print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<11} {'Val Loss':<12} {'Val Acc':<10} {'LR':<10}")
print("-" * 65)

for epoch in range(40):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (preds.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, labels)

            val_loss += loss.item()
            val_correct += (preds.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    scheduler.step(val_loss)

    lr = optimizer.param_groups[1]['lr']
    saved = ""

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        torch.save({
            "model": model.state_dict(),
            "classes": full_dataset.classes
        }, save_path)
        saved = " *"

    print(f"{epoch:<6} {train_loss:<12.4f} {train_acc:<11.1%} {val_loss:<12.4f} {val_acc:<10.1%} {lr:<10.1e}{saved}")

print("-" * 65)
print(f"Best model saved to {save_path}")
print(f"Val loss: {best_val_loss:.4f}, Val accuracy: {best_val_acc:.1%}")
