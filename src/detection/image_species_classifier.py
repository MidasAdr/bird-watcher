import torchvision
from torchvision import transforms
import torch


class ImageSpeciesClassifier:
    def __init__(self):

        checkpoint = torch.load("models/european_birds.pth", map_location="cpu")

        self.classes = checkpoint["classes"]

        self.model = torchvision.models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(2048, len(self.classes))

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):

        x = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            out = self.model(x)

        idx = out.argmax().item()

        return self.classes[idx]
