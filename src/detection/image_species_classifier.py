import torchvision
from torchvision import transforms
import torch


class ImageSpeciesClassifier:
    def __init__(self):

        checkpoint = torch.load("model/models/european_birds.pth", map_location="cpu")

        self.classes = checkpoint["classes"]

        self.model = torchvision.models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(2048, len(self.classes))

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, image):

        x = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            out = self.model(x)

        idx = out.argmax().item()

        return self.classes[idx]