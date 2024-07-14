import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import json
import os

# Define the CNN network class (should be the same as the trained model)


class ConvNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(in_features=16 * 16 * 32, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = output.view(-1, 32*16*16)

        output = self.fc(output)

        return output


# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(num_classes=100)
model.load_state_dict(torch.load('best_checkpoint.model', map_location=device))
model.to(device)
model.eval()

# Load CIFAR-100 class names
cifar100_classes = datasets.CIFAR100(
    root='./data', train=False, download=True).classes

# Define the transformation for the input image
transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def predict_image(image_path):
    # Load and transform the image
    image = Image.open(image_path)
    image = transformer(image).unsqueeze(0)

    # Move the image to the device
    image = image.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()


image_path = r'C:\Users\Nithish Chouti\Desktop\CNN\elephant.png'
# image_path = r'C:\Users\Nithish Chouti\Desktop\CNN\African_Bush_Elephant.jpg'
predicted_class_index = predict_image(image_path)
predicted_class_label = cifar100_classes[predicted_class_index]
print(f'The predicted class is: {
      predicted_class_label} (index: {predicted_class_index})')
