import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define your CNN model class (same as trained model)
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

# Modify the fully connected layer to have 10 output classes
model.fc = nn.Linear(in_features=16 * 16 * 32, out_features=10)
model.to(device)
model.eval()

# Load CIFAR-10 class names for example purposes (not really used here)
cifar10_classes = datasets.CIFAR10(
    root='./data', train=False, download=True).classes

# Define the transformation for the input image
transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load CIFAR-10 dataset for inference
test_dataset = datasets.CIFAR10(
    root='./data', train=False, transform=transformer, download=True)

test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


def predict_dataset(model, data_loader, class_names, num_images=5):
    predictions = []
    model.eval()
    count = 0
    for images, _ in data_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = class_names[predicted.item()]
            print(f'Predicted class: {predicted_class}')
            predictions.append(predicted_class)
            count += 1
            if count >= num_images:
                break


# Perform inference on CIFAR-10 dataset for 5 images
predict_dataset(model, test_loader, cifar10_classes, num_images=5)
