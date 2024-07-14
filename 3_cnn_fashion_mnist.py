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

        output = output.view(-1, 32 * 16 * 16)

        output = self.fc(output)

        return output


# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(num_classes=100)
model.load_state_dict(torch.load('best_checkpoint.model', map_location=device))
model.to(device)

# Modify the fully connected layer to have 10 output classes
model.fc = nn.Linear(in_features=16 * 16 * 32, out_features=11)
model.to(device)
model.eval()

# Define the class names for Fashion-MNIST
fashion_mnist_classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Define the transformation for the input image for Fashion-MNIST
transformer_fashion_mnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load Fashion-MNIST dataset for inference
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, transform=transformer_fashion_mnist, download=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def predict_dataset(model, data_loader, class_names, num_images=10):
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
    return predictions


# Perform inference on Fashion-MNIST dataset for 5 images
predictions = predict_dataset(
    model, test_loader, fashion_mnist_classes, num_images=5)
print(predictions)
