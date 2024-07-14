####################################################################
"""MOBILENETV3 ACCESSING THE 11TH LAYER"""
####################################################################

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Checking for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Transforms
transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load CIFAR-100 dataset
train_dataset = datasets.CIFAR100(
    root='./data', train=True, transform=transformer, download=True)
test_dataset = datasets.CIFAR100(
    root='./data', train=False, transform=transformer, download=True)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

# Custom CNN


class CustomCNN(nn.Module):
    def __init__(self, in_channels, num_classes=100):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=12, kernel_size=3, stride=1, padding=1)
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

        # self.fc = nn.Linear(32 * 3 * 3, num_classes)
        self.fc = nn.Linear(1568, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # print("1:", x.shape)
        x = self.bn1(x)
        # print("2:", x.shape)
        x = self.relu1(x)
        # print("3:", x.shape)
        x = self.pool(x)
        # print("4:", x.shape)
        x = self.conv2(x)
        # print("5:", x.shape)
        x = self.relu2(x)
        # print("6:", x.shape)
        x = self.conv3(x)
        # print("7:", x.shape)
        x = self.bn3(x)
        # print("8:", x.shape)
        x = self.relu3(x)
        # print("9:", x.shape)
        x = x.view(x.size(0), -1)  # Flatten the output
        # print("10:", x.shape)
        x = self.fc(x)
        return x


# Load pre-trained mobilenetv3 (large version)
mobilenet_v3 = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3 = nn.Sequential(*list(mobilenet_v3.features.children())[:11])
# print(mobilenet_v3)

# Combined model with MobileNetV3 followed by Custom CNN


class CombinedModel(nn.Module):
    def __init__(self, num_classes=100):
        super(CombinedModel, self).__init__()
        self.mobilenet_v3 = mobilenet_v3
        self.custom_cnn = CustomCNN(in_channels=80, num_classes=num_classes)

    def forward(self, x):
        x = self.mobilenet_v3(x)
        x = self.custom_cnn(x)
        return x


model = CombinedModel(num_classes=100).to(device)

# Optimizer and loss function
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 10

# Calculating the size of training and testing images
train_count = len(train_dataset)
test_count = len(test_dataset)

print(train_count, test_count)

# Model training and saving best model
best_accuracy = 0.0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # print(model)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().item() * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    model.eval()
    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) +
          ' Train Accuracy: ' + str(train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(),
                   'best_checkpoint_mobilenetv3_large.model')
        best_accuracy = test_accuracy
