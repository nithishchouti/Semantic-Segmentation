import torch
import torch.nn as nn
from torch.optim import SGD 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

# Checking for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Transforms
transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1], formula (x-mean)/std
                         [0.5, 0.5, 0.5])
])

# Load CIFAR-100 dataset
train_dataset = datasets.CIFAR100(
    root='./data', train=True, transform=transformer, download=True)
test_dataset = datasets.CIFAR100(
    root='./data', train=False, transform=transformer, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# CNN Network


class ConvNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1
        # Input shape= (256,3,32,32)

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (256,12,32,32)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (256,12,32,32)
        self.relu1 = nn.ReLU()
        # Shape= (256,12,32,32)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size by factor 2
        # Shape= (256,12,16,16)

        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (256,20,16,16)
        self.relu2 = nn.ReLU()
        # Shape= (256,20,16,16)

        self.conv3 = nn.Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (256,32,16,16)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (256,32,16,16)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,16,16)

        self.fc = nn.Linear(in_features=16 * 16 * 32, out_features=num_classes)

    # Feed forward function
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

        # Above output will be in matrix form, with shape (256,32,16,16)
        output = output.view(-1, 32*16*16)

        output = self.fc(output)

        return output


model = ConvNet(num_classes=100).to(device)

# Optimizer and loss function
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9,
                weight_decay=0.0001)  # Using SGD optimizer
loss_function = nn.CrossEntropyLoss()

num_epochs = 5

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
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    model.eval()
    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) +
          ' Train Accuracy: ' + str(train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy
