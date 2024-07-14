import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torchvision.models import mobilenet_v3_large
from torch.optim import SGD
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import psutil

# Function to calculate memory usage


def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

# Function to calculate GPU VRAM usage


def gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    else:
        return 0


class VOCSegmentationCustom(datasets.VOCSegmentation):
    def __init__(self, root, year='2012', image_set='train', transform=None, target_transform=None, transforms=None):
        super(VOCSegmentationCustom, self).__init__(root, year,
                                                    image_set, transform, target_transform, transforms)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# Transforms
transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

target_transformer = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.squeeze(x, 0).long())
])

# Initialize custom dataset
train_dataset = VOCSegmentationCustom(
    root='./data', year='2012', image_set='train', transform=transformer, target_transform=target_transformer)
test_dataset = VOCSegmentationCustom(
    root='./data', year='2012', image_set='val', transform=transformer, target_transform=target_transformer)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

# Custom IntermediateLayerGetter


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = {k: v for k, v in model.named_children()
                  if k in return_layers}
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = {}
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

# ASPP (Atrous Spatial Pyramid Pooling) Module


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride):
        super(ASPP, self).__init__()
        self.act = nn.ReLU6()
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.bn_3 = nn.BatchNorm2d(out_channels)
        self.bn_4 = nn.BatchNorm2d(out_channels)
        self.bn_5 = nn.BatchNorm2d(out_channels)
        self.bn_6 = nn.BatchNorm2d(out_channels)

        if output_stride == 16:
            self.operation_1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1)
            self.operation_2 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
            self.operation_3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
            self.operation_4 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        elif output_stride == 8:
            self.operation_1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1)
            self.operation_2 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
            self.operation_3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=24, dilation=24)
            self.operation_4 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=36, dilation=36)
        else:
            raise ValueError('Output stride must be 8 or 16')

        self.pool = nn.AdaptiveAvgPool2d((1))
        self.conv_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        output_1 = self.act(self.bn_1(self.operation_1(x)))
        output_2 = self.act(self.bn_2(self.operation_2(x)))
        output_3 = self.act(self.bn_3(self.operation_3(x)))
        output_4 = self.act(self.bn_4(self.operation_4(x)))
        pool = self.pool(x)
        pool = self.act(self.bn_5(self.conv_pool(pool)))
        pool = F.interpolate(pool, size=x.size()[
                             2:], mode='bilinear', align_corners=True)
        output = torch.cat(
            (output_1, output_2, output_3, output_4, pool), dim=1)
        output = self.act(self.bn_6(self.conv(output)))
        return output

# Deeplab Decoder


class Deeplab(nn.Module):
    def __init__(self, low_feat_ch, high_feat_ch, num_classes, output_stride):
        super(Deeplab, self).__init__()
        self.aspp = ASPP(high_feat_ch, 256, output_stride)
        self.low_conv = nn.Conv2d(low_feat_ch, 48, kernel_size=1)
        self.low_bn = nn.BatchNorm2d(48)
        self.act = nn.ReLU6()
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, low_features, high_features):
        high_features = self.aspp(high_features)
        low_features = self.act(self.low_bn(self.low_conv(low_features)))
        high_features = F.interpolate(high_features, size=low_features.size()[
                                      2:], mode='bilinear', align_corners=True)
        concat_features = torch.cat([high_features, low_features], dim=1)
        output = self.classifier(concat_features)
        return output

# Backbone Loader


def backbone_loader():
    backbone = mobilenet_v3_large(pretrained=True)
    backbone.low_level_features = backbone.features[:7]
    backbone.high_level_features = backbone.features[7:]

    return_layers = {'high_level_features': 'out',
                     'low_level_features': 'low_level'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    return backbone

# Segmentation Model


class SegmentationCustom(nn.Module):
    def __init__(self, num_classes, output_stride):
        super(SegmentationCustom, self).__init__()
        self.feature_extractor = backbone_loader()
        self.deeplab = Deeplab(low_feat_ch=40, high_feat_ch=960,
                               num_classes=num_classes, output_stride=output_stride)

    def forward(self, x):
        original_shape = x.shape[2:]
        features = self.feature_extractor(x)
        output_map = self.deeplab(features['low_level'], features['out'])
        output_map = F.interpolate(
            output_map, size=original_shape, mode='bilinear', align_corners=True)
        return output_map


# Initialize Model
model = SegmentationCustom(num_classes=21, output_stride=16)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

# Optimizer and loss function
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss(ignore_index=255)

num_epochs = 10

# Training Loop
best_accuracy = 0.0
train_count = len(train_dataset)
test_count = len(test_dataset)

# Track initial memory usage
initial_cpu_memory = memory_usage()
initial_gpu_memory = gpu_memory_usage()

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().item() * images.size(0)

    train_loss = train_loss / train_count

    # Calculate training accuracy
    correct_train_pixels = 0
    total_train_pixels = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)

            # Calculate the number of correct pixels
            correct_train_pixels += torch.sum(predictions == labels).item()
            total_train_pixels += torch.numel(labels)

    train_accuracy = correct_train_pixels / \
        total_train_pixels  # Calculate accuracy as a fraction

    # Evaluation on testing dataset
    model.eval()
    correct_pixels = 0
    total_pixels = 0
    inference_times = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            inference_times.append(end_time - start_time)

            _, predictions = torch.max(outputs.data, 1)

            # Calculate the number of correct pixels
            correct_pixels += torch.sum(predictions == labels).item()
            total_pixels += torch.numel(labels)

    test_accuracy = correct_pixels / total_pixels  # Calculate accuracy as a fraction
    avg_inference_time = np.mean(inference_times)

    print(f'Epoch: {epoch} Train Loss: {train_loss:.4f} Train Accuracy: {
          train_accuracy:.4f} Test Accuracy: {test_accuracy:.4f}')
    print(f'Average Inference Time: {avg_inference_time:.4f} seconds')

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(),
                   'best_checkpoint_deeplabv3_mobilenetv3_large.model')
        best_accuracy = test_accuracy

# Track final memory usage
final_cpu_memory = memory_usage()
final_gpu_memory = gpu_memory_usage()

# Print total memory usage
print(f'Initial CPU Memory Usage: {initial_cpu_memory:.2f} MB')
print(f'Final CPU Memory Usage: {final_cpu_memory:.2f} MB')
print(f'Total CPU Memory Used: {final_cpu_memory - initial_cpu_memory:.2f} MB')
if torch.cuda.is_available():
    print(f'Initial GPU Memory Usage: {initial_gpu_memory:.2f} MB')
    print(f'Final GPU Memory Usage: {final_gpu_memory:.2f} MB')
    print(f'Total GPU Memory Used: {
          final_gpu_memory - initial_gpu_memory:.2f} MB')
