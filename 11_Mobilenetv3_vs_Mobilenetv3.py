import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torchvision.models import mobilenet_v2, mobilenet_v3_large
from torch.optim import SGD
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import psutil
import torchmetrics
from sklearn.metrics import confusion_matrix

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


def backbone_loader(model_name):
    if model_name == 'mobilenet_v2':
        backbone = models.mobilenet_v2(pretrained=True)
        low_level_features = backbone.features[:4]
        high_level_features = backbone.features[4:-1]
        low_feat_ch = 24
        high_feat_ch = 320
    elif model_name == 'mobilenet_v3':
        backbone = mobilenet_v3_large(pretrained=True)
        low_level_features = backbone.features[:7]
        high_level_features = backbone.features[7:]
        low_feat_ch = 40
        high_feat_ch = 960

    return_layers = {'high_level_features': 'out',
                     'low_level_features': 'low_level'}
    backbone = IntermediateLayerGetter(nn.ModuleDict(
        {'low_level_features': low_level_features, 'high_level_features': high_level_features}), return_layers=return_layers)
    return backbone, low_feat_ch, high_feat_ch

# Segmentation Model


class SegmentationCustom(nn.Module):
    def __init__(self, num_classes, output_stride, model_name):
        super(SegmentationCustom, self).__init__()
        self.feature_extractor, low_feat_ch, high_feat_ch = backbone_loader(
            model_name)
        self.deeplab = Deeplab(low_feat_ch=low_feat_ch, high_feat_ch=high_feat_ch,
                               num_classes=num_classes, output_stride=output_stride)

    def forward(self, x):
        original_shape = x.shape[2:]
        features = self.feature_extractor(x)
        output_map = self.deeplab(features['low_level'], features['out'])
        output_map = F.interpolate(
            output_map, size=original_shape, mode='bilinear', align_corners=True)
        return output_map


# Initialize models
mobilenetv2_model = SegmentationCustom(
    num_classes=21, output_stride=16, model_name='mobilenet_v2')
mobilenetv3_model = SegmentationCustom(
    num_classes=21, output_stride=16, model_name='mobilenet_v3')

if torch.cuda.is_available():
    mobilenetv2_model.cuda()
    mobilenetv3_model.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_v2 = SGD(mobilenetv2_model.parameters(), lr=0.001,
                   momentum=0.9, weight_decay=1e-4)
optimizer_v3 = SGD(mobilenetv3_model.parameters(), lr=0.001,
                   momentum=0.9, weight_decay=1e-4)

# Function to calculate mIoU, mAcc, and PA


def calculate_metrics(predictions, labels, num_classes):
    iou = torchmetrics.JaccardIndex(
        task='multiclass', num_classes=num_classes).to(predictions.device)
    miou = iou(predictions, labels)

    accuracy = torchmetrics.Accuracy(
        task='multiclass', num_classes=num_classes).to(predictions.device)
    pa = accuracy(predictions, labels)

    confusion = confusion_matrix(labels.cpu().numpy().flatten(
    ), predictions.cpu().numpy().flatten(), labels=list(range(num_classes)))
    per_class_acc = confusion.diagonal() / confusion.sum(axis=1)
    macc = per_class_acc.mean()

    return miou, pa, macc

# Train and Evaluate function with additional metrics


def train_and_evaluate(model, optimizer, train_loader, test_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

    model.eval()
    correct = 0
    total = 0
    total_time = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.nelement()
            correct += (predicted == labels).sum().item()

            all_predictions.append(predicted)
            all_labels.append(labels)

    accuracy = correct / total
    avg_inference_time = total_time / len(test_loader)
    mem_usage = memory_usage()
    gpu_mem_usage = gpu_memory_usage()

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    miou, pa, macc = calculate_metrics(
        all_predictions, all_labels, num_classes=21)

    return accuracy, avg_inference_time, mem_usage, gpu_mem_usage, miou, pa, macc


# Train and evaluate models
metrics_v2 = train_and_evaluate(
    mobilenetv2_model, optimizer_v2, train_loader, test_loader)
metrics_v3 = train_and_evaluate(
    mobilenetv3_model, optimizer_v3, train_loader, test_loader)

print(f'MobileNetV2 - Accuracy: {metrics_v2[0]}, Inference Time: {metrics_v2[1]}, CPU Memory Usage: {metrics_v2[2]} MB, GPU Memory Usage: {
      metrics_v2[3]} MB, mIoU: {metrics_v2[4]}, Pixel Accuracy: {metrics_v2[5]}, Mean Accuracy: {metrics_v2[6]}')
print(f'MobileNetV3 - Accuracy: {metrics_v3[0]}, Inference Time: {metrics_v3[1]}, CPU Memory Usage: {metrics_v3[2]} MB, GPU Memory Usage: {
      metrics_v3[3]} MB, mIoU: {metrics_v3[4]}, Pixel Accuracy: {metrics_v3[5]}, Mean Accuracy: {metrics_v3[6]}')
