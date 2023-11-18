import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 7 * 7, 512)  # Adjust input size based on dataset
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        return x


# device_configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 50
batch_size = 32
learning_rate = 0.001

# data_transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

dataset = ImageFolder(root=r'G:\data_set_for_yoga\new_dataset_1', transform=data_transforms)

# Define the percentage for each split
train_ratio = 0.8
test_ratio = 0.2
# val_ratio = 0.15

total_size = len(dataset)
train_size = int(train_ratio * total_size)
test_size = total_size - train_size
# val_size = total_size - train_size - test_size

train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)
# val_loader = DataLoader(val_set, batch_size=batch_size)

# Define the number of classes based on the dataset
num_classes = len(dataset.classes)

# Create an instance of the ConvNet model
model = ConvNet(num_classes)

# Define the loss function (CrossEntropyLoss) and optimizer (SGD)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    for step, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradient buffers
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  # Update weights
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Step [{step + 1}/{len(train_loader)}] - Loss: {loss.item()}')
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item()}')

model.eval()  # Set the model to evaluation mode

# correct = 0
# total = 0
# class_correct = [0] * len(dataset.classes)  # Initialize class-wise correct predictions
# class_total = [0] * len(dataset.classes)  # Initialize class-wise total predictions

# with torch.no_grad():
#     for inputs, labels in val_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#         # Calculate class-wise accuracy
#         for i in range(len(labels)):
#             label = labels[i].item()
#             class_correct[label] += (predicted[i] == label).item()
#             class_total[label] += 1
#
# # Calculate and print class-wise accuracy
# for i, class_name in enumerate(dataset.classes):
#     if class_total[i] != 0:
#         class_accuracy = 100 * class_correct[i] / class_total[i]
#         print(f'Validation Class {class_name} Accuracy: {class_accuracy:.2f}%')
#
# # Print overall validation accuracy
# accuracy = 100 * correct / total
# print(f'Validation Overall Accuracy: {accuracy:.2f}%')

# Test loop for class-wise accuracy
model.eval()  # Set the model to evaluation mode

correct = 0
total = 0
class_correct = [0] * len(dataset.classes)  # Initialize class-wise correct predictions
class_total = [0] * len(dataset.classes)  # Initialize class-wise total predictions

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calculate class-wise accuracy
        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

# Calculate and print class-wise accuracy for the test set
for i, class_name in enumerate(dataset.classes):
    if class_total[i] != 0:
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print(f'Test Class {class_name} Accuracy: {class_accuracy:.2f}%')

# Print overall test accuracy
accuracy = 100 * correct / total
print(f'Test Overall Accuracy: {accuracy:.2f}%')

# Save the model's state_dict (architecture and learned weights)
torch.save(model.state_dict(), 'model1.pth')

# Optionally, save the optimizer's state_dict (if you want to resume training)
torch.save(optimizer.state_dict(), 'optimizer1.pth')
