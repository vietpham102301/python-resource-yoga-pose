import time
import torch
from PIL import Image
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

start_time = time.time()


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
        print(x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        print(x.shape)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        print(x.shape)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        print(x.shape)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        return x


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    # transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

dataset = ImageFolder(root=r'G:\data_set_for_yoga\dataset', transform=data_transforms)
num_classes = len(dataset.classes)
model = ConvNet(num_classes)  # Re-create the model with the same architecture
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode (important for inference)

image = Image.open(r'F:\saved_frames\4052_1699336789.png')

# Convert the image to RGB (3 channels)
image = image.convert('RGB')

# Apply the same transformations you used during training
image = data_transforms(image).unsqueeze(0)  # Add batch dimension

# Convert the image tensor back to a numpy array for visualization
image_for_display = image.squeeze(0).permute(1, 2, 0).numpy()

# Display the transformed image
plt.imshow(image_for_display)
plt.axis('off')
plt.show()

# Use the loaded model for prediction
with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)
    confidence = torch.max(probabilities)

    # Get the class name corresponding to the predicted class index
    predicted_class = dataset.classes[predicted.item()]

    # Print the predicted class and confidence
    print(f'Predicted Class: {predicted_class}')
    print(f'Confidence: {confidence.item()}')

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

correct = 0
total = 0
class_correct = [0] * len(dataset.classes)  # Initialize class-wise correct predictions
class_total = [0] * len(dataset.classes)  # Initialize class-wise total predictions

dataset = ImageFolder(root=r'G:\data_set_for_yoga\dataset', transform=data_transforms)
# dataset = ImageFolder(root=r'F:\pytorch_learn_1\reduced_dataset', transform=data_transforms)

# Define the percentage for each split
train_ratio = 0.7
test_ratio = 0.15
val_ratio = 0.15

total_size = len(dataset)
train_size = int(train_ratio * total_size)
test_size = int(test_ratio * total_size)
val_size = total_size - train_size - test_size

train_set, test_set, val_set = random_split(dataset, [train_size, test_size, val_size])
batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calculate class-wise accuracy
        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

# Calculate and print class-wise accuracy
for i, class_name in enumerate(dataset.classes):
    if class_correct[i] != 0:
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print(f'Validation Class {class_name} Accuracy: {class_accuracy:.2f}%')

# Print overall validation accuracy
accuracy = 100 * correct / total
print(f'Validation Overall Accuracy: {accuracy:.2f}%')
