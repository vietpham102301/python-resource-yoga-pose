from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Define the data transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a common size
    transforms.ToTensor(),  # Convert images to tensors
])

# Path to dataset
dataset_path = r'G:\data_set_for_yoga\dataset'

# Create an ImageFolder dataset
dataset = ImageFolder(root=dataset_path, transform=data_transforms)

# Iterate through the dataset and log problematic images
error_log = []

for image_path, _ in dataset.samples:
    try:
        with Image.open(image_path) as img:
            img.verify()  # Attempt to open the image
    except Exception as e:
        error_log.append((image_path, str(e)))

# Print or log the list of problematic images
for image_path, error_msg in error_log:
    print(f"Error with image: {image_path} - {error_msg}")
