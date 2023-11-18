import os
import random
from shutil import copyfile

# Define paths
original_dataset_path = r'G:\data_set_for_yoga\dataset'
reduced_dataset_path = r'reduced_dataset'

# Define the list of selected classes
selected_classes = ['adho mukha svanasana', 'adho mukha vriksasana',
                    'bakasana', "dandasana", "garudasana", "malasana",
                    "sukhasana", "tulasana", "vajrasana", "vrischikasana"]  # selected classes

# Define the percentage of images to keep (50%)
percentage_to_keep = 0.5

# Create a directory for the reduced dataset
os.makedirs(reduced_dataset_path, exist_ok=True)

# Iterate through selected classes and copy a subset of images
for class_name in selected_classes:
    class_dir = os.path.join(original_dataset_path, class_name)
    reduced_class_dir = os.path.join(reduced_dataset_path, class_name)
    os.makedirs(reduced_class_dir, exist_ok=True)

    # List all image files in the class directory
    image_files = os.listdir(class_dir)

    # Calculate the number of images to keep
    num_images_to_keep = int(percentage_to_keep * len(image_files))

    # Randomly sample and copy the selected images
    selected_images = random.sample(image_files, num_images_to_keep)
    for image in selected_images:
        source_path = os.path.join(class_dir, image)
        destination_path = os.path.join(reduced_class_dir, image)
        copyfile(source_path, destination_path)

