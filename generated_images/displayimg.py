import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

# Function to load and display images side by side
def display_images_side_by_side(image_paths):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(20, 4))
    for i, path in enumerate(image_paths):
        img = imread(path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(os.path.basename(path))
    plt.tight_layout()
    plt.show()

# Function to load and display images in a 4x4 grid
def display_images_in_grid(image_paths):
    fig, axes = plt.subplots(5, 2, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < len(image_paths):
            img = imread(image_paths[i])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(os.path.basename(image_paths[i]))
        else:
            break

    plt.tight_layout()
    plt.show()

# List of image paths (change these paths according to your actual image paths)
image_paths = [
    'gen_1.png',
    'gen_2.png',
    'gen_3.png',
    'gen_4.png',
    'gen_5.png',
    'gen_6.png',
    'gen_7.png',
    'gen_8.png',
    'gen_9.png',
    'gen_10.png',
]

# Display images 
display_images_in_grid(image_paths)
