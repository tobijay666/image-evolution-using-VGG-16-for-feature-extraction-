import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os

# Load the reference image
def load_reference_image(image_path, size):
    image = load_img(image_path, target_size=size)
    image = img_to_array(image) / 255.0  # Normalize the image to [0, 1]
    if image.shape[2] == 1:  # Check if the image is grayscale
        image = np.concatenate([image] * 3, axis=2)  # Convert to RGB
    return image

# Display an image
def display_image(image, title='Image'):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Generate a random image
def generate_random_image(size):
    return np.random.rand(*size)

# Extract features using VGG19
def extract_features(image, model):
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image * 255.0)  # Convert back to [0, 255] before preprocessing
    features = model.predict(image)
    return features.flatten()

# Calculate fitness based on similarity of extracted features
def fitness_function(image, reference_features, model):
    image_features = extract_features(image, model)
    return -np.sum((image_features - reference_features) ** 2)

# Crossover between two images
def crossover(image1, image2):
    alpha = np.random.rand()
    return np.clip(image1 * alpha + image2 * (1 - alpha), 0, 1)

# Mutate an image
def mutate(image, reference_image, correct_pixels, mutation_rate=0.01):
    mutated_image = image.copy()
    height, width, channels = image.shape

    for i in range(height):
        for j in range(width):
            if (i, j) not in correct_pixels:
                if np.random.rand() < mutation_rate:
                    mutated_image[i, j] = np.random.rand(3)  # Random mutation
                else:
                    # Slightly adjust pixel values towards the reference image
                    mutated_image[i, j] = np.clip(mutated_image[i, j] + 0.1 * (reference_image[i, j] - mutated_image[i, j]), 0, 1)
    return mutated_image

# Identify correctly guessed pixels
def identify_correct_pixels(image, reference_image, threshold=0.1):
    correct_pixels = set()
    height, width, channels = image.shape

    for i in range(height):
        for j in range(width):
            if np.linalg.norm(image[i, j] - reference_image[i, j]) < threshold:
                correct_pixels.add((i, j))
    return correct_pixels

# Evolve images to resemble the reference image
def evolve_images(reference_image, reference_features, model, population_size, generations, mutation_rate):
    image_size = reference_image.shape
    population = [generate_random_image(image_size) for _ in range(population_size)]

    # Create a directory to save the images
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')

    for gen in range(generations):
        print(f'Generation {gen+1}/{generations}')

        # Calculate fitness for each image
        fitness_scores = [fitness_function(image, reference_features, model) for image in population]
        
        # Sort population by fitness
        population = [image for _, image in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)]
        
        # Select the top half
        next_gen = population[:population_size // 2]

        # Identify correct pixels in the best image
        correct_pixels = identify_correct_pixels(next_gen[0], reference_image)

        # Generate new population through crossover and mutation
        while len(next_gen) < population_size:
            parent_indices = np.random.choice(len(next_gen), 2, replace=False)
            parents = [next_gen[i] for i in parent_indices]
            offspring = crossover(parents[0], parents[1])
            offspring = mutate(offspring, reference_image, correct_pixels, mutation_rate)
            next_gen.append(offspring)
        
        population = next_gen

        # Save and display the best image of the current generation
        best_image = population[0]
        plt.imshow(best_image)
        plt.title(f'Generation {gen+1}')
        plt.axis('off')
        plt.savefig(f'generated_images/gen_{gen+1}.png')
        plt.show()

    return population[0]  # Return the best image from the final generation

# Main execution
reference_image_path = 'test.jpg'
image_size = (224, 224)  # Image size for the reference image and generated images

reference_image = load_reference_image(reference_image_path, image_size)

# Display the loaded reference image
display_image(reference_image, title='Reference Image')

# Load VGG19 model for feature extraction
vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-1].output)

# Extract features from the reference image
reference_features = extract_features(reference_image, vgg_model)

population_size = 20
generations = 10
mutation_rate = 0.01

print('Evolving images...')
best_image = evolve_images(reference_image, reference_features, vgg_model, population_size, generations, mutation_rate)
print('Evolution completed.')

# Display final results
plt.subplot(1, 2, 1)
plt.title('Reference Image')
plt.imshow(reference_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Final Evolved Image')
plt.imshow(best_image)
plt.axis('off')

plt.show()
print('Done.')
