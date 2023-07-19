import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Step 1: Read the image
image = cv2.imread('x2.jpeg')

# Step 2: Preprocessing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Segmentation
_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 4: Counting chromosomes
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
chromosome_count = len(contours)

# Step 5: Sorting
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Step 6: Create a directory to save the new images
output_directory = 'output_images'
os.makedirs(output_directory, exist_ok=True)

# Step 7: Save the processed image
processed_image_path = os.path.join(output_directory, 'processed_image.jpg')
cv2.imwrite(processed_image_path, thresholded_image)

# Step 8: Create a subdirectory for cropped chromosomes
cropped_directory = os.path.join(output_directory, 'cropped_chromosomes')
os.makedirs(cropped_directory, exist_ok=True)

# Step 9: Cropping and saving the chromosomes
cropped_chromosomes = []
max_height = 0
total_width = 0
for i, contour in enumerate(sorted_contours):
    x, y, w, h = cv2.boundingRect(contour)
    cropped_chromosome = image[y:y + h, x:x + w]
    output_path = os.path.join(cropped_directory, f'cropped_chromosome_{i+1}.jpg')
    cv2.imwrite(output_path, cropped_chromosome)
    cropped_chromosomes.append(cropped_chromosome)
    if h > max_height:
        max_height = h
    total_width += w

# Step 10: Calculate the scaling factor
scaling_factor = 1.0
if total_width > 65500:
    scaling_factor = 65500 / total_width

# Step 11: Resize cropped chromosomes to have the same height
for i in range(chromosome_count):
    resized_width = int(cropped_chromosomes[i].shape[1] * scaling_factor)
    resized_height = int(max_height * scaling_factor)
    cropped_chromosomes[i] = cv2.resize(cropped_chromosomes[i], (resized_width, resized_height))

# Step 12: Sort the cropped chromosomes by their x-coordinate
cropped_chromosomes.sort(key=lambda x: cv2.moments(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))['m10'] / cv2.moments(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))['m00'])

# Step 13: Concatenate cropped chromosomes horizontally
ordered_image = np.concatenate(cropped_chromosomes, axis=1)

# Step 14: Save the ordered image
ordered_image_path = os.path.join(output_directory, 'ordered_image.jpg')
plt.imsave(ordered_image_path, cv2.cvtColor(ordered_image, cv2.COLOR_BGR2RGB))

# Print the count and paths of the processed image, cropped chromosomes, and the ordered image
print(f"Chromosome count: {chromosome_count}")
print(f"Processed image saved as: {processed_image_path}")
for i in range(chromosome_count):
    print(f"Cropped chromosome {i+1} saved as: {os.path.join(cropped_directory, f'cropped_chromosome_{i+1}.jpg')}")
print(f"Ordered image saved as: {ordered_image_path}")
