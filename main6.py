import cv2
import os
import numpy as np

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

# Step 6: Cropping
cropped_chromosomes = []
for contour in sorted_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cropped_chromosome = image[y:y + h, x:x + w]
    cropped_chromosomes.append(cropped_chromosome)

# Step 7: Create a directory to save the new images
output_directory = 'output_images'
os.makedirs(output_directory, exist_ok=True)

# Step 8: Save the processed image and cropped chromosomes
cv2.imwrite(os.path.join(output_directory, 'processed_image.jpg'), thresholded_image)

# Step 9: Group similar chromosomes together
grouped_chromosomes = []
grouped_chromosomes.append([cropped_chromosomes[0]])  # Initialize the first group
for i in range(1, chromosome_count):
    matched = False
    for group in grouped_chromosomes:
        reference_chromosome = group[0]
        max_score = -1  # Initialize the maximum correlation score
        index = -1  # Initialize the index of the matching chromosome
        for j in range(i, min(i + 5, chromosome_count)):  # Compare with the next few chromosomes
            target_chromosome = cropped_chromosomes[j]
            resized_reference = cv2.resize(reference_chromosome, (target_chromosome.shape[1], target_chromosome.shape[0]))
            correlation_score = cv2.matchTemplate(resized_reference, target_chromosome, cv2.TM_CCOEFF_NORMED)
            if np.max(correlation_score) > max_score:  # Compare the maximum score
                max_score = np.max(correlation_score)
                index = j
        if max_score > 0.8:  # Adjust the threshold as per your requirement
            matched = True
            group.append(cropped_chromosomes[index])
            break
    if not matched:
        grouped_chromosomes.append([cropped_chromosomes[i]])

# Step 10: Save grouped chromosomes
for i, group in enumerate(grouped_chromosomes):
    resized_group = [cv2.resize(img, group[0].shape[:2][::-1]) for img in group]
    group_image = cv2.hconcat(resized_group)
    cv2.imwrite(os.path.join(output_directory, f'group_{i+1}.jpg'), group_image)

# Print the count and paths of the processed image, cropped chromosomes, and grouped chromosomes
print(f"Chromosome count: {chromosome_count}")
print(f"Processed image saved as: {os.path.join(output_directory, 'processed_image.jpg')}")
for i in range(chromosome_count):
    print(f"Cropped chromosome {i+1} saved as: {os.path.join(output_directory, f'chromosome_{i+1}.jpg')}")
for i, group in enumerate(grouped_chromosomes):
    print(f"Group {i+1} saved as: {os.path.join(output_directory, f'group_{i+1}.jpg')}")
