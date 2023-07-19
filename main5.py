import cv2
import os

# Step 1: Read the image
image = cv2.imread('7899.png')

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
for i, chromosome in enumerate(cropped_chromosomes):
    cv2.imwrite(os.path.join(output_directory, f'chromosome_{i+1}.jpg'), chromosome)

# Print the count and paths of the processed image and cropped chromosomes
print(f"Chromosome count: {chromosome_count}")
print(f"Processed image saved as: {os.path.join(output_directory, 'processed_image.jpg')}")
for i in range(chromosome_count):
    print(f"Cropped chromosome {i+1} saved as: {os.path.join(output_directory, f'chromosome_{i+1}.jpg')}")
