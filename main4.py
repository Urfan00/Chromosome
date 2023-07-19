import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('11111.png')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
chromosome_count = len(contours)

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

cropped_chromosomes = []
for contour in sorted_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cropped_chromosome = image[y:y+h, x:x+w]
    cropped_chromosomes.append(cropped_chromosome)

# Display the processed image
cv2.imshow('Processed Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the cropped chromosome images
for i, chromosome in enumerate(cropped_chromosomes):
    cv2.imwrite(f'chromosome_{i+1}.jpg', chromosome)

