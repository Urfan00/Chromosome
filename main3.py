import cv2
import numpy as np

image = cv2.imread("11111.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


_, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

num_chromosomes = len(contours)

cv2.putText(image, f"Chromosomes: {num_chromosomes}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Chromosome Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
