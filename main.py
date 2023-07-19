import pytesseract
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

img = Image.open('123.jpg')
print(pytesseract.image_to_string(img))