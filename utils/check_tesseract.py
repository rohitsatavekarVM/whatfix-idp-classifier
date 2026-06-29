import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = (
    "/usr/bin/tesseract"
)
img = Image.new("RGB", (200, 60), color="white")
# draw text
from PIL import ImageDraw
draw = ImageDraw.Draw(img)
draw.text((10, 10), "Hello OCR", fill="black")

text = pytesseract.image_to_string(img)

print("OCR OUTPUT:", text)