import re
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import pytesseract
except ImportError:
    pytesseract = None
import os
import shutil
try:
    from app.helpers.config import TESSERACT_CMD
except Exception:
    from helpers.config import TESSERACT_CMD

if pytesseract is not None:
    if TESSERACT_CMD:
        if os.path.isabs(TESSERACT_CMD) and os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        elif not os.path.isabs(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        else:
            fallback = shutil.which("tesseract")
            if fallback:
                pytesseract.pytesseract.tesseract_cmd = fallback
    else:
        fallback = shutil.which("tesseract")
        if fallback:
            pytesseract.pytesseract.tesseract_cmd = fallback

def extract_text_and_boxes(image):
    """
    Extract text and bounding boxes from an image using Tesseract OCR.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    text (list): List of words.
    boxes (list): List of bounding boxes corresponding to each word.
    """
    width, height = image.size

    # Perform OCR using pytesseract
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    boxes = []

    n_boxes = len(ocr_data['level'])
    for i in range(n_boxes):
        if int(ocr_data['conf'][i]) > 0:
            word = ocr_data['text'][i]
            if word.strip() == '':
                continue
            words.append(word)
            (left, top, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
            # Normalize the bounding boxes
            x1 = int(1000 * (left / width))
            y1 = int(1000 * (top / height))
            x2 = int(1000 * ((left + w) / width))
            y2 = int(1000 * ((top + h) / height))
            boxes.append([x1, y1, x2, y2])

    return words, boxes
    

def parse_hocr(hocr_content):
    soup = BeautifulSoup(hocr_content, 'html.parser')
    
    # Extract words with their bounding boxes
    words = []
    for span in soup.find_all('span', class_='ocrx_word'):
        text = span.get_text()
        # Extract bounding box coordinates
        title = span['title']
        bbox = re.search(r'bbox (\d+) (\d+) (\d+) (\d+);', title)
        if bbox:
            x1, y1, x2, y2 = map(int, bbox.groups())
            words.append({
                'text': text,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'center_y': (y1 + y2) / 2,
                'center_x': (x1 + x2) / 2
            })
    
    return words

def group_words_into_lines(words, y_tolerance=10):
    # Sort words by their vertical position
    words = sorted(words, key=lambda w: w['center_y'])
    
    lines = []
    current_line = []
    current_y = None
    
    for word in words:
        if current_y is None:
            current_y = word['center_y']
        
        if abs(word['center_y'] - current_y) <= y_tolerance:
            current_line.append(word)
        else:
            # Sort current line words by x position
            current_line = sorted(current_line, key=lambda w: w['x1'])
            lines.append(current_line)
            current_line = [word]
            current_y = word['center_y']
    
    if current_line:
        current_line = sorted(current_line, key=lambda w: w['x1'])
        lines.append(current_line)
    
    return lines

def extract_key_value_pairs(lines, x_tolerance=50):
    key_value_pairs = []
    for line in lines:
        median_x = pd.Series([word['x1'] for word in line]).median()
        key_words = [word for word in line if word['x1'] <= median_x - x_tolerance]
        value_words = [word for word in line if word['x1'] > median_x + x_tolerance]
        
        # If we have key and value in the line
        if key_words and value_words:
            key_text = ' '.join([w['text'] for w in key_words])
            value_text = ' '.join([w['text'] for w in value_words])
            key_value_pairs.append((key_text.strip(), value_text.strip()))
        else:
            # Handle cases where key or value spans the entire line
            line_text = ' '.join([w['text'] for w in line])
            key_value_pairs.append((None, line_text.strip()))
    
    return key_value_pairs

def extract_information(hocr_content):
    words = parse_hocr(hocr_content)
    lines = group_words_into_lines(words)
    return lines

    # Current implementation works without k,v extraction
    key_value_pairs = extract_key_value_pairs(lines)
    key_value_pairs = [pair for pair in key_value_pairs if pair[0] or pair[1]]
    
    return key_value_pairs

def extract_and_process_hocr(image, config, timeout=30):
    """Extract text using Tesseract with timeout."""
    try:
        # Set a custom timeout for pytesseract
        extracted_text = pytesseract.image_to_pdf_or_hocr(
            image, 
            lang='eng', 
            extension='hocr',
            timeout=timeout,
           config = f'{config} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz[]().,@ -c preserve_interword_spaces=1'
        )
        hocr_str = extracted_text.decode('utf-8')
        extracted_text = ""
        
        lines = extract_information(hocr_str)

        for line in lines:
            extracted_text += " ".join(w['text'] for w in line)
            extracted_text += "\n"

        return extracted_text
    except Exception as e:
        print(f"OCR extraction failed: {str(e)}")
        # Fallback to simpler extraction method
        return extract_text(image, config, timeout)

def extract_text(image, config, timeout=30):
    """Extract text using basic Tesseract with timeout."""
    try:
        extracted_text = pytesseract.image_to_string(
            image, 
            lang='eng',
            timeout=timeout, 
            config=f'{config} -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$abcdefghijklmnopqrstuvwxyz[]().,@ "'
        )
        return extracted_text
    except Exception as e:
        print(f"Basic text extraction failed: {str(e)}")
        return ""

def preprocessImage(image):
    new_size = (image.width * 2, image.height * 2)  # Adjust scaling factor as needed
    image_rescaled = image.resize(new_size, Image.Resampling.LANCZOS)
    np_image_rescaled = np.array(image_rescaled)
    gray_image = cv2.cvtColor(np_image_rescaled, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 37, 1
    )
    return binary_image

    # The below code is used for cleaning borders and lines in images, 
    # very useful for forms and tables. But new implementation uses segmentation
    # to get region of interests. Removing borders and lines might be useful in future
    # hence retaining the logic.

    img_inverted = cv2.bitwise_not(binary_image)
    
    horizontal = img_inverted.copy()
    vertical = img_inverted.copy()

    rows, cols = horizontal.shape

    horizontalsize = int(cols / 15)
    horizontal_structure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontalsize, 1)
    )

    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    horizontal_inv = cv2.bitwise_not(horizontal)

    masked_img = cv2.bitwise_and(img_inverted, img_inverted, mask=horizontal_inv)
    masked_img_inv = cv2.bitwise_not(masked_img)

    img_vertical = masked_img_inv.copy()
    img_vertical_inverted = cv2.bitwise_not(img_vertical)


    verticalsize = int(rows / 40)
    vertical_structure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, verticalsize)
    )

    vertical = cv2.erode(img_vertical_inverted, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    vertical_inv = cv2.bitwise_not(vertical)

    masked_img2 = cv2.bitwise_and(
        img_vertical_inverted, img_vertical_inverted, mask=vertical_inv
    )

    masked_img_inv2 = cv2.bitwise_not(masked_img2)
    image = masked_img_inv2.copy()
    return image

def has_text(segment, threshold=60, min_area=10000):
    # Calculate the standard deviation of pixel intensities
    h, w = segment.shape
    area = h * w
    if area < min_area or area > min_area*70: # increased max_area to allow more text
        return False
    
    std_dev = np.std(segment)
    return std_dev > threshold

def segment(img):
    contours, hierarchy = cv2.findContours(
    img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape[:2]
    # Sort contours from top to bottom, left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))

    segments = []
    for i, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox
        # Add padding if necessary
        roi = img[y:y+h, x:x+w]
        if not has_text(roi, min_area=((height*width)/10000)*5):
            continue
        segments.append(roi)

    return segments
