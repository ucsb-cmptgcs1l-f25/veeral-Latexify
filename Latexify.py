import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
image_path = "/Users/veeral.shroff/Downloads/sample_math.jpeg"

img = cv2.imread(image_path)
print(img is not None)  # Should print True

def preprocess_image(image_path, scale_percent=50):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Resize image for faster processing
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return clean, img

def segment_symbols(binary_img, min_area=850):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbols = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        symbol_img = binary_img[y:y+h, x:x+w]
        symbols.append((x, y, w, h, symbol_img))

    # Sort symbols top-left to bottom-right
    symbols = sorted(symbols, key=lambda b: (b[1], b[0]))
    return symbols

def recognize_symbol(symbol_img):
    pil_img = Image.fromarray(symbol_img)
    text = pytesseract.image_to_string(pil_img, config="--psm 10")
    return text.strip()

symbol_to_latex = {
    '+': '+',
    '-': '-',
    '=': '=',
    '(': '(',
    ')': ')',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '7': '7',
    'x': 'x',
    'y': 'y',
}

def map_to_latex(symbols):
    latex = "$"
    for x, y, w, h, img in symbols:
        char = recognize_symbol(img)
        print("OCR:", char)
        latex += symbol_to_latex.get(char, char)
        
    return latex + "$"

def handwritten_to_latex(image_path):
    binary_img, orig = preprocess_image(image_path)
    symbols = segment_symbols(binary_img)

    # visualization
    vis = orig.copy()
    for x, y, w, h, _ in symbols:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 1)

    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Found {len(symbols)} symbols")
    plt.axis("off")
    plt.show()

    # convert to LaTeX-like string
    return map_to_latex(symbols)

img = cv2.imread(image_path)
print(img.shape)  # should print height, width, channels

%matplotlib inline

latex = handwritten_to_latex(image_path)
print("Generated Latex: ", latex)

