import cv2
import pytesseract
from pytesseract import Output

def process_image(image_path, region=None):
    img = cv2.imread(image_path)

    if region is not None:
        x, y, w, h = region
        img = img[y:y+h, x:x+w]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_percent = 150  
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_LINEAR)

    blur = cv2.GaussianBlur(resized, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(blur, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    text = pytesseract.image_to_string(eroded, config='--psm 6')

    cv2.imshow('Final Preprocessed Image', eroded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return text

credits = process_image('bptest.jpg', region=(184, 83, 49, 17))
credits = credits.replace('.', '')
print(f"Credits: {credits}")

frounds = process_image('bptest.jpg', region=(726, 25, 14, 18))
print(f"Friendly Rounds: {frounds}")

erounds = process_image('bptest.jpg', region=(726, 25, 13, 16))
print(f"Enemy Rounds: {erounds}")

side = process_image('bptest.jpg', region=(94, 14, 48, 10))
print(f"Side: {side}")