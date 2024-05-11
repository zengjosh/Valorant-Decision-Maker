import cv2
import pytesseract
from pytesseract import Output

def preprocess_image(image_path, region=None):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]  # Get the height and width of the image

    if region is not None:
        # Convert percentages to absolute pixel values
        x_pct, y_pct, w_pct, h_pct = region
        x = int(x_pct * w)
        y = int(y_pct * h)
        w = int(w_pct * w)
        h = int(h_pct * h)
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

def process_credits(credits):
    credits = credits.replace(',', '')
    credits = credits.replace('o', '')
    credits = credits.replace('-', '')
    credits = credits.replace('\n', '')
    credits = credits.replace('n', '')
    credits = credits.replace('.', '')
    credits = credits.strip()
    if (len(credits) != 1):
        credits  =credits.lstrip('0')

    return credits

image_path = 'vctdata/vct7.png'

u_credits = preprocess_image(image_path, region=(0.44492, 0.7125, 0.04805, 0.04861))
print(u_credits)
# process_credits(u_credits)
# print(f"Credits: {u_credits}")



