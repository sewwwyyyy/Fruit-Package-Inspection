# Fruit-Package-Inspection
 Built a robust pipeline to extract structured data from package labels using OCR and image processing. 
 # 📦 OCR-Based Package Label Data Extraction System

This project is an intelligent OCR pipeline that reads printed package labels from product images and extracts key information (like country, fruit type, weight, best-before date, company name, and product code). The output is cleaned, verified against a known database, and compared with a ground truth JSON for evaluation.

---

## 🧠 How the Code Works – Step by Step

This section explains what the code does internally, broken down by stages:

### 1. **Text Detection – EasyOCR**

- The `detect_text_in_image()` function loads the image using **OpenCV** (`cv2.imread`) and feeds it into **EasyOCR** to detect text and get bounding boxes.
- Each detected text entry has 3 parts: bounding box coordinates, the actual text, and the confidence.


reader = easyocr.Reader(['en'])
result = reader.readtext(img)
width = max(x-coords) - min(x-coords), height = max(y-coords) - min(y-coords)
area = width * height

Select options (separated by commas):
1. country: India
2. fruit: Mango
...

### OCR Detected Text: ['Best Before: 24/06/2024', 'India', 'Mango', '1kg', '123456', 'FruitCo']

Generated Output:
1. country: India
2. fruit: Mango
3. weight: 1kg
4. productcode: 123456
5. best before: 24/06/2024
6. company: FruitCo

Select options (separated by commas): 1,2,3,4,5,6

Ground Truth: ['india', 'mango', '1kg', '123456', '24/06/2024', 'fruitco']
6 common elements
100.0 % common elements percentage

