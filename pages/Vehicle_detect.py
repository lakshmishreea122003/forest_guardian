# import streamlit as st
# import numpy as np
# import torch
# from ultralytics import YOLO
# import cvzone
# import math
# import tempfile
# import os
# import cv2

import torch
import numpy as np
import streamlit as st
import cv2
import easyocr
import imutils

st.set_page_config(
    page_title="Vehicle Detect",
    page_icon="🚗",
)

st.markdown("<h1 style='color: green; font-style: italic; font-family: Comic Sans MS; font-size:5rem' >Vehicle Detection & Access</h1> <h3 style='color: #00755E; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>This project ensures secure access of vehicles in forests by identifying vehicles and validating license plates, preventing unauthorized entry to sensitive forest areas, and deterring illegal deforestation activities effectively.</h3>", unsafe_allow_html=True)


def num_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        return img, "Not allowed"

    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [location], 255)  # Fill the polygon
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    if len(result) > 0:
        text = result[0][-2]
        if text in allowed_plate_list:
            res = cv2.putText(
                img,
                text=text,
                org=(approx[0][0][0], approx[1][0][1] + 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            res = cv2.rectangle(
                img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3
            )
            return res, "Allowed"
        else:
            res = cv2.putText(
                img,
                text=text,
                org=(approx[0][0][0], approx[1][0][1] + 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            res = cv2.rectangle(
                img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 0, 255), 3
            )
            return res, "Not allowed"
    else:
        return img, "Not allowed"


allowed_plate_list = ["3B95A", "BJ69H"]  # List of allowed number plates

def obj_detect(img):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(img)
    class_labels = results.names
    predictions = results.pred[0]  # Get predictions for the first image
    unique_class_labels = set(predictions[:, -1].tolist())
    detected_objects = [class_labels[int(label)] for label in unique_class_labels]
    
    # Filter bounding boxes of only 'car' and 'truck'
    car_truck_boxes = []
    car_truck_classes = ['car', 'truck']
    for box_info in predictions:
        class_index = int(box_info[-1])
        if class_labels[class_index] in car_truck_classes:
            x1, y1, x2, y2 = box_info[:4]
            class_label = class_labels[class_index]
            car_truck_boxes.append((x1, y1, x2, y2, class_label))
    
    return detected_objects, car_truck_boxes



# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Read the uploaded image
    img = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    
    # Perform object detection and get bounding boxes
    detected_objects, car_truck_boxes = obj_detect(img)
    
    # Display detected objects and bounding boxes
    st.write("Detected Objects:", detected_objects)
    st.write("Bounding Boxes (Cars and Trucks):", car_truck_boxes)
    
    # Crop, recognize number plates, and display images of detected cars and trucks
    for index, box in enumerate(car_truck_boxes):
        x1, y1, x2, y2, class_label = box
        cropped_image = img[int(y1):int(y2), int(x1):int(x2)]
        
        # Perform number plate recognition and check if allowed
        num_plate_img, status = num_plate(cropped_image)
        
        # Display the cropped image with number plate recognition
        st.image(num_plate_img, caption=f"Detected {class_label} {index} ({status})", use_column_width=True)#


