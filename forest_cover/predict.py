from ultralytics import YOLO
# from ultralytics import utails


import cv2


model_path = r"C:\Users\Lakshmi\Downloads\last.pt"

# def predict(img):
#     img = cv2.imread(img)
#     H, W, _ = img.shape

#     model = YOLO(model_path)

#     results = model(img)

#     for result in results:
#         for j, mask in enumerate(result.masks.data):
#             mask = mask.numpy() * 255
#             mask = cv2.resize(mask, (W, H))
#             cv2.imwrite('./output.jpg', mask)

#     out_img = cv2.imread('./output.jpg')
#     return out_img

def predict(img):
    img = cv2.imread(img)
    H, W, _ = img.shape

    model = YOLO(model_path)

    results = model(img)

    for result in results:
        for j, mask in enumerate(result.masks.data):
            mask = mask.numpy() * 255
            mask = cv2.resize(mask, (W, H))
            cv2.imwrite('./pictures/output.jpg', mask)

    out_img = cv2.imread('./output.jpg')
    return out_img



