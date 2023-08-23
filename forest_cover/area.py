import cv2

# # Load the segmented image
# # segmented_image = cv2.imread(r"C:\Users\Lakshmi\Downloads\data\SegmentationClass\OIP.png", cv2.IMREAD_GRAYSCALE)

# # Define the forest class label (you might need to adjust this based on your image)
# # forest_class_label = 255

# # Count pixels belonging to the forest class
# # forest_pixel_count = cv2.countNonZero(segmented_image)

# # Calculate the area (assuming each pixel represents 1 square meter)
# # pixel_area = 1.0  # square meter
# # forest_area = forest_pixel_count * pixel_area

# # print("Forest Area:", forest_area, "square meters")

# #############
# def area_dif(img):
def area_dif():
    pixel_area = 1.0  # square meter
    # ###############
    img = cv2.imread(r"D:\llm projects\Forest-Amazon\forest_cover\pictures\new.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_count = cv2.countNonZero(img)
    new_area = new_count * pixel_area
    print(new_area)
    prev_img = cv2.imread(r"forest_cover/pictures/previous.png", cv2.IMREAD_GRAYSCALE)
    prev_count = cv2.countNonZero(prev_img)
    prev_area = prev_count * pixel_area
    print(prev_area)
    return (prev_area - new_area)/prev_area * 100








