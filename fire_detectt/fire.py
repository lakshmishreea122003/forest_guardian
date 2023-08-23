<<<<<<< HEAD
# from ultralytics import YOLO
# import cvzone
# import cv2
# import math

# # # Running real time from webcam
# cap = cv2.VideoCapture(r"D:\llm projects\Forest-Amazon\fire_detectt\fire2.mp4")
# model = YOLO(r"D:\llm projects\Forest-Amazon\fire_detectt\fire.pt")

# # Reading the classes
# classnames = ['fire']
# fire_detected = False
# while True:
#     ret,frame = cap.read()
#     frame = cv2.resize(frame,(640,480))
#     result = model(frame,stream=True)

#     # Getting bbox,confidence and class names informations to work with
#     for info in result:
#         boxes = info.boxes
#         for box in boxes:
#             confidence = box.conf[0]
#             confidence = math.ceil(confidence * 100)
#             Class = int(box.cls[0])
#             if confidence > 50:
#                 x1,y1,x2,y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
#                 cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
#                 cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
#                                    scale=1.5,thickness=2)
#                 fire_detected = True
#     if fire_detected:
#         cv2.imwrite(frame,r"D:\llm projects\Forest-Amazon\fire_detectt\fire.jpg")
#         break
#     cv2.imshow('frame',frame)
#     cv2.waitKey(1)

# img = cv2.imread(r"D:\llm projects\Forest-Amazon\fire_detectt\fire.jpg")
# cv2.imshow('img',img)
# while True:
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# # Close the OpenCV window
# cv2.destroyAllWindows()


from ultralytics import YOLO
import cvzone
import cv2
import math

# Running real time from webcam
cap = cv2.VideoCapture(r"D:\llm projects\Forest-Amazon\fire_detectt\fire2.mp4")
model = YOLO(r"D:\llm projects\Forest-Amazon\fire_detectt\fire.pt")

# Reading the classes
classnames = ['fire']
fire_detected = False

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Getting bbox, confidence, and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50 and classnames[Class] == 'fire':
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)
                fire_detected = True

    if fire_detected:
        cv2.imwrite(r"D:\llm projects\Forest-Amazon\fire_detectt\fire.jpg", frame)
        break

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Read the saved image and display it
img = cv2.imread(r"D:\llm projects\Forest-Amazon\fire_detectt\fire.jpg")
cv2.imshow('img', img)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Close the OpenCV windows
cv2.destroyAllWindows()























# import streamlit as st
# import tempfile
# import os

# # Streamlit app title
# st.title("Video Uploader and Saver")

# # File upload widget
# uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
# model = YOLO(r"D:\llm projects\Forest-Amazon\fire_detectt\fire.pt")

# classnames = ['fire']

# if uploaded_file is not None:
#     # Display the uploaded video
#     vid = st.video(uploaded_file)
#     while True:
#         ret,frame = vid.read()
#         frame = cv2.resize(frame,(640,480))
#         result = model(frame,stream=True)
#         if result is not None:
#             st.write("Fire detected")
#             for info in result:
#                 boxes = info.boxes
#                 for box in boxes:
#                     confidence = box.conf[0]
#                     confidence = math.ceil(confidence * 100)
#                     Class = int(box.cls[0])
#                     if confidence > 50:
#                         x1,y1,x2,y2 = box.xyxy[0]
#                         x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
#                         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
#                         cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
#                                    scale=1.5,thickness=2)
#         st.image(frame)
#         break

#     # Button to save the video
#     if st.button("Save Video"):
#         # Create a temporary directory to store the video
#         temp_dir = tempfile.TemporaryDirectory()
#         temp_path = os.path.join(temp_dir.name, uploaded_file.name)

#         # Save the uploaded video to the temporary directory
#         with open(temp_path, "wb") as f:
#             f.write(uploaded_file.read())

#         # Display success message
#         st.success("Video saved successfully!")

#         # You can add additional processing or cleanup steps here

#         # Delete the temporary directory after saving
#         temp_dir.cleanup()



# import streamlit as st
# import tempfile
# import os
# from ultralytics import YOLO
# import cv2
# import math
# import cvzone

# # Streamlit app title
# st.markdown("<h1 style='color: green; font-style: italic; font-family: Comic Sans MS; font-size:5rem' >Detect the forest fires</h1> <h3 style='color: #ADFF2F; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Change </h3>", unsafe_allow_html=True)

# # File upload widget
# uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

# classnames = ['fire']

# if uploaded_file is not None:
#     # Display the uploaded video
#     vid = cv2.VideoCapture(uploaded_file)

#     # Initialize YOLO model
#     model = YOLO(r"D:\llm projects\Forest-Amazon\fire_detectt\fire.pt")

#     while True:
#         ret, frame = vid.read()
#         if not ret:
#             break

#         frame = cv2.resize(frame, (640, 480))
#         result = model(frame, stream=True)

#         if result is not None:
#             st.write("Fire detected")

#             for info in result:
#                 boxes = info.boxes
#                 for box in boxes:
#                     confidence = box.conf[0]
#                     confidence = math.ceil(confidence * 100)
#                     Class = int(box.cls[0])

#                     if confidence > 50:
#                         x1, y1, x2, y2 = box.xyxy[0]
#                         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
#                         cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
#                                            scale=1.5, thickness=2)

#         # Convert the frame to BGR to RGB format
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Display the frame using Streamlit
#         st.image(frame_rgb, channels="RGB", use_column_width=True)

#     vid.release()

#     # Button to save the video
#     if st.button("Save Video"):
#         # Create a temporary directory to store the video
#         temp_dir = tempfile.TemporaryDirectory()
#         temp_path = os.path.join(temp_dir.name, uploaded_file.name)

#         # Save the uploaded video to the temporary directory
#         with open(temp_path, "wb") as f:
#             f.write(uploaded_file.read())

#         # Display success message
#         st.success("Video saved successfully!")

#         # You can add additional processing or cleanup steps here

#         # Delete the temporary directory after saving
#         temp_dir.cleanup()




# import streamlit as st
# import tempfile
# import os

# # Streamlit app title
# st.title("Video Uploader and Saver")

# # File upload widget
# uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

# if uploaded_file is not None:
#     # Display the uploaded video
#     st.video(uploaded_file)

#     # Button to save the video
#     if st.button("Save Video"):
#         # Create a temporary directory to store the video
#         temp_dir = tempfile.TemporaryDirectory()
#         temp_path = os.path.join(temp_dir.name, uploaded_file.name)

#         # Save the uploaded video to the temporary directory
#         with open(temp_path, "wb") as f:
#             f.write(uploaded_file.read())

#         # Display success message
#         st.success("Video saved successfully!")

#         # You can add additional processing or cleanup steps here

#         # Delete the temporary directory after saving
#         temp_dir.cleanup()
=======
from ultralytics import YOLO
import cvzone
import cv2
import math




# Running real time from webcam
cap = cv2.VideoCapture('fire2.mp4')
model = YOLO('fire.pt')


# Reading the classes
classnames = ['fire']

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    result = model(frame,stream=True)

    # Getting bbox,confidence and class names informations to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5,thickness=2)




    cv2.imshow('frame',frame)
    cv2.waitKey(1)
>>>>>>> 35cd85c54096433fbb0b509b6e4d84d2ab461907
