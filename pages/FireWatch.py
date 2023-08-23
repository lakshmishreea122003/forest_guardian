import streamlit as st
import cv2
from ultralytics import YOLO
import cvzone
import math
import tempfile
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import joblib

st.set_page_config(
    page_title="Amazon FireWatch",
    page_icon="🔥",
)

st.markdown("<h1 style='color: green; font-style: italic; font-family: Comic Sans MS; font-size:5rem' >Amazon FireWatch</h1> <h3 style='color: #00755E; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Advanced Fire Detection System using Computer Vision to aid early detection of the forest fires. </h3>", unsafe_allow_html=True)


model = YOLO(r"D:\llm projects\Forest-Amazon\fire_detectt\fire.pt")

classnames = ['fire']
fire_detected = False
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

if uploaded_file is not None:
    # Display the uploaded video
    st.video(uploaded_file)

    # Button to save the video
    if st.button("Save Video"):
        # Create a temporary directory to store the video
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name, uploaded_file.name)

        # Save the uploaded video to the temporary directory
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Video saved successfully!")
        temp_dir.cleanup()

    cap = cv2.VideoCapture(r"D:\llm projects\Forest-Amazon\fire_detectt\fire2.mp4")
    while True:
        ret,frame = cap.read()
        frame = cv2.resize(frame,(640,480))
        result = model(frame)
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
                    fire_detected = True

        if fire_detected:
            cv2.imwrite(r"D:\llm projects\Forest-Amazon\fire_detectt\fire.jpg", frame)
            break

    img = cv2.imread(r"D:\llm projects\Forest-Amazon\fire_detectt\fire.jpg")
    st.image(img)
year = None
year = st.text_input("Enter the year")
if year is not None:
   
    fire_model = joblib.load(r"D:\llm projects\Forest-Amazon\models\fire_model.pkl")
    fire_res = fire_model.predict(year)
    info=f'In palces ACRE	AMAPA	AMAZONAS	MARANHAO	MATO GROSSO	PARA	RONDONIA	RORAIMA	TOCANTINS the firespotes in the year {year} is {fire_res} respectively'
    st.write(info)

    st.write("Data Analysis of amazon firespots dataset")
    amazon_fires_data = pd.read_csv(r"C:\Users\Lakshmi\Downloads\archive (1)\inpe_brazilian_amazon_fires_1999_2019.csv")
    amazon_fires_data.drop(columns=['month','latitude','longitude'])
    data = amazon_fires_data.groupby(['year', 'state']).agg({'firespots': 'sum'}).reset_index()
    data = data.pivot(index='year', columns='state', values='firespots')
    transposed_data = data

# Create a bar chart
    fig = px.bar(transposed_data, x=transposed_data.index, y=transposed_data.columns)

# Update layout and axes
    fig.update_layout(title_text='Total Firespots per State')
    fig.update_xaxes(title_text='State')
    fig.update_yaxes(title_text='Firespots')
    st.plotly_chart(fig)






