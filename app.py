import streamlit as st
import numpy as np
import cv2
from video import video_detect
from webcam import cam_detect
import tempfile
from math import pow, sqrt

st.markdown("<h2 style='text-align: center; color: white;'>Social Distancing Detection AI System</h2>", unsafe_allow_html=True)
#st.title("Social Distancing Detector")
#st.markdown("<h4 style='text-align: center; color: white;'>Project done by Students of PSG COLLEGE OF TECHNOLOGY</h4>", unsafe_allow_html=True)

#st.subheader('Project by Students of PSG COLLEGE OF TECHNOLOGY')

MIN_CONF = st.slider(
    'Minimum probability To Filter Weak Detections', 0.0, 1.0, 0.3)

st.subheader('Test Video Or Try Live Detection')
option = st.selectbox('Choose your option',
                      ('Test Video', 'Try Live Detection Using Webcam'))

if option == 'Test Video':
    cho = st.file_uploader("Choose a video...", type=["mp4", "mpeg","avi"])
    if cho is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(cho.read())

MIN_CONF = float(MIN_CONF)


if st.button('Start'):
    # Load model
    st.info("[INFO] Loading Model...")
    st.info("[INFO] Accessing Video Stream...")
    
    if option == "Try Live Detection Using Webcam":
        cam_detect(1, MIN_CONF)
    else:
        if tfile.name is not None:
            video_detect(tfile.name, MIN_CONF)


st.success("Design and Developed By Bhuvaneshwaran R")
