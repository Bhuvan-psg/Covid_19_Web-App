import streamlit as st
import numpy as np
import imutils
import cv2
import tempfile
from math import pow, sqrt

def cam_detect(name, MIN_CONF):
    
    MIN_DISTANCE = 200

    model = "./yolo.caffemodel"

    labels = "./class_labels.txt"

    prototxt = "./yolo.txt"

    bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))

    network = cv2.dnn.readNetFromCaffe(prototxt, model)

    writer = None
    image_placeholder = st.empty()

    cap = cv2.VideoCapture(name)
    frame_no = 0
    while cap.isOpened():

        frame_no = frame_no+1

        # Capture one frame after another
        ret, frame = cap.read()

        if not ret:
            break

        (h, w) = frame.shape[:2]

        # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        network.setInput(blob)
        detections = network.forward()

        pos_dict = dict()
        coordinates = dict()

        # Focal length
        F = 615

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > MIN_CONF:

                class_id = int(detections[0, 0, i, 1])

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # Filtering only persons detected in the frame. Class Id of 'person' is 15
                if class_id == 15.00:

                    # Draw bounding box for the object
                    cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 2)

                    label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
                    #print("{}".format(label))


                    coordinates[i] = (startX, startY, endX, endY)

                    # Mid point of bounding box
                    x_mid = round((startX+endX)/2,4)
                    y_mid = round((startY+endY)/2,4)

                    height = round(endY-startY,4)
                    # Distance from camera based on triangle similarity
                    distance = (165 * F)/height
                    #print("Distance(cm):{dist}\n".format(dist=distance))

                    # Mid-point of bounding boxes (in cm) based on triangle similarity technique
                    x_mid_cm = (x_mid * distance) / F
                    y_mid_cm = (y_mid * distance) / F
                    pos_dict[i] = (x_mid_cm,y_mid_cm,distance)

        # Distance between every object detected in a frame
        close_objects = set()
        safe = []
        for i in pos_dict.keys():
            for j in pos_dict.keys():
                if i < j:
                    dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))

                    # Check if distance less than 2 metres or 200 centimetres
                    if dist < MIN_DISTANCE:
                        close_objects.add(i)
                        close_objects.add(j)
                    

        for i in pos_dict.keys():
            if i in close_objects:
                COLOR = (0,0,255)
            else:
                safe.append(1)
                COLOR = (0,255,0)
            (startX, startY, endX, endY) = coordinates[i]

            cv2.rectangle(frame, (startX, startY), (endX,endY), COLOR, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            # MAX_DISTANCE = 6
            # Safe_Distance = "Safe distance: > {} ft".format(MAX_DISTANCE)
            # cv2.putText(frame, Safe_Distance, (400, frame.shape[0] - 15),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            
            # draw the total number of social distancing violations on the output frame
            text = "Social Distancing Violations: {}".format(len(close_objects))
            cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
            
            # text1 = "Safe Zone: {}".format(len(safe))
            # cv2.putText(frame, text1, (10, frame.shape[0] - 15),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 165, 0), 2)
            
            # human_count = "Human count: {}".format(len(close_objects)+len(safe))
            # cv2.putText(frame, human_count, (400, frame.shape[0] - 45), 
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 0, 0), 2)
            
            # Convert cms to feet
            # cv2.putText(frame, 'Depth: {i} ft'.format(i=round(pos_dict[i][2]/30.48,4)), (startX, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

        display = 1
        if display > 0:

            image_placeholder.image(
                frame, caption='Live Social Distancing Monitor Running..!', channels="BGR")

        if writer is not None:
            writer.write(frame)



