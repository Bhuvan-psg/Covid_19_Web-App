from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import streamlit as st
import imutils
import cv2

def video_detect(video_name, MIN_CONF):
    
    cuda = 'False'

    labelsPath = "yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    USE_GPU = bool(cuda)
    MIN_DISTANCE = 50
    file_url = 'https://pjreddie.com/media/files/yolov3.weights'
    file_name = wget.download(file_url)
    #weightsPath = "yolov3.weights"
    weightsPath = file_name
    configPath = "yolo-coco/yolov3.cfg"

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


    if USE_GPU:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    vs = cv2.VideoCapture(video_name)
    writer = None
    image_placeholder = st.empty()
    while True:

        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        frame = cv2.resize(frame, (700,700))
        results = detect_people(frame, net, ln,
                                personIdx=LABELS.index("person"))

        violate = set()

        if len(results) >= 2:

            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):

                    if D[i, j] < MIN_DISTANCE:

                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):

            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            #cv2.circle(frame, (cX, cY), 5, color, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        #datet = str(datetime.datetime.now())
        #frame = cv2.putText(frame, datet, (0, 35), font, 1,
         #                   (0, 255, 255), 2, cv2.LINE_AA)
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        display = 1
        if display > 0:

            image_placeholder.image(
                frame, caption='Live Social Distancing Monitor Running..!', channels="BGR")

        if writer is not None:
            writer.write(frame)


