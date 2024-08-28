import cv2
import serial
import time
import numpy as np

# Initialize the YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Connect to DroidCam
cap = cv2.VideoCapture('192.168.1.5:4747/video')

# Initialize serial communication with Arduino
arduino = serial.Serial('COM3', 9600)
time.sleep(2)  # Wait for the connection to initialize

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to send to Arduino
    detected_objects = []

    # Analyzing the outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add the detected object class ID to the list
                detected_objects.append(str(class_id))

    # Sending data to Arduino
    if detected_objects:
        detected_string = ",".join(detected_objects)
        arduino.write((detected_string + "\n").encode())

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
