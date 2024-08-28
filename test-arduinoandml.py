import cv2
import numpy as np
import serial
import time
import pygame
import os

# Initialize Pygame mixer
pygame.mixer.init()

# Function to play audio for a specific class
def play_audio(class_name):
    audio_path = f'audio/{class_name}.mp3'
    if os.path.exists(audio_path):
        print(f"Playing audio for {class_name}")
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait until the audio finishes playing
            time.sleep(0.1)
    else:
        print(f"Audio file for {class_name} not found.")

# Initialize serial communication
try:
    ser = serial.Serial('COM4', 9600)  # Adjust COM port and baud rate as needed
    time.sleep(2)  # Wait for the serial connection to initialize
except serial.SerialException as e:
    print(f"Error: {e}")
    exit()

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
if isinstance(output_layers_indices, np.ndarray):
    output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]
else:
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]

# Load classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
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
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        return [(boxes[i], class_ids[i], confidences[i]) for i in indexes.flatten()]
    else:
        return []

# Capture from webcam or video
cap = cv2.VideoCapture(0)
while True:
    if ser.in_waiting > 0:
        signal = ser.readline().decode('latin-1').strip()
        print(f"Received signal: {signal}")  # Output the received signal
        if signal == "DETECT":
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = detect_objects(frame)
                detected_classes = set()  # Track detected classes to play audio only once per frame
                for box, class_id, confidence in detections:
                    x, y, w, h = box
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected_classes.add(classes[class_id])

                # Play audio for detected classes
                for class_name in detected_classes:
                    play_audio(class_name)

                cv2.imshow("Image", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        elif signal == "QUIT":
            break
cap.release()
cv2.destroyAllWindows()
