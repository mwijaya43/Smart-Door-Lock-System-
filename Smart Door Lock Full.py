import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import time
import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import paho.mqtt.client as mqtt

# MQTT broker details
broker_address = "103.52.114.200" #VPS
broker_port = 1883

# Create MQTT client
client = mqtt.Client()
client.connect(broker_address, broker_port)

# GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)

# Set GPIO Pins
GPIO_TRIGGER = 18
GPIO_ECHO = 24
GPIO_DOOR_LOCK = 18  # Solenoid lock pin

# Set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
GPIO.setup(GPIO_DOOR_LOCK, GPIO.OUT)


# Function to open the door lock
def open_door_lock(pin):
    GPIO.output(pin, GPIO.HIGH)
    client.publish("lock", "open")  # Publish door lock status

# Function to close the door lock
def close_door_lock(pin):
    GPIO.output(pin, GPIO.LOW)
    client.publish("lock", "closed")  # Publish door lock status

# Function to read images for face recognition training
def read_images(path, image_size):
    names = ["owner","matth","owner2"]
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                training_images.append(img)
                training_labels.append(label)

            label += 1
    training_images = np.asarray(training_images, np.uint8)
    training_labels = np.asarray(training_labels, np.int32)
    return names, training_images, training_labels

def distance():
    # Set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)

    # Set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    StartTime = time.time()
    StopTime = time.time()

    # Save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()

    # Save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()

    # Time difference between start and arrival
    TimeElapsed = StopTime - StartTime

    # Multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2

    return distance
# Function to control the door based on sensor inputs
def door_control(ultrasonic_distance, face_detected, rfid_access, unknown_person_time):
    X_train = np.array([[50, 0, 0],  # Ultrasonic, Camera, RFID
                        [80, 1, 0],
                        [120, 0, 1],
                        [30, 1, 1],
                        [70, 0, 0],
                        [40, 1, 0],
                        [90, 0, 1],
                        [60, 1, 1],
                        [100, 0, 0],
                        [20, 1, 0],
                        [45, 0, 1],
                        [75, 1, 0]])

    y_train = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1])  # Door open (1) or closed (0)

    # Creating Random Forest classifier model
    model = RandomForestClassifier()

    # Training the model
    model.fit(X_train, y_train)

    sensor_values = np.array([ultrasonic_distance, face_detected, rfid_access])

    # Reshaping the array to 2D matrix
    sensor_values2 = sensor_values.reshape(1, -1)

    # Predicting using the trained model
    predicted_output = model.predict(sensor_values2)

    # Controlling the door based on the prediction
    if predicted_output[0] == 1:
        open_door_lock(GPIO_DOOR_LOCK)
    else:
        close_door_lock(GPIO_DOOR_LOCK)

    # Additional logic: If face is detected and person is approaching rapidly, open the door 50cm/2s
    if face_detected == 1 and ultrasonic_distance <= 50:
        prev_time = time.time()
        while ultrasonic_distance <= 50:
            current_time = time.time()
            if current_time - prev_time >= 2:
                open_door_lock(GPIO_DOOR_LOCK)
                break
            ultrasonic_distance = distance()

    # Notification for unknown person
    if face_detected == 1 and unknown_person_time >= 60:
        client.publish("camera_notif", "Unknown person detected for over 1 minute")

    # Notification for RFID access without face detection or wrong RFID
    if (rfid_access == 1 and face_detected == 0) or (rfid_access == 0 and face_detected == 1):
        client.publish("rfid_notif", "RFID access without face detection or wrong RFID")

if __name__ == '__main__':
    try:
        path_to_training_images = './data/at'
        training_image_size = (200, 200)
        names, training_images, training_labels = read_images(path_to_training_images, training_image_size)

        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(training_images, training_labels)

        face_cascade = cv2.CascadeClassifier("pi/haarcascade_frontalface_alt.xml")
        eye_cascade = cv2.CascadeClassifier("pi/haarcascade_eye.xml")

        camera = cv2.VideoCapture(0)
        unknown_person_time = 0

        while True:
            try:
                rfid = SimpleMFRC522()
                id, text = rfid.read()
                print(id)

                if id == 633926595783:
                    rfid_access = 1
                    print("Access granted")
                else:
                    rfid_access = 0
                    print("Not allowed...")
            except Exception as e:
                print("Error:", e)
                rfid_access = 0

            # Face recognition
            success, frame = camera.read()
            if success:
                faces = face_cascade.detectMultiScale(frame, 1.3, 5)
                if len(faces) > 0:
                    face_detected = 1
                    unknown_person_time = 0  # Reset unknown person timer
                else:
                    face_detected = 0
                    unknown_person_time += 1  # Increment unknown person timer

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_gray = gray[x:x+w, y:y+h]
                    if roi_gray.size == 0:
                        continue
                    roi_gray = cv2.resize(roi_gray, training_image_size)
                    label, confidence = model.predict(roi_gray)
                    text = '%s, confidence=%.2f' % (names[label], confidence)
                    print(text)
                    if confidence < 90:
                        text = names[label]
                    else:
                        text = 'Unknown'
                        unknown_person_time += 1  # Increment unknown person timer
                    cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Face Recognition', frame)

            # Ultrasonic distance measurement
            ultrasonic_distance = distance()
            print("Measured Distance = %.1f cm" % ultrasonic_distance)

            # Control the door based on sensor inputs
            door_control(ultrasonic_distance, face_detected, rfid_access, unknown_person_time)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Add a delay after each iteration
            time.sleep(1)

        # Release the video capture object and close all windows
        camera.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("Measurement stopped by User")
        GPIO.cleanup()