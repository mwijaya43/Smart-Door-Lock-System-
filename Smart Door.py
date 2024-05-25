from pynodered import node_red
from sklearn.linear_model import LinearRegression
import numpy as np

@node_red(category="pyfuncs")
def multivar_regression_with_actuators(node, msg):
    # Training dataset and labels
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

    y_train = np.array([[0, 0, 0],  # RFID Notification, Camera Notification, Lock
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [0, 0, 0],
                        [0, 1, 1],
                        [1, 0, 1],
                        [0, 1, 1]])

    # Creating linear regression model
    model = LinearRegression()

    # Training the model
    model.fit(X_train, y_train)

    # Reading sensor readings from the message payload
    sensor_values = np.array(msg['payload'])

    # Reshaping the array to 2D matrix
    sensor_values2 = sensor_values.reshape(1, -1)

    # Predicting using the trained model
    predicted_output = model.predict(sensor_values2)

    # Constructing the output message to be sent
    msg['payload'] = {
        "camera_notification": predicted_output[0][0],
        "rfid_notification": predicted_output[0][1],
        "lock_state": predicted_output[0][2]
    }

    return msg