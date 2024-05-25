import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

channel = 18

def relay_on(pin):
    GPIO.output(pin, GPIO.HIGH)

def relay_off(pin):
    GPIO.output(pin, GPIO.LOW)

while True:
    try:
        rfid = SimpleMFRC522()
        id, text = rfid.read()
        print(id)
           
        if id == 633926595783:
            relay_on(channel)
            print("Access granted")
            time.sleep(5)
            relay_off(channel)
        else:
            relay_off(channel)
            print("Not allowed...")
    except Exception as e:
        print("Error:", e)
    
    # Add a delay after each RFID read operation
    time.sleep(1)  # Adjust this delay as needed
