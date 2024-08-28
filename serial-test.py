import serial
import time

# Replace 'COM3' with the port your Arduino is connected to
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)

def send_message(message):
    arduino.write((message + '\n').encode())
    # Print the message being sent for debugging
    print(f"Sent: {message}")

while True:
    user_input = input("Enter a message to display on the LCD: ")
    send_message(user_input)
    time.sleep(1)
