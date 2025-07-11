import cv2
import numpy as np
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def detect_color(img):
    avg_color = img.mean(axis=0).mean(axis=0)
    b, g, r = avg_color
    if r > 150 and g < 100 and b < 100:
        return "Red"
    elif g > 150 and r < 100 and b < 100:
        return "Green"
    elif b > 150 and r < 100 and g < 100:
        return "Blue"
    elif r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    else:
        return "Unknown"

def save_snapshot(image, prediction):
    filename = f"Snapshots/{prediction}.jpg"
    cv2.imwrite(filename, image)
    return filename
