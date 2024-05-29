import cv2
import numpy as np
from mss import mss
from pynput.mouse import Controller
import ctypes
import time

mouse = Controller()

sct = mss()

def click():
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  
    time.sleep(0.01)  
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  

def meowmeow():
    while True:
        screenshot = sct.grab(sct.monitors[1])
        img = np.array(screenshot)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        lower_green = np.array([0, 220, 205])
        upper_green = np.array([0, 220, 205])

        mask = cv2.inRange(img, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            target_position = (x + w // 2, y + h // 2)
            mouse.position = target_position

            current_position = mouse.position
            if (current_position[0] >= x and current_position[0] <= x + w and
                current_position[1] >= y and current_position[1] <= y + h):
                click()

if __name__ == "__main__":
    meowmeow()
