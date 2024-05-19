import asyncio
from ultralytics import YOLO # for v9
import pygetwindow as gw
import pyautogui
import numpy as np
import cv2
import os

class GameWindow:
    def __init__(self, title):
        self.game_win = gw.getWindowsWithTitle(title)[0]

    def capture_screen(self):
        try:
            self.game_win.restore()
            self.game_win.activate()

            win_left, win_top, win_width, win_height = self.game_win.left, self.game_win.top, self.game_win.width, self.game_win.height
            screen = np.array(pyautogui.screenshot(region=(win_left, win_top, win_width, win_height)))
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

            return screen, win_left, win_top
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None, None, None

class Detect:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, image):
        try:
            predictions = self.model.predict(image, conf=0.6, show=True, show_labels=True, show_conf=True, line_width=1)
            boxes = predictions[0].boxes
            names = predictions[0].names

            return boxes, names
        except Exception as e:
            print(f"Error detecting objects: {e}")
            return None, None

async def smooth_aim(x_target, y_target, speed=1):
    x_current, y_current = pyautogui.position()
    x_diff = x_target - x_current
    y_diff = y_target - y_current
    x_step = x_diff * speed
    y_step = y_diff * speed
    pyautogui.moveRel(x_step, y_step, duration=0.1)
    await asyncio.sleep(0.1)  

async def aimbot(boxes, win_left, win_top):
    if boxes.xyxy.shape[0] != 0:
        xyxy = boxes.xyxy[0]
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = (xyxy[1] + xyxy[3]) / 2
        target_x = int(x_center + win_left)
        target_y = int(y_center + win_top)

        while True:
            x_current, y_current = pyautogui.position()
            if abs(target_x - x_current) < 5 and abs(target_y - y_current) < 5:
                pyautogui.click()
                break
            await smooth_aim(target_x, target_y)
            await asyncio.sleep(0.05)


async def process_frame(game_window, model):
    while True:
        screen, win_left, win_top = game_window.capture_screen()
        if screen is not None:
            boxes, names = model.detect_objects(screen)
            if boxes is not None and boxes.xyxy.shape[0] != 0:
                await aimbot(boxes, win_left, win_top)
        await asyncio.sleep(0.1) 

async def main():
    os.system('cls')
    game_window = GameWindow("Telegram Web")
    model = Detect('best.pt')

    await process_frame(game_window, model)

if __name__ == "__main__":
    asyncio.run(main())
