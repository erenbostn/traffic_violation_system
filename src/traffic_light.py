import cv2
import numpy as np


class TrafficLightDetector:
    def __init__(self, roi):
        # roi = (x, y, w, h)
        x, y, w, h = roi
        self.x1 = int(x)
        self.y1 = int(y)
        self.x2 = int(x + w)
        self.y2 = int(y + h)

    def get_light_state(self, frame):
        roi_frame = frame[self.y1 : self.y2, self.x1 : self.x2]

        if roi_frame.size == 0:
            return "UNKNOWN"

        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        # RED (iki aralık)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask_red1 + mask_red2

        # GREEN
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        red_ratio = np.sum(red_mask > 0) / red_mask.size
        green_ratio = np.sum(green_mask > 0) / green_mask.size

        if red_ratio > 0.05:
            return "RED"
        elif green_ratio > 0.05:
            return "GREEN"
        else:
            return "UNKNOWN"
