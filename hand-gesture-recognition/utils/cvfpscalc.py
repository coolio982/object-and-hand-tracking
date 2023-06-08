from collections import deque
import cv2 as cv


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get_fps(self):
        current_tick = cv.getTickCount()
        difftime = (current_tick - self._start) * self._freq
        self._start = current_tick
        self._difftimes.append(difftime)
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 1)

        return fps_rounded
