import cv2
from typing import Optional

class CameraService:
    # Function
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        mirror: bool = True,
    ):
        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera.")

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.mirror = mirror

    def read(self):
        success, frame = self.camera.read()
        if not success:
            raise RuntimeError("Failed to read frame from camera.")

        if self.mirror:
            frame = cv2.flip(frame, 1)

        return frame

    def release(self):
        self.camera.release()

    # Properties
    @property
    def resolution(self):
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return width, height