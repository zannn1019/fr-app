import cv2

from services.camera import CameraService
from services.detector import FaceDetector
from services.registration import RegistrationService

camera = CameraService(camera_index=0, width=640, height=480, mirror=True)
detector = FaceDetector()   
registration_service = RegistrationService()

while True:
    frame = camera.read()
    if frame is None:
        break
    faces = detector.detect(frame)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (face.x1, face.y1),
            (face.x2, face.y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{face.confidence:.2f}",
            (face.x1, face.y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:
        break
camera.release()
cv2.destroyAllWindows()