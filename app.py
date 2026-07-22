import cv2

from repository.user_repository import UserRepository
from services.camera import CameraService
from services.detector import FaceDetector
from services.registration import RegistrationService
from services.verifier import VerificationService

camera = CameraService(
    camera_index=0,
    width=640,
    height=480,
    mirror=True,
)

detector = FaceDetector()

repository = UserRepository()

registration = RegistrationService(
    detector=detector,
    repository=repository,
)

verification = VerificationService(
    detector=detector,
    repository=repository,
)
while True:
    frame = camera.read()
    if frame is None:
        break
    faces = detector.detect(frame)
    for face in faces:
       detector.draw_faces(frame, [face])

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:
        break
camera.release()
cv2.destroyAllWindows()