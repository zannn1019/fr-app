from repository.user_repository import UserRepository
from services.detector import FaceDetector
import cv2


class RegistrationService:

    def __init__(self):
        self.detector = FaceDetector()
        self.repository = UserRepository()

    def register(
        self,
        user_id: str,
        image_path: str,
    ):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        faces = self.detector.detect(image)

        if len(faces) == 0:
            raise ValueError("No face detected.")

        if len(faces) > 1:
            raise ValueError("Multiple faces detected.")

        face = faces[0]

        portrait = image[
            face.y1:face.y2,
            face.x1:face.x2,
        ]

        self.repository.save_portrait(
            user_id,
            portrait,
        )

        self.repository.save_embedding(
            user_id,
            face.embedding,
        )