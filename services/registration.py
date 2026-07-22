from entities.registration_result import RegistrationResult
from repository.user_repository import UserRepository
from services.detector import FaceDetector
import cv2

class RegistrationService:

    def __init__(self, detector: FaceDetector, repository: UserRepository):
        self.detector = detector
        self.repository = repository

    def register(self, user_id, image):
        self._ensure_user_does_not_exist(user_id)

        image = self._load_image(image)
        face = self._detect_single_face(image)
        portrait = self._extract_portrait(image, face)

        self.repository.save_portrait(user_id, portrait)
        self.repository.save_embedding(user_id, face.embedding)

        return RegistrationResult(success=True, user_id=user_id)

    def _ensure_user_does_not_exist(self, user_id):
        if self.repository.exists(user_id):
            raise ValueError(f"User with id {user_id} already exists.")

    def _load_image(self, image):
        image_file = cv2.imread(image)
        if image_file is None:
            raise FileNotFoundError(image)
        return image_file

    def _detect_single_face(self, image):
        faces = self.detector.detect(image)
        if not faces:
            raise ValueError("No face detected.")
        if len(faces) > 1:
            raise ValueError("Multiple faces detected.")
        return faces[0]

    def _extract_portrait(self, image, face):
        return image[face.y1:face.y2, face.x1:face.x2]
