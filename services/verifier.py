from entities.verification_result import VerificationResult
from repository.user_repository import UserRepository
from services.detector import FaceDetector
from utils.similarity import cosine_similarity

class VerificationService:

    THRESHOLD = 0.5

    def __init__(
        self,
        detector: FaceDetector,
        repository: UserRepository,
    ):
        self.detector = detector
        self.repository = repository

    def verify(self, image) -> VerificationResult:
        face = self._detect_single_face(image)

        user_id, similarity = self._find_best_match(
            face.embedding
        )

        return self._build_result(
            user_id,
            similarity,
        )

    def _detect_single_face(self, image):
        faces = self.detector.detect(image)

        if len(faces) == 0:
            raise ValueError("No face detected.")

        if len(faces) > 1:
            raise ValueError("Multiple faces detected.")

        return faces[0]

    def _find_best_match(self, embedding):
        embeddings = self.repository.load_all_embeddings()

        best_user = None
        best_similarity = -1.0

        for user_id, registered_embedding in embeddings.items():

            similarity = cosine_similarity(
                embedding,
                registered_embedding,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_user = user_id

        return best_user, best_similarity

    def _build_result(
        self,
        user_id,
        similarity,
    ) -> VerificationResult:

        if similarity >= self.THRESHOLD:
            return VerificationResult(
                matched=True,
                user_id=user_id,
                similarity=similarity,
            )

        return VerificationResult(
            matched=False,
            user_id=None,
            similarity=similarity,
        )