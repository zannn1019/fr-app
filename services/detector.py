import cv2
from insightface.app import FaceAnalysis

from config import (
    MIN_CONFIDENCE,
    PROVIDER,
    CTX_ID,
    DETECTION_SIZE,
    MODEL_NAME,
)
from entities.face import Face

class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(
            model_name=MODEL_NAME,
            providers=[PROVIDER]
        )
        self.app.prepare(
            ctx_id=CTX_ID,
            det_size=DETECTION_SIZE
        )

    def detect(self, frame):

        results = []

        for detected in self.app.get(frame):

            if detected.det_score < MIN_CONFIDENCE:
                continue

            results.append(
                Face(
                    bbox=detected.bbox,
                    confidence=float(detected.det_score),
                    embedding=detected.embedding,
                )
            )

        return results

    def draw_faces(self, frame, faces, result):
        for face in faces:
            border_color = (0, 255, 0) if result.user_id else (0, 0, 255)
            cv2.rectangle(
                frame,
                (face.x1, face.y1),
                (face.x2, face.y2),
                border_color,
                2
            )

            cv2.putText(
                frame,
                f"User ID: {result.user_id}" if result.user_id else f"Confidence: {result.similarity:.2f}",
                (face.x1, face.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                border_color,
                2,
            )