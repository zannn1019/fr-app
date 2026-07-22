from insightface.app import FaceAnalysis
from typing import List

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