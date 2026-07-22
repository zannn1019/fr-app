from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass(slots=True)
class Face:
    bbox: np.ndarray
    confidence: float
    embedding: Optional[np.ndarray] = None

    @property
    def x1(self):
        return int(self.bbox[0])

    @property
    def y1(self):
        return int(self.bbox[1])

    @property
    def x2(self):
        return int(self.bbox[2])

    @property
    def y2(self):
        return int(self.bbox[3])

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1