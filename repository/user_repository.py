from pathlib import Path

import cv2
import numpy as np

class UserRepository:
    def __init__(self):
        self.embedding_path = Path("storage/embeddings")
        self.portrait_path = Path("storage/portraits")

        self.embedding_path.mkdir(parents=True, exist_ok=True)
        self.portrait_path.mkdir(parents=True, exist_ok=True)

    def save_portrait(self, user_id: str, portrait):
        path = self.portrait_path / f"{user_id}.jpg"

        cv2.imwrite(
            str(path),
            portrait,
        )

    def save_embedding(self, user_id: str, embedding):
        path = self.embedding_path / f"{user_id}.npy"

        np.save(
            path,
            embedding,
        )

    def load_embedding(self, user_id: str):
        path = self.embedding_path / f"{user_id}.npy"

        if not path.exists():
            return None

        return np.load(path)

    def exists( self, user_id: str) -> bool:
        return (
            self.embedding_path /
            f"{user_id}.npy"
        ).exists()

    def load_all_embeddings(self):
        embeddings = {}

        for file in self.embedding_path.glob("*.npy"):
            user_id = file.stem
            embeddings[user_id] = np.load(file)

        return embeddings