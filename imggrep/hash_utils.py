from typing import Any

import imagehash
import PIL
from PIL import Image

from .embedder import Embedder

def phash(img: PIL.Image.Image) -> imagehash.ImageHash:
    hash_value = imagehash.phash(img)
    return hash_value

class pHashEmbedder(Embedder):
    def __init__(self) -> None:
        self.reference_hash = None

    def embed(self, img: PIL.Image.Image) -> int:
        image_embeddings = phash(img)
        return image_embeddings

    def set_reference_embeddings(self, img: PIL.Image.Image) -> None:
        self.reference_embeddings = self.embed(img)

    def is_similiar_to_reference_embeddings(self, img: PIL.Image.Image, distance: int=10) -> bool:
        if self.reference_embeddings is None:
            raise RuntimeError("Must call `set_reference_embeddings` before `is_similiar_to_reference_embeddings`")
        image_embeddings = self.embed(img)
        result = abs(self.reference_embeddings - image_embeddings) < distance
        return result
