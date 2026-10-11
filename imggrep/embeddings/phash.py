import PIL
import imagehash

from .base import Embedder

class pHashEmbedder(Embedder):
    def __init__(self) -> None:
        self.reference_embeddings: imagehash.ImageHash | None = None

    def embed(self, img: PIL.Image.Image) -> imagehash.ImageHash:
        if not isinstance(img, PIL.Image.Image):
            raise TypeError(f"Unable to embed image. Expected PIL.Image.Image, got {type(img).__name__}")

        image_embeddings = imagehash.phash(img)
        return image_embeddings

    def set_reference_embeddings(
        self,
        img: PIL.Image.Image,
    ) -> None:
        self.reference_embeddings = self.embed(img)

    def is_similar_to_reference_embeddings(
        self,
        img: PIL.Image.Image,
        distance: int = 10,
    ) -> bool:
        if self.reference_embeddings is None:
            raise RuntimeError("Reference embeddings have not been set. Call `set_reference_embeddings` before calling `is_similar_to_reference_embeddings`.")

        image_embeddings = self.embed(img)
        return abs(self.reference_embeddings - image_embeddings) < distance
