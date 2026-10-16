from abc import ABC, abstractmethod
from typing import Any

import PIL


class Embedder(ABC):
    @abstractmethod
    def embed(self, img: PIL.Image.Image) -> Any:
        """Generate an embedding for an image.

        Args:
            img: The image to embed.

        Returns:
            The embedding representation of the image.
        """
        ...

    @abstractmethod
    def set_reference_embeddings(self, img: PIL.Image.Image) -> None:
        """Set the reference image used for similarity comparisons.

        Args:
            img: The reference image to embed and store.
        """
        ...

    @abstractmethod
    def distance_to_reference_embeddings(
        self, img: PIL.Image.Image | list[PIL.Image.Image]
    ) -> float | list[float]:
        """Calculate cosine distances between images and the reference embedding.

        Args:
            img: A single image or a batch of images to compare.

        Returns:
            A single distance for one image or a list of distances for a batch.
            Smaller distances indicate greater similarity.
        """
        ...

    @abstractmethod
    def are_similar_to_reference_embeddings(
        self,
        images: list[PIL.Image.Image],
        distance: float,
    ) -> list[tuple[int, float]]:
        """Find images whose distance from the reference is below a threshold.

        Args:
            images: The images to compare against the reference embedding.
            distance: The maximum distance threshold for a match.

        Returns:
            A list of (index, distance) tuples for matching images.
            Each index refers to the image's position in the input list.
        """
        ...
