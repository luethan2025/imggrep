from abc import ABC, abstractmethod
from typing import Any

import PIL


class Embedder(ABC):
    @abstractmethod
    def embed(self, img: PIL.Image.Image) -> Any: ...

    @abstractmethod
    def set_reference_embeddings(self, img: PIL.Image.Image) -> None: ...

    @abstractmethod
    def distance_to_reference_embeddings(
        self, img: PIL.Image.Image | list[PIL.Image.Image]
    ) -> float | list[float]: ...

    @abstractmethod
    def are_similar_to_reference_embeddings(
        self,
        images: list[PIL.Image.Image],
        distance: float,
    ) -> list[tuple[int, float]]: ...
