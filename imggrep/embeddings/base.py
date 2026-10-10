from abc import ABC, abstractmethod
from typing import Any

import PIL

class Embedder(ABC):
    @abstractmethod
    def embed(self, img: PIL.Image.Image) -> Any:
        ...

    @abstractmethod
    def set_reference_embeddings(self, img: PIL.Image.Image) -> None:
        ...

    @abstractmethod
    def is_similar_to_reference_embeddings(self, img: PIL.Image.Image) -> bool:
        ...
