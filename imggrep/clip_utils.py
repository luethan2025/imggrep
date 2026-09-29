from .embedder import Embedder
from .torch_utils import get_device

import torch
import clip
from PIL import Image

class CLIPEmbedder(Embedder):
    def __init__(self, fp16: bool = False) -> None:
        self.device = get_device()

        self.model, self.preprocess = clip.load(
            "ViT-B/32",
            device=self.device,
        )

        self.reference_embeddings: torch.Tensor | None = None

    @torch.no_grad()
    def embed(self, value: Image.Image | str) -> torch.Tensor:
        if isinstance(value, Image.Image):
            image = self.preprocess(value).unsqueeze(0).to(self.device)
            embeddings = self.model.encode_image(image)

        elif isinstance(value, str):
            text = clip.tokenize([value]).to(self.device)
            embeddings = self.model.encode_text(text)

        else:
            raise TypeError(
                f"Expected PIL.Image.Image or str, got {type(value)}"
            )

        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings

    def set_reference_embeddings(
        self,
        value: Image.Image | str,
    ) -> None:
        self.reference_embeddings = self.embed(value)

    @torch.no_grad()
    def is_similar_to_reference_embeddings(
        self,
        value: Image.Image | str,
        distance: float = 0.1,
    ) -> bool:
        if self.reference_embeddings is None:
            raise RuntimeError(
                "Must call `set_reference_embeddings` before "
                "`is_similar_to_reference_embeddings`"
            )

        embeddings = self.embed(value)

        similarity = torch.cosine_similarity(
            self.reference_embeddings,
            embeddings,
        ).item()

        return (1.0 - similarity) < distance
