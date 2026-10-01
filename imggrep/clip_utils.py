from .embedder import Embedder
from .torch_utils import get_device

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class CLIPEmbedder(Embedder):
    def __init__(self, fp16: bool = False) -> None:
        self.device = get_device()

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
        )

        self.reference_embeddings: torch.Tensor | None = None

    @torch.inference_mode()
    def embed(self, value: Image.Image | str) -> torch.Tensor:
        if isinstance(value, Image.Image):
            image = self.processor(
                images=value,
                return_tensors="pt",
            )["pixel_values"].to(self.device)
            embeddings = self.model.get_image_features(pixel_values=image)

        elif isinstance(value, str):
            text = self.processor(
                text=[value],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            embeddings = self.model.get_text_features(**text)

        else:
            raise TypeError(
                f"Expected PIL.Image.Image or str, got {type(value)}"
            )

        embeddings = getattr(embeddings, "pooler_output", embeddings)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings

    def set_reference_embeddings(
        self,
        value: Image.Image | str,
    ) -> None:
        self.reference_embeddings = self.embed(value)

    @torch.inference_mode()
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
