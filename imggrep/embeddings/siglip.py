import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

from .base import Embedder
from .utils import get_device


class SigLIPEmbedder(Embedder):
    def __init__(
        self, model_id: str = "google/siglip-base-patch16-224", fp16: bool = False
    ) -> None:
        self.device = get_device()

        self.model = SiglipModel.from_pretrained(
            model_id,
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = SiglipProcessor.from_pretrained(
            model_id,
        )

        self.reference_embeddings: torch.Tensor | None = None

    @torch.inference_mode()
    def embed(self, value: Image.Image | str) -> torch.Tensor:
        if isinstance(value, Image.Image):
            return self._embed_images([value])

        elif isinstance(value, str):
            text = self.processor(
                text=[value],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)
            embeddings = self.model.get_text_features(**text)

        else:
            raise TypeError(
                f"Unable to embed image. Expected PIL.Image.Image or str, got {type(value).__name__}"
            )

        embeddings = getattr(embeddings, "pooler_output", embeddings)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings

    @torch.inference_mode()
    def _embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        image_batch = self.processor(
            images=images,
            return_tensors="pt",
        )["pixel_values"].to(self.device)
        embeddings = self.model.get_image_features(pixel_values=image_batch)
        embeddings = getattr(embeddings, "pooler_output", embeddings)
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

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
                "Reference embeddings have not been set. Call `set_reference_embeddings` before calling `is_similar_to_reference_embeddings`."
            )

        embeddings = self.embed(value)

        similarity = torch.cosine_similarity(
            self.reference_embeddings,
            embeddings,
        ).item()
        return (1.0 - similarity) < distance

    @torch.inference_mode()
    def are_similar_to_reference_embeddings(
        self,
        images: list[Image.Image],
        distance: float,
    ) -> list[bool]:
        if self.reference_embeddings is None:
            raise RuntimeError(
                "Reference embeddings have not been set. Call `set_reference_embeddings` before calling `are_similar_to_reference_embeddings`."
            )

        embeddings = self._embed_images(images)
        similarities = torch.cosine_similarity(
            self.reference_embeddings,
            embeddings,
        )
        return [(1.0 - similarity) < distance for similarity in similarities.tolist()]
