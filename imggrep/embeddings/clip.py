import PIL
import torch
from transformers import CLIPModel, CLIPProcessor

from .base import Embedder
from .utils import get_device


class CLIPEmbedder(Embedder):
    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        dtype: torch.dtype | None = None,
    ) -> None:
        self.device = get_device()
        self.dtype = dtype

        self.model = CLIPModel.from_pretrained(
            model_id,
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(
            model_id,
        )

        self.reference_embeddings: torch.Tensor | None = None

    @torch.inference_mode()
    def embed(self, value: PIL.Image.Image | str) -> torch.Tensor:
        if isinstance(value, PIL.Image.Image):
            return self._embed_images([value])

        elif isinstance(value, str):
            text = self.processor(
                text=[value],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.dtype is not None,
            ):
                embeddings = self.model.get_text_features(**text)

        else:
            raise TypeError(
                f"Unable to embed image. Expected PIL.Image.Image or str, got {type(value).__name__}"
            )

        embeddings = getattr(embeddings, "pooler_output", embeddings)
        embeddings = embeddings.float()
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings

    @torch.inference_mode()
    def _embed_images(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        image_batch = self.processor(
            images=images,
            return_tensors="pt",
        )["pixel_values"].to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.dtype is not None,
        ):
            embeddings = self.model.get_image_features(pixel_values=image_batch)
        embeddings = getattr(embeddings, "pooler_output", embeddings)
        embeddings = embeddings.float()
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def set_reference_embeddings(
        self,
        value: PIL.Image.Image | str,
    ) -> None:
        self.reference_embeddings = self.embed(value)

    @torch.inference_mode()
    def distance_to_reference_embeddings(
        self,
        value: PIL.Image.Image | str,
    ) -> float:
        if self.reference_embeddings is None:
            raise RuntimeError(
                "Reference embeddings have not been set. Call "
                "`set_reference_embeddings` first."
            )

        embeddings = self.embed(value)

        similarity = torch.cosine_similarity(
            self.reference_embeddings,
            embeddings,
            dim=-1,
        ).item()

        return 1.0 - similarity

    @torch.inference_mode()
    def are_similar_to_reference_embeddings(
        self,
        images: list[PIL.Image.Image],
        distance: float = 0.1,
    ) -> list[tuple[int, float]]:
        if self.reference_embeddings is None:
            raise RuntimeError(
                "Reference embeddings have not been set. Call "
                "`set_reference_embeddings` first."
            )

        embeddings = self._embed_images(images)

        similarities = torch.cosine_similarity(
            self.reference_embeddings,
            embeddings,
            dim=-1,
        )

        distances = 1.0 - similarities

        return [
            (index, dist)
            for index, dist in enumerate(distances.tolist())
            if dist < distance
        ]
