DEFAULT_PHASH_CONFIG: dict[str, str | float | None] = {
    "model_id": None,
    "distance": 1,
}

DEFAULT_CLIP_CONFIG: dict[str, str | float | None] = {
    "model_id": "openai/clip-vit-base-patch32",
    "distance": 0.1,
}

DEFAULT_SIGLIP_CONFIG: dict[str, str | float | None] = {
    "model_id": "google/siglip-base-patch16-224",
    "distance": 0.1,
}
