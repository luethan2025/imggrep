DEFAULT_PHASH_CONFIG: dict[str, str | float | int | None] = {
    "model_id": None,
    "distance": 1,
    "top_k": 10,
    "num_threads": 4,
}

DEFAULT_CLIP_CONFIG: dict[str, str | float | int | None] = {
    "model_id": "openai/clip-vit-base-patch32",
    "distance": 0.1,
    "batch_size": 16,
    "top_k": 10,
    "dtype": "fp16",
}

DEFAULT_SIGLIP_CONFIG: dict[str, str | float | int | None] = {
    "model_id": "google/siglip-base-patch16-224",
    "distance": 0.1,
    "batch_size": 16,
    "top_k": 10,
    "dtype": "fp16",
}
