import argparse
import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..embeddings.clip import CLIPEmbedder
from ..embeddings.phash import pHashEmbedder
from ..embeddings.siglip import SigLIPEmbedder
from .config import load_config
from .utils import list_files, resolve_input

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="imggrep")
    parser.add_argument(
        "input",
        type=resolve_input,
        help="Path to image.",
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to target directory.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        choices=["openai/clip-vit-base-patch32", "google/siglip-base-patch16-224"],
        help="Embedding model. Must be one of: `openai/clip-vit-base-patch32`, `google/siglip-base-patch16-224`",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Maximum distance between two images.",
    )
    args = parser.parse_args()

    target = Path(args.target)
    if not (target.exists() and target.is_dir()):
        raise ValueError("`--target` value is not a directory")

    return args


def run_search() -> None:
    args = parse_args()
    config = load_config()

    input = args.input

    embedder = pHashEmbedder()
    model_id = args.model_id if args.model_id is not None else config.get("model_id")
    if model_id is not None:
        match model_id:
            case "openai/clip-vit-base-patch32":
                embedder = CLIPEmbedder(model_id=model_id)
            case "google/siglip-base-patch16-224":
                embedder = SigLIPEmbedder(model_id=model_id)

    if isinstance(input, Path):
        with Image.open(input) as image:
            image.load()
            embedder.set_reference_embeddings(image.copy())
    elif isinstance(input, str):
        embedder.set_reference_embeddings(input)
    else:
        raise TypeError("`--input` is neither a Path or string")

    paths = list_files(args.target)
    batch_size = config.get("batch_size", 16) if model_id is not None else 1
    matches = []
    batches = (
        paths[index : index + batch_size] for index in range(0, len(paths), batch_size)
    )
    for batch in tqdm(
        batches,
        total=(len(paths) + batch_size - 1) // batch_size,
        desc="Searching",
        bar_format="{desc}: |{bar}| {percentage:3.0f}% [{elapsed}]",
    ):
        images = []
        image_paths = []
        for target in batch:
            if isinstance(input, Path) and input.resolve() == target.resolve():
                continue
            try:
                with Image.open(target) as image:
                    image.load()
                    target_image = image.copy()
            except OSError as error:
                logger.warning("Skipping unreadable image %s: %s", target, error)
                continue
            images.append(target_image)
            image_paths.append(target)

        if not images:
            continue

        distance = (
            args.distance
            if args.distance is not None
            else config.get("distance", 10 if model_id is None else 0.1)
        )
        similarities = embedder.are_similar_to_reference_embeddings(
            images,
            distance,
        )
        for target, is_similar in zip(image_paths, similarities, strict=True):
            if is_similar:
                matches.append(
                    target.resolve().relative_to(
                        Path.cwd(),
                        walk_up=True,
                    )
                )
