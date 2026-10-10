import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..embeddings.clip import CLIPEmbedder
from ..embeddings.phash import pHashEmbedder
from ..embeddings.siglip import SigLIPEmbedder
from .config import load_config
from .utils import list_files, resolve_input


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
    model_id = (
        args.model_id
        if args.model_id is not None
        else config.get("model_id")
    )
    if model_id is not None:
        match model_id:
            case "openai/clip-vit-base-patch32":
                embedder = CLIPEmbedder(model_id=model_id)
            case "google/siglip-base-patch16-224":
                embedder = SigLIPEmbedder(model_id=model_id)

    if isinstance(input, Path):
        embedder.set_reference_embeddings(Image.open(input))
    elif isinstance(input, str):
        embedder.set_reference_embeddings(input)
    else:
        raise ValueError("--input value is neither a Path or string")

    paths = list_files(args.target)

    matches = []
    for target in tqdm(
        paths,
        desc="Searching",
        bar_format="{desc}: |{bar}| {percentage:3.0f}% [{elapsed}]",
    ):
        if isinstance(input, str) or input.resolve() != target.resolve():
            try:
                is_similar = embedder.is_similar_to_reference_embeddings(
                    Image.open(target),
                    **(
                        {
                            "distance": (
                                args.distance
                                if args.distance is not None
                                else config["distance"]
                            )
                        }
                        if args.distance is not None or "distance" in config
                        else {}
                    ),
                )
                if is_similar:
                    matches.append(
                        target.resolve().relative_to(
                            Path.cwd(),
                            walk_up=True,
                        )
                    )
            except Exception as e:
                pass
